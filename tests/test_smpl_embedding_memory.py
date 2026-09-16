"""Causality, association, streaming equivalence and gradient contracts."""
import unittest

import torch

from vggt.heads.smpl_embedding_memory_head import SMPLEmbeddingMemoryHead
from vggt.heads.smpl_multi_query_trans_rot_temporal_rel_head import SMPLMultiQueryTemporalRelConfig


class EmbeddingMemoryTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        torch.set_num_threads(2)
        cfg = SMPLMultiQueryTemporalRelConfig(
            transformer_depth=1, transformer_heads=2, transformer_mlp_dim=32,
            transformer_dim_head=8, transformer_dim=16, max_T=8,
        )
        self.head = SMPLEmbeddingMemoryHead(
            dim_in=24, num_people=3, smpl_cfg=cfg, memory_depth=2,
            memory_dropout=0, presence_threshold=0, match_max_distance=1e3,
            match_max_cost=10,
        ).eval()
        self.tokens = torch.randn(2, 4, 2, 5, 24)

    def run_clip(self, tokens, **extra):
        B, T, V = tokens.shape[:3]
        return self.head([tokens.flatten(0, 1)], 1, {
            "temporal_num_frames": torch.full((B,), T),
            "views_per_frame": torch.full((B,), V),
            "frame_ids": torch.arange(T)[None].expand(B, -1), **extra,
        })

    def test_causal_and_exact_two_frame_horizon(self):
        with torch.no_grad():
            reference = self.run_clip(self.tokens)["smpl_pose"].reshape(2, 4, 3, 72)
            changed = self.tokens.clone()
            changed[:, 3] += 10 * torch.randn_like(changed[:, 3])
            result = self.run_clip(changed)["smpl_pose"].reshape(2, 4, 3, 72)
            torch.testing.assert_close(reference[:, :3], result[:, :3])
            changed = self.tokens.clone()
            changed[:, 0] += 10 * torch.randn_like(changed[:, 0])
            result = self.run_clip(changed)["smpl_pose"].reshape(2, 4, 3, 72)
            torch.testing.assert_close(reference[:, 3], result[:, 3])
            self.assertGreater((reference[:, 1] - result[:, 1]).abs().max().item(), 1e-7)

    def test_streaming_matches_clip_and_caches_spatial_tokens(self):
        with torch.no_grad():
            clip = self.run_clip(self.tokens)
            memory, outputs = None, []
            for t in range(4):
                result = self.run_clip(self.tokens[:, t:t + 1],
                                       frame_ids=torch.full((2, 1), t), smpl_memory=memory)
                memory = result["smpl_memory"]
                outputs.append(result["smpl_pose"])
            torch.testing.assert_close(torch.stack(outputs, 1).flatten(0, 1), clip["smpl_pose"],
                                       atol=2e-6, rtol=2e-5)
            spatial = clip["spatial_smpl_outputs"]["person_tokens"].reshape(2, 4, 3, 16)
            torch.testing.assert_close(memory["tokens"], spatial[:, -2:])
            self.assertFalse(memory["tokens"].requires_grad)

    def test_no_history_is_exact_spatial_fallback(self):
        self.head.presence_threshold = 1.0
        result = self.run_clip(self.tokens)
        self.assertFalse(result["smpl_memory_valid"].any())
        torch.testing.assert_close(result["smpl_pose"], result["spatial_smpl_outputs"]["smpl_pose"])
        self.assertTrue(torch.isfinite(result["smpl_pose"]).all())

    def test_association_handles_slot_permutation_and_rejection(self):
        current = {"tokens": torch.eye(3)[None], "translate": torch.zeros(1, 3, 3)}
        order = torch.tensor([2, 0, 1])
        history = {"tokens": current["tokens"][:, order], "translate": current["translate"].clone(),
                   "presence": torch.full((1, 3), 10.0)}
        index, valid = self.head.associate(current, history)
        self.assertEqual(index.tolist(), [[1, 2, 0]])
        self.assertTrue(valid.all())
        self.head.match_max_distance = 1
        history["translate"] += 100
        _, valid = self.head.associate(current, history)
        self.assertFalse(valid.any())

    def test_detach_history_and_temporal_gradients(self):
        for detach in (True, False):
            self.head.zero_grad(set_to_none=True)
            self.head.detach_history = detach
            tokens = self.tokens[:, :3].clone().requires_grad_(True)
            result = self.run_clip(tokens)
            result["smpl_pose"].reshape(2, 3, 3, 72)[:, -1].square().mean().backward()
            history_grad = tokens.grad[:, :2].abs().sum().item()
            self.assertEqual(history_grad == 0, detach)
            self.assertGreater(tokens.grad[:, 2].abs().sum().item(), 0)
            for name in ("time_embedding.weight", "memory_layers.0.attention.in_proj_weight",
                         "memory_layers.0.gate.weight"):
                grad = dict(self.head.named_parameters())[name].grad
                self.assertIsNotNone(grad, name)
                self.assertTrue(torch.isfinite(grad).all(), name)
                self.assertGreater(grad.abs().sum().item(), 0, name)

    def test_gaps_do_not_fabricate_history(self):
        result = self.run_clip(self.tokens[:, :3], frame_ids=torch.tensor([[0, 5, 9], [1, 6, 10]]))
        self.assertFalse(result["smpl_memory_valid"].any())
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            self.run_clip(self.tokens[:, :3], frame_ids=torch.tensor([[2, 1, 3], [0, 1, 2]]))


if __name__ == "__main__":
    unittest.main()
