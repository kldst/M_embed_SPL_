"""Contracts for body-parameter conditioning, masking, and gradient flow."""
import unittest

import torch

from vggt.heads.smpl_gt_body_history_head import SMPLGTBodyHistoryHead, body_rotation_6d
from vggt.heads.smpl_multi_query_trans_rot_temporal_rel_head import SMPLMultiQueryTemporalRelConfig
from training.smpl_body import axis_angle_to_rotmat


class GTBodyHistoryTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        config = SMPLMultiQueryTemporalRelConfig(transformer_depth=1, transformer_dim=32,
            transformer_heads=4, transformer_dim_head=8, transformer_mlp_dim=64)
        self.head = SMPLGTBodyHistoryHead(dim_in=24, num_people=3, smpl_cfg=config,
            parameter_dim=16, memory_depth=1, history_dropout=0).eval()
        self.images = torch.randn(2, 2, 6, 24)
        self.inputs = {"history_body_pose": torch.randn(2, 2, 4, 66) * 0.2,
            "history_body_beta": torch.randn(2, 2, 4, 10),
            "history_root_position": torch.randn(2, 2, 4, 3),
            "history_valid": torch.ones(2, 2, 4, dtype=torch.bool),
            "history_frame_ids": torch.tensor([[10, 11], [21, 22]]),
            "frame_ids": torch.tensor([[12], [23]]), "temporal_num_frames": 1, "views_per_frame": 2}

    def predict(self, **overrides):
        return self.head([self.images], 1, {**self.inputs, **overrides})

    def test_rotation_representation_and_zero_gradients_are_finite(self):
        for pose in (torch.zeros(2, 66), torch.randn(2, 66)):
            pose.requires_grad_(True)
            encoded = body_rotation_6d(pose)
            reference = axis_angle_to_rotmat(pose)[..., :, :2].flatten(-2)
            torch.testing.assert_close(encoded, reference, atol=1e-6, rtol=1e-5)
            encoded.sum().backward()
            self.assertTrue(torch.isfinite(pose.grad).all())
        with self.assertRaisesRegex(ValueError, "66"):
            body_rotation_6d(torch.randn(1, 72))

    def test_history_changes_prediction_and_encoder_gets_gradients(self):
        output = self.predict()
        self.assertEqual(output["smpl_pose"].shape, (2, 3, 72))
        self.assertEqual(output["smpl_pose"][..., 66:].abs().sum(), 0)
        changed = self.predict(history_body_pose=self.inputs["history_body_pose"] + 0.3)
        self.assertGreater((changed["smpl_pose"] - output["smpl_pose"]).abs().max().item(), 1e-7)
        output["smpl_pose"].square().mean().backward()
        for module in (self.head.parameter_encoder, self.head.history_fusion):
            total = 0
            for p in module.parameters():
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
                total += float(p.grad.abs().sum())
            self.assertGreater(total, 0)

    def test_history_person_permutation_does_not_change_output(self):
        perm = torch.tensor([2, 0, 3, 1])
        changed = {k: v[:, :, perm] for k, v in self.inputs.items()
                   if k in ("history_body_pose", "history_body_beta", "history_root_position", "history_valid")}
        with torch.no_grad():
            torch.testing.assert_close(self.predict()["smpl_pose"], self.predict(**changed)["smpl_pose"],
                                       atol=1e-6, rtol=1e-5)

    def test_empty_history_and_nan_padding_fall_back(self):
        with torch.no_grad():
            output = self.predict(history_valid=torch.zeros(2, 2, 4, dtype=torch.bool),
                                  history_body_pose=torch.full((2, 2, 4, 66), float("nan")))
        self.assertTrue(torch.isfinite(output["smpl_pose"]).all())
        torch.testing.assert_close(output["smpl_pose"], output["spatial_smpl_outputs"]["smpl_pose"])

    def test_no_current_labels_are_consumed_and_future_is_rejected(self):
        with torch.no_grad():
            torch.testing.assert_close(self.predict()["smpl_pose"],
                self.predict(smpl_pose=torch.full((2, 4, 72), float("nan")),
                             has_smpl=torch.zeros(2, 4))["smpl_pose"])
        with self.assertRaisesRegex(ValueError, "PAST"):
            self.predict(history_frame_ids=torch.tensor([[10, 12], [21, 23]]))
        with self.assertRaisesRegex(ValueError, "ONLY current"):
            self.predict(temporal_num_frames=3)


if __name__ == "__main__":
    unittest.main()
