"""Causal SMPL refinement from current images and two spatial person embeddings.

The spatial decoder runs with T=1, even during clip training. History therefore
never contains future images or recursively fused tokens. All causal outputs
are returned as [B*T,P,...] for the existing camera/SMPL/mask loss interface.
Streaming callers pass the returned ``smpl_memory`` explicitly, with frame_ids;
reset it on sequence, camera gauge, view ordering, or batch membership changes.
No GT parameters or GT identities are consumed by this module.
"""

import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

from vggt.heads.smpl_multi_query_trans_rot_temporal_rel_head import (
    SMPLMultiQueryTransRotTemporalRelHead,
)


class GatedMemoryLayer(nn.Module):
    def __init__(self, dim, heads, mlp_dim):
        super().__init__()
        self.norm_query = nn.LayerNorm(dim)
        self.norm_memory = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.gate = nn.Linear(2 * dim + 2, dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim))

    def forward(self, current, memory, valid, confidence):
        B, P, D = current.shape
        q = current.reshape(B * P, 1, D)
        mem = memory.reshape(B * P, 2, D)
        mask = valid.reshape(B * P, 2)
        has_history = mask.any(-1)
        # An unmasked null key only for empty histories prevents all-masked NaNs.
        mem = torch.cat([self.norm_memory(mem), mem.new_zeros(B * P, 1, D)], dim=1)
        padding = torch.cat([~mask, has_history[:, None]], dim=1)
        attended, _ = self.attention(self.norm_query(q), mem, mem,
                                     key_padding_mask=padding, need_weights=False)
        reliability = (confidence * valid).reshape(B * P, 1, 2).to(q.dtype)
        gate = self.gate(torch.cat([q, attended, reliability], dim=-1)).sigmoid()
        x = q + gate * attended
        x = x + self.ff(self.norm_ff(x))
        return torch.where(has_history[:, None, None], x, q).reshape(B, P, D)


class SMPLEmbeddingMemoryHead(SMPLMultiQueryTransRotTemporalRelHead):
    """Spatial checkpoint-compatible decoder plus a small two-frame memory head."""

    def __init__(self, *, dim_in, num_people, smpl_cfg=None, memory_depth=2,
                 detach_history=True, memory_dropout=0.1, presence_threshold=0.3,
                 match_max_distance=2.0, match_max_cost=1.5, match_translation_weight=0.25):
        super().__init__(dim_in=dim_in, num_people=num_people, smpl_cfg=smpl_cfg)
        cfg = self.decoder.cfg
        if memory_depth < 1 or not 0 <= memory_dropout < 1:
            raise ValueError("memory_depth must be positive and memory_dropout in [0,1)")
        if not 0 <= presence_threshold <= 1 or match_max_distance <= 0 or match_max_cost <= 0:
            raise ValueError("Invalid memory association thresholds")
        self.detach_history = bool(detach_history)
        self.memory_dropout = float(memory_dropout)
        self.presence_threshold = float(presence_threshold)
        self.match_max_distance = float(match_max_distance)
        self.match_max_cost = float(match_max_cost)
        self.match_translation_weight = float(match_translation_weight)
        self.time_embedding = nn.Embedding(2, cfg.transformer_dim)  # lag 1, lag 2
        nn.init.normal_(self.time_embedding.weight, std=0.02)
        self.memory_layers = nn.ModuleList([
            GatedMemoryLayer(cfg.transformer_dim, cfg.transformer_heads, cfg.transformer_mlp_dim)
            for _ in range(memory_depth)
        ])

    @torch.no_grad()
    def associate(self, current, history):
        """One-to-one, prediction-only matching, with explicit unmatched slots.

        Cost = cosine distance + translation_weight * distance in the common
        cam0 gauge. The default distance bound assumes scale_by_extrinsics=False.
        This is association for memory retrieval, not GT Hungarian loss matching.
        """
        B, P, _ = current["tokens"].shape
        indices = torch.zeros(B, P, device=current["tokens"].device, dtype=torch.long)
        valid = torch.zeros_like(indices, dtype=torch.bool)
        for b in range(B):
            features = F.normalize(current["tokens"][b].float(), dim=-1)
            previous = F.normalize(history["tokens"][b].float(), dim=-1)
            distance = torch.cdist(current["translate"][b].float(), history["translate"][b].float())
            cost = 1 - features @ previous.T + self.match_translation_weight * distance
            allowed = (history["presence"][b].sigmoid()[None] >= self.presence_threshold)
            allowed = allowed & (distance <= self.match_max_distance) & (cost < self.match_max_cost)
            cost = cost.masked_fill(~allowed, 1e6)
            # P dummy columns allow every current slot to reject all past slots.
            augmented = torch.cat([cost, cost.new_full((P, P), self.match_max_cost)], dim=1)
            rows, cols = linear_sum_assignment(augmented.cpu().numpy())
            keep = cols < P
            rows = torch.as_tensor(rows[keep], device=indices.device)
            cols = torch.as_tensor(cols[keep], device=indices.device)
            indices[b, rows] = cols
            valid[b, rows] = True
        return indices, valid

    def _refine(self, current, history):
        tokens = current["tokens"]
        B, P, D = tokens.shape
        memory = tokens.new_zeros(B, P, 2, D)
        confidence = tokens.new_zeros(B, P, 2)
        valid = torch.zeros(B, P, 2, device=tokens.device, dtype=torch.bool)
        assignments = torch.full((B, P, 2), -1, device=tokens.device, dtype=torch.long)
        for previous in history[-2:]:
            lag = current["frame_ids"] - previous["frame_ids"]
            indices, matched = self.associate(current, previous)
            past = previous["tokens"].detach() if self.detach_history else previous["tokens"]
            aligned = past.gather(1, indices[..., None].expand(B, P, D))
            score = previous["presence"].detach().sigmoid().gather(1, indices)
            for k in (1, 2):
                usable = matched & (lag == k)[:, None]
                if self.training and self.memory_dropout:
                    usable = usable & (torch.rand(B, P, device=tokens.device) >= self.memory_dropout)
                memory[:, :, k - 1] = torch.where(usable[..., None], aligned, memory[:, :, k - 1])
                confidence[:, :, k - 1] = torch.where(usable, score, confidence[:, :, k - 1])
                valid[:, :, k - 1] |= usable
                assignments[:, :, k - 1] = torch.where(usable, indices, assignments[:, :, k - 1])
        memory = memory + self.time_embedding.weight.to(tokens.dtype)[None, None]
        result = tokens
        for layer in self.memory_layers:
            result = layer(result, memory, valid, confidence)
        return result, valid, assignments

    def _decode_tokens(self, tokens):
        pose = self.decoder.init_pose + self.decoder.decpose(tokens)
        return {
            "smpl_pose": pose,
            "smpl_beta": self.decoder.init_betas + self.decoder.decshape(tokens),
            "mesh_translate": self.decoder.dectrans(tokens),
            "mesh_rot": pose[..., :3],
            "smpl_presence_logits": self.decoder.decpresence(tokens).squeeze(-1),
            "pred_pose_0": pose,
            "person_tokens": tokens,
        }

    def forward(self, aggregated_tokens_list, patch_start_idx, smpl_inputs=None):
        inputs = smpl_inputs or {}
        patch = aggregated_tokens_list[-1][:, :, patch_start_idx:]
        BT, V, N, C = patch.shape
        T = self._metadata_value(inputs, "temporal_num_frames", 1)
        if T < 1 or T > self.decoder.cfg.max_T or BT % T:
            raise ValueError(f"Invalid temporal shape: BT={BT}, T={T}")
        if self._metadata_value(inputs, "views_per_frame", V) != V:
            raise ValueError("views_per_frame does not match image tokens")
        B, P = BT // T, self.decoder.num_people
        # Crucial: independent spatial decode; no image context from other times.
        spatial = self.decoder(patch.reshape(BT, 1, V * N, C))
        tokens = spatial[-1].reshape(B, T, P, -1)
        coarse = self._decode_tokens(tokens.reshape(BT, P, -1))
        translations = coarse["mesh_translate"].reshape(B, T, P, 3)
        presence = coarse["smpl_presence_logits"].reshape(B, T, P)
        external = inputs.get("smpl_memory")
        frame_ids = inputs.get("frame_ids")
        if external is not None and frame_ids is None:
            raise ValueError("Streaming memory requires explicit frame_ids")
        if frame_ids is None:
            frame_ids = torch.arange(T, device=tokens.device)[None].expand(B, -1)
        else:
            frame_ids = torch.as_tensor(frame_ids, device=tokens.device).reshape(B, T)
        if T > 1 and not torch.all(frame_ids[:, 1:] > frame_ids[:, :-1]):
            raise ValueError("frame_ids must be strictly increasing within each clip")
        history = []
        if external is not None:
            H = external["tokens"].shape[1]
            if H > 2 or external["tokens"].shape != (B, H, P, tokens.shape[-1]):
                raise ValueError("smpl_memory has incompatible batch/person/feature dimensions")
            if external["tokens"].device != tokens.device:
                raise ValueError("smpl_memory must be on the current model device")
            if H and not torch.all(external["frame_ids"][:, -1] < frame_ids[:, 0]):
                raise ValueError("smpl_memory must precede current frames; reset memory for a new sequence")
            for h in range(H):
                history.append({key: external[key][:, h] for key in ("tokens", "translate", "presence", "frame_ids")})
        refined, validity, assignments = [], [], []
        for t in range(T):
            current = {"tokens": tokens[:, t], "translate": translations[:, t],
                       "presence": presence[:, t], "frame_ids": frame_ids[:, t]}
            result, valid, indices = self._refine(current, history)
            refined.append(result)
            validity.append(valid)
            assignments.append(indices)
            history = (history + [current])[-2:]
        output = self._decode_tokens(torch.stack(refined, dim=1).reshape(BT, P, -1))
        output["spatial_smpl_outputs"] = coarse
        output["smpl_memory_valid"] = torch.stack(validity, dim=1).reshape(BT, P, 2)
        output["smpl_memory_indices"] = torch.stack(assignments, dim=1).reshape(BT, P, 2)
        # Cache spatial E, never fused H. Explicit state avoids leakage across batches.
        output["smpl_memory"] = {
            key: torch.stack([item[key] for item in history], dim=1).detach()
            for key in ("tokens", "translate", "presence", "frame_ids")
        }
        return output
