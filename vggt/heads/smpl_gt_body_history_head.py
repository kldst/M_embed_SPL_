"""Current images + two GT body-parameter frames; no historical image tokens.

Only SMPL-X root + 21 body joints (66 axis-angle values), beta10 and pelvis xyz
are encoded. Historical people are aligned to each other by the loader. Current
queries retrieve from the entire history set; no target GT identity/pose enters
the forward path. All persons are permutation-equivariant in the history encoder.
"""
import torch
from torch import nn

from vggt.heads.smpl_multi_query_trans_rot_temporal_rel_head import SMPLMultiQueryTransRotTemporalRelHead


def body_rotation_6d(pose):
    """Continuous first-two-columns representation of 22 SO(3) matrices."""
    if pose.shape[-1] != 66:
        raise ValueError("Body encoder accepts exactly 66 pose values (root + 21 joints)")
    vector = pose.float().reshape(*pose.shape[:-1], 22, 3)
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(*x.shape, 3, 3)
    theta = torch.linalg.vector_norm(vector, dim=-1)[..., None, None]
    identity = torch.eye(3, device=pose.device, dtype=vector.dtype)
    rotation = identity + torch.sinc(theta / torch.pi) * skew + 0.5 * torch.sinc(theta / (2 * torch.pi)).square() * (skew @ skew)
    return rotation[..., :, :2].flatten(-2)


def masked_encode(encoder, tokens, valid):
    """Null token for empty groups; invalid entries never contribute as K/V."""
    N, _, D = tokens.shape
    has_any = valid.any(-1)
    tokens = torch.cat([tokens, tokens.new_zeros(N, 1, D)], dim=1)
    mask = torch.cat([~valid, has_any[:, None]], dim=1)
    return encoder(tokens, src_key_padding_mask=mask)[:, :-1]


class BodyParameterEncoder(nn.Module):
    def __init__(self, dim=256, heads=4, depth=1):
        super().__init__()
        self.dim = dim
        self.pose_mlp = nn.Sequential(nn.Linear(6, dim), nn.GELU(), nn.Linear(dim, dim))
        self.root_mlp = nn.Linear(6, dim)
        self.shape_mlp = nn.Linear(10, dim)
        self.position_mlp = nn.Linear(3, dim)
        self.part_embedding = nn.Embedding(24, dim)  # 22 rotations + beta + pelvis
        self.lag_embedding = nn.Embedding(2, dim)
        layer = lambda: nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout=0.0,
                                                    activation="gelu", batch_first=True, norm_first=True)
        self.spatial = nn.TransformerEncoder(layer(), depth, enable_nested_tensor=False)
        self.temporal = nn.TransformerEncoder(layer(), depth, enable_nested_tensor=False)

    def forward(self, pose, beta, position, valid, lags):
        B, H, P, _ = pose.shape
        if H != 2 or beta.shape != (B, H, P, 10) or position.shape != (B, H, P, 3):
            raise ValueError("Expected two aligned history frames with body66, beta10, xyz3")
        # Ignore arbitrary/NaN padded values before arithmetic, not only in attention.
        pose = torch.where(valid[..., None], pose, 0)
        beta = torch.where(valid[..., None], beta, 0)
        position = torch.where(valid[..., None], position, 0)
        rotations = body_rotation_6d(pose)
        root = self.root_mlp(rotations[..., :1, :])
        joints = self.pose_mlp(rotations[..., 1:, :])
        tokens = torch.cat([root, joints, self.shape_mlp(beta)[..., None, :],
                            self.position_mlp(position)[..., None, :]], dim=-2)
        tokens = tokens + self.part_embedding.weight
        token_valid = valid[..., None].expand(B, H, P, 24)
        tokens = masked_encode(self.spatial, tokens.reshape(B * H * P, 24, self.dim),
                               token_valid.reshape(B * H * P, 24)).reshape(B, H, P, 24, self.dim)
        time = self.lag_embedding((lags - 1).clamp(0, 1))[:, :, None, None]
        tokens = (tokens + time).permute(0, 2, 1, 3, 4).reshape(B * P, H * 24, self.dim)
        mask = token_valid.permute(0, 2, 1, 3).reshape(B * P, H * 24)
        tokens = masked_encode(self.temporal, tokens, mask)
        return tokens.reshape(B, P * H * 24, self.dim), mask.reshape(B, P * H * 24)


class BodyHistoryFusion(nn.Module):
    def __init__(self, query_dim, history_dim, heads, mlp_dim):
        super().__init__()
        self.norm_query = nn.LayerNorm(query_dim)
        self.norm_history = nn.LayerNorm(history_dim)
        self.attention = nn.MultiheadAttention(query_dim, heads, kdim=history_dim,
                                               vdim=history_dim, batch_first=True)
        self.gate = nn.Linear(2 * query_dim, query_dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2)
        self.norm_ff = nn.LayerNorm(query_dim)
        self.ff = nn.Sequential(nn.Linear(query_dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, query_dim))

    def forward(self, query, history, valid):
        B, _, D = history.shape
        has_history = valid.any(-1)
        history = torch.cat([self.norm_history(history), history.new_zeros(B, 1, D)], dim=1)
        mask = torch.cat([~valid, has_history[:, None]], dim=1)
        attended, _ = self.attention(self.norm_query(query), history, history,
                                     key_padding_mask=mask, need_weights=False)
        gate = self.gate(torch.cat([query, attended], dim=-1)).sigmoid()
        fused = query + gate * attended
        fused = fused + gate * self.ff(self.norm_ff(fused))
        return torch.where(has_history[:, None, None], fused, query)


class SMPLGTBodyHistoryHead(SMPLMultiQueryTransRotTemporalRelHead):
    def __init__(self, *, dim_in, num_people, smpl_cfg=None, parameter_dim=256,
                 parameter_depth=1, memory_depth=2, history_dropout=0.1):
        super().__init__(dim_in=dim_in, num_people=num_people, smpl_cfg=smpl_cfg)
        cfg = self.decoder.cfg
        if not 0 <= history_dropout < 1 or memory_depth < 1 or parameter_depth < 1:
            raise ValueError("Invalid GT body-history encoder configuration")
        self.history_dropout = history_dropout
        self.parameter_encoder = BodyParameterEncoder(parameter_dim, heads=4, depth=parameter_depth)
        self.history_fusion = nn.ModuleList([
            BodyHistoryFusion(cfg.transformer_dim, parameter_dim, cfg.transformer_heads, cfg.transformer_mlp_dim)
            for _ in range(memory_depth)
        ])

    def decode(self, tokens):
        full_pose = self.decoder.init_pose + self.decoder.decpose(tokens)
        # Preserve checkpoint/output shape, but never predict jaw/eyes/fingers.
        pose = torch.cat([full_pose[..., :66], torch.zeros_like(full_pose[..., 66:])], dim=-1)
        return {"smpl_pose": pose, "mesh_rot": pose[..., :3], "pred_pose_0": pose,
                "smpl_beta": self.decoder.init_betas + self.decoder.decshape(tokens),
                "mesh_translate": self.decoder.dectrans(tokens),
                "smpl_presence_logits": self.decoder.decpresence(tokens).squeeze(-1),
                "person_tokens": tokens}

    def forward(self, aggregated_tokens_list, patch_start_idx, smpl_inputs=None):
        inputs = smpl_inputs or {}
        if self._metadata_value(inputs, "temporal_num_frames", 1) != 1:
            raise ValueError("GT body-history mode accepts ONLY current-frame images (T=1)")
        patches = aggregated_tokens_list[-1][:, :, patch_start_idx:]
        B, V, N, C = patches.shape
        if self._metadata_value(inputs, "views_per_frame", V) != V:
            raise ValueError("Expected exactly views_per_frame current images")
        pose = inputs["history_body_pose"]
        valid = inputs["history_valid"].bool()
        if pose.shape[:3] != valid.shape or pose.shape[0] != B or pose.shape[1] != 2:
            raise ValueError("History batch/time/person dimensions must agree")
        times = inputs["history_frame_ids"].reshape(B, 2)
        current_time = inputs["frame_ids"].reshape(B, 1)
        if not torch.all(times[:, 0] < times[:, 1]) or not torch.all(times < current_time):
            raise ValueError("History must contain two strictly ordered PAST frame IDs")
        lags = current_time - times
        valid = valid & (lags <= 2)[..., None]
        if self.training and self.history_dropout:
            valid = valid & (torch.rand_like(valid.float()) >= self.history_dropout)
        memory, memory_valid = self.parameter_encoder(
            pose, inputs["history_body_beta"], inputs["history_root_position"], valid, lags,
        )
        spatial = self.decoder(patches.reshape(B, 1, V * N, C))[-1][:, 0]
        fused = spatial
        for layer in self.history_fusion:
            fused = layer(fused, memory, memory_valid)
        outputs = self.decode(fused)
        outputs["spatial_smpl_outputs"] = self.decode(spatial)
        outputs["smpl_memory_valid"] = valid.permute(0, 2, 1)
        return outputs
