"""Convert ONLY past GT body parameters to the current frame's cam0 gauge."""
import torch

from training.smpl_body import compute_gt_mesh_rot, compute_gt_mesh_translate


@torch.no_grad()
def prepare_gt_body_history(batch):
    """Call after current cameras are normalized, before model.forward.

    No current pose/beta/translation/has_smpl/identity is accessed. Raw SMPL
    translation is not pelvis position: compute_gt_mesh_translate accounts for
    the shape-dependent pelvis offset before the world -> current cam0 transform.
    """
    pose = batch["history_smpl_pose"].detach().float().clone()
    B, H, P, D = pose.shape
    if H != 2 or D != 72:
        raise ValueError("Expected history_smpl_pose [B,2,P,72]")
    pose[..., 66:] = 0
    valid = batch["history_valid"].bool()
    beta = batch["history_smpl_beta"].detach().float()
    trans = batch["history_smpl_trans"].detach().float()
    pose = torch.where(valid[..., None], pose, 0)
    beta = torch.where(valid[..., None], beta, 0)
    trans = torch.where(valid[..., None], trans, 0)
    past = {
        "smpl_pose": pose.reshape(B * H, P, 72),
        "smpl_beta": beta.reshape(B * H, P, 10),
        "smpl_trans": trans.reshape(B * H, P, 3),
        "smpl_gender": batch["history_smpl_gender"].reshape(B * H, P),
        # Both past poses use the SAME current camera gauge, even with moving cameras.
        "raw_extrinsics": batch["raw_extrinsics"].repeat_interleave(H, dim=0),
        "avg_scale": batch["avg_scale"].repeat_interleave(H, dim=0),
    }
    root_rotation = compute_gt_mesh_rot(past).reshape(B, H, P, 3)
    root_position = compute_gt_mesh_translate(past, normalize_cam=True, use_mamma=True).reshape(B, H, P, 3)
    pose[..., :3] = root_rotation
    return {
        "history_body_pose": torch.where(valid[..., None], pose[..., :66], 0),
        "history_body_beta": beta,
        "history_root_position": torch.where(valid[..., None], root_position, 0),
        "history_valid": valid,
        "history_frame_ids": batch["history_frame_ids"].detach(),
        "frame_ids": batch["frame_ids"].detach(),
        "temporal_num_frames": batch["temporal_num_frames"],
        "views_per_frame": batch["views_per_frame"],
    }
