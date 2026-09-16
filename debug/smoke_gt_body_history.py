#!/usr/bin/env python
"""Both real datasets: only current RGB, past body GT, geometry oracle, full train step."""
import argparse
import fnmatch
import json
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "training")]

import numpy as np
import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from debug.debug_mamma_mask_dpt_gt_as_pred import (
    init_single_process_dist, move_to_device, process_batch_like_trainer,
    make_gt_as_prediction, scalar_metrics, save_projection_overlays,
)
from training.gt_body_history import prepare_gt_body_history
from training.data.datasets.sys_smpl_multi import SysSMPLMultiDataset
from training.smpl_body import (_decode_smpl_batch, set_smplx_model_root,
                               _project_points_opencv, scale_joints_to_batch_gauge)


@torch.no_grad()
def check_history_geometry(batch, inputs):
    """Independent world-parameter vs cam0-pelvis reconstruction and projection."""
    B, H, P, _ = batch["history_smpl_pose"].shape
    genders = [{0: "male", 1: "female", 2: "neutral"}[int(g)]
               for g in batch["history_smpl_gender"].flatten().tolist()]
    pose = batch["history_smpl_pose"].clone()
    pose[..., 66:] = 0
    beta = batch["history_smpl_beta"].reshape(B * H * P, 10)
    world_j, world_v = _decode_smpl_batch(pose.reshape(-1, 72), beta,
        batch["history_smpl_trans"].reshape(-1, 3), genders, use_mamma=True)
    encoded_pose = torch.nn.functional.pad(inputs["history_body_pose"], (0, 6))
    local_j, local_v = _decode_smpl_batch(encoded_pose.reshape(-1, 72), beta,
        torch.zeros(B * H * P, 3, device=pose.device), genders, use_mamma=True)
    scale = batch["avg_scale"].repeat_interleave(H)
    local_j = scale_joints_to_batch_gauge(local_j.reshape(B * H, P, -1, 3), scale)
    local_v = scale_joints_to_batch_gauge(local_v.reshape(B * H, P, -1, 3), scale)
    offset = inputs["history_root_position"].reshape(B * H, P, 3) - local_j[..., 0, :]
    local_j, local_v = local_j + offset[..., None, :], local_v + offset[..., None, :]
    raw = batch["raw_extrinsics"].repeat_interleave(H, 0)
    cameras = batch["extrinsics"].repeat_interleave(H, 0)
    intrinsics = batch["intrinsics"].repeat_interleave(H, 0)
    valid = batch["history_valid"].reshape(B * H, P).bool()
    metrics = {}
    for label, world, local in (("joints", world_j, local_j), ("vertices", world_v, local_v)):
        world = world.reshape(B * H, P, -1, 3)
        reference = (world @ raw[:, :1, :3, :3].transpose(-1, -2)
                     + raw[:, 0, :3, 3][:, None, None]) / scale[:, None, None, None]
        error = torch.linalg.vector_norm(reference - local, dim=-1)[valid]
        metrics[f"{label}_gauge_max"] = float(error.max())
        assert metrics[f"{label}_gauge_max"] < 1e-4, metrics
        # Keep all joints; sample vertices for a cheap independent projection check.
        world = world[..., ::max(1, world.shape[-2] // 500), :]
        local = local[..., ::max(1, local.shape[-2] // 500), :]
        V, N = raw.shape[1], world.shape[-2]
        project = lambda points, ext: _project_points_opencv(
            points.flatten(1, 2)[:, None].expand(-1, V, -1, -1), ext, intrinsics,
        ).reshape(B * H, V, P, N, 2)
        difference = torch.linalg.vector_norm(project(world, raw) - project(local, cameras), dim=-1)
        selected = difference[valid[:, None, :, None].expand_as(difference)]
        metrics[f"{label}_projection_max_px"] = float(selected.max())
        assert metrics[f"{label}_projection_max_px"] < 0.05, metrics
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="mamma_smpl_embedding_memory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=1, help="Steps per dataset on one real batch")
    parser.add_argument("--output", type=Path, default=REPO / "debug_outputs/gt_body_history_smoke")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    torch.set_num_threads(8)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    started = time.time()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    with initialize_config_dir(version_base=None, config_dir=str(REPO / "training/config")):
        cfg = compose(config_name=args.config)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    set_smplx_model_root(cfg.loss.smplx_model_dir)
    owns_dist = init_single_process_dist()
    report = {"passed": False, "config": args.config, "sources": [], "device": torch.cuda.get_device_name(device)}
    try:
        print("Building model and loading pretrained spatial weights", flush=True)
        model = instantiate(cfg.model, _recursive_=False)
        ckpt = torch.load(cfg.checkpoint.resume_checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
        incompatible = model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        allowed = ("smpl_multi_query_trans_rot_head.parameter_encoder.", "smpl_multi_query_trans_rot_head.history_fusion.")
        bad = [key for key in incompatible.missing_keys if not key.startswith(allowed)]
        assert not bad and not incompatible.unexpected_keys, (bad, incompatible.unexpected_keys)
        report["checkpoint_missing_new_keys"] = list(incompatible.missing_keys)
        del ckpt
        for name, parameter in model.named_parameters():
            if any(fnmatch.fnmatch(name, pattern) for pattern in cfg.optim.frozen_module_names):
                parameter.requires_grad_(False)
        model.to(device).train()
        criterion = instantiate(cfg.loss, _recursive_=False).to(device)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.optim.optimizer.lr)
        for source, dataset_config in zip(("mamma", "harmony4d"), cfg.data.train.dataset.dataset_configs):
            print(f"{source}: load current images + past GT", flush=True)
            loader_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train, resolve=True))
            selected = OmegaConf.create(OmegaConf.to_container(dataset_config, resolve=True))
            selected.max_sequences, selected.max_frames_per_sequence = 1, 4
            selected.val_sequence_fraction = 0.0
            loader_cfg.dataset.dataset_configs = [selected]
            loader_cfg.max_img_per_gpu = 8
            loader_cfg.num_workers = 0
            loader_cfg.shuffle = False
            loader_cfg.common_config.fixed_view_sampling = True
            loader_cfg.common_config.augs.color_jitter = None
            OmegaConf.save(loader_cfg, out / f"{source}_loader_resolved.yaml")
            dynamic = instantiate(loader_cfg, _recursive_=False)
            reads = []
            original_read = SysSMPLMultiDataset._load_view
            def record_read(instance, annotation, *a, **kw):
                reads.append(annotation["image_path"])
                return original_read(instance, annotation, *a, **kw)
            with patch.object(SysSMPLMultiDataset, "_load_view", record_read):
                raw = next(iter(dynamic.get_loader(epoch=0)))
            assert raw["images"].shape == (1, 8, 3, 518, 518)
            assert len(reads) == 8, f"Read {len(reads)} images, expected only current 8 views"
            assert all(SysSMPLMultiDataset._numeric_frame_id(Path(path).name) == int(raw["frame_ids"][0, 0])
                       for path in reads), reads
            # Exercise the REAL Trainer preprocessing and _step (without a long training run).
            from training.trainer import Trainer
            trainer = Trainer.__new__(Trainer)
            trainer.data_conf, trainer.normalize_cam = cfg.data, True
            trainer.scale_by_extrinsics = cfg.scale_by_extrinsics
            processed = trainer._process_batch(raw)
            inputs_cpu = prepare_gt_body_history(processed)
            changed = dict(processed)
            for key in ("smpl_pose", "smpl_beta", "smpl_trans", "has_smpl"):
                changed[key] = torch.full_like(changed[key], float("nan"))
            clean = prepare_gt_body_history(changed)
            for key in inputs_cpu:
                torch.testing.assert_close(inputs_cpu[key], clean[key])
            batch = move_to_device(processed, device)
            inputs = move_to_device(inputs_cpu, device)
            item = {"source": source, "sequence": raw["seq_name"], "current_frame_ids": raw["frame_ids"].tolist(),
                    "history_frame_ids": raw["history_frame_ids"].tolist(), "image_reads": len(reads),
                    "input_shape": list(raw["images"].shape), "no_target_leakage": True,
                    "history_geometry": check_history_geometry(batch, inputs), "steps": []}
            item["current_gt_reprojection"] = save_projection_overlays(batch, out / source / "gt_reprojection")
            assert item["current_gt_reprojection"]["joint_reprojection_max_px"] < 0.05, item
            oracle = make_gt_as_prediction(batch, cfg)
            oracle["spatial_smpl_outputs"] = dict(oracle)
            oracle["smpl_memory_valid"] = inputs["history_valid"].permute(0, 2, 1)
            with torch.no_grad():
                oracle_loss = criterion(oracle, batch)
            item["gt_as_pred_losses"] = scalar_metrics(oracle_loss)
            for key in ("loss_mesh_translate", "loss_smpl_joints2d", "loss_smpl_joints3d", "loss_smpl_vertices"):
                assert abs(item["gt_as_pred_losses"][key]) < 1e-4, (key, item["gt_as_pred_losses"][key])
            del oracle, oracle_loss
            trainer.loss, trainer.steps = criterion, {"train": 0}
            trainer._update_and_log_scalars = lambda *a, **kw: None
            trainer._log_tb_visuals = lambda *a, **kw: None
            captured = {}
            handle = model.register_forward_hook(lambda module, arguments, result: captured.update(result))
            for step in range(args.steps):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats(device)
                print(f"{source}: full forward / losses / backward, step {step}", flush=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    losses = trainer._step(batch, model, "train", {})
                for key, value in losses.items():
                    if torch.is_tensor(value):
                        assert torch.isfinite(value).all(), key
                assert captured["smpl_pose"].shape == (1, 20, 72)
                assert captured["person_mask_logits"].shape == (1, 8, 20, 518, 518)
                assert not captured["smpl_pose"][..., 66:].any()
                losses["objective"].backward()
                gradients = {}
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        assert torch.isfinite(parameter.grad).all(), name
                        for group in ("parameter_encoder", "history_fusion", "decoder", "camera_head", "person_mask_head"):
                            if group in name:
                                gradients[group] = gradients.get(group, 0) + float(parameter.grad.abs().sum())
                assert all(gradients.get(g, 0) > 0 for g in ("parameter_encoder", "history_fusion", "decoder", "camera_head", "person_mask_head")), gradients
                assert all(p.grad is None for p in model.aggregator.parameters())
                for module in (model.camera_head, model.smpl_multi_query_trans_rot_head, model.person_mask_head):
                    torch.nn.utils.clip_grad_norm_(module.parameters(), 1, error_if_nonfinite=True)
                target = model.smpl_multi_query_trans_rot_head.parameter_encoder.pose_mlp[0].weight
                before = target.detach().clone()
                optimizer.step()
                assert not torch.equal(before, target.detach())
                item["steps"].append({"step": step, "losses": scalar_metrics(losses),
                    "gradient_abs_sums": gradients, "encoder_updated": True,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30})
                print(f"{source}: objective={float(losses['objective']):.6f}; all gradients finite", flush=True)
                captured.clear()
                del losses
            handle.remove()
            report["sources"].append(item)
            print(json.dumps({"source": source, "geometry": item["history_geometry"],
                              "current_reprojection": item["current_gt_reprojection"]}), flush=True)
        report["passed"] = True
    finally:
        report["elapsed_seconds"] = time.time() - started
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if owns_dist:
            dist.destroy_process_group()
    print(f"PASS: {out / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
