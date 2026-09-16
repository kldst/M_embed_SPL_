#!/usr/bin/env python
"""Real compose data -> full configured VGGT -> all losses -> backward -> AdamW.

Uses one 3-frame / 8-view / 518px clip. No model dimensions or active losses are
reduced. Pass --sequence to bound dataset discovery; all paths are recorded in
the JSON report. This checks execution/gradients, not trained model accuracy.
"""
import argparse
import fnmatch
import json
import os
from pathlib import Path
import random
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "training")]

import numpy as np
import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

from debug.debug_mamma_mask_dpt_gt_as_pred import (
    init_single_process_dist, move_to_device, process_batch_like_trainer,
)
from training.temporal import flatten_temporal_batch_for_framewise_model
from training.smpl_body import set_smplx_model_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="mamma_smpl_embedding_memory")
    parser.add_argument("--compose-root", type=Path, default=REPO.parent / "mamma_compose")
    parser.add_argument("--sequence", default="harmony4d_train_1_NC_200_00_contact/be_HsuS3iLSSWWZ_seq_000090")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--output", type=Path, default=REPO / "debug_outputs/smpl_embedding_memory_smoke")
    parser.add_argument("--random-init", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    started = time.time()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(8)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Full-resolution smoke test requires an available CUDA device")
    torch.cuda.set_device(device)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with initialize_config_dir(version_base=None, config_dir=str(REPO / "training/config")):
        cfg = compose(config_name=args.config)
    with open_dict(cfg):
        cfg.mamma_compose_root = str((args.compose_root / args.sequence).resolve())
        cfg.val_sequence_fraction = 0.0
        cfg.debug_max_sequences = 1
        cfg.debug_max_frames_per_sequence = 6
        cfg.data.train.shuffle = False
        cfg.data.train.common_config.fixed_view_sampling = True
        cfg.data.train.common_config.augs.color_jitter = None
        cfg.num_workers = 0
        cfg.max_img_per_gpu = 24
    OmegaConf.save(OmegaConf.to_container(cfg, resolve=True), out / "config_resolved.yaml")
    owns_dist = init_single_process_dist()
    report = {"config": args.config, "compose_root": cfg.mamma_compose_root,
              "device": torch.cuda.get_device_name(device), "steps": [], "passed": False}
    try:
        # Dataset decoding also uses these body models, before loss instantiation.
        set_smplx_model_root(cfg.loss.smplx_model_dir)
        print("Loading real compose clip", flush=True)
        dynamic = instantiate(cfg.data.train, _recursive_=False)
        raw = next(iter(dynamic.get_loader(epoch=0)))
        report["sequences"] = list(raw["seq_name"])
        report["frame_ids"] = raw["frame_ids"].tolist()
        report["input_shape"] = list(raw["images"].shape)
        assert report["input_shape"] == [1, 24, 3, 518, 518], report["input_shape"]
        # Trainer normalizes the collated CPU batch before moving it to CUDA.
        batch = move_to_device(process_batch_like_trainer(raw, cfg.scale_by_extrinsics), device)
        flat_batch = flatten_temporal_batch_for_framewise_model(batch)
        inputs = {key: batch[key] for key in ("views_per_frame", "temporal_num_frames", "frame_ids", "view_ids")}
        print("Building full configured model", flush=True)
        model = instantiate(cfg.model, _recursive_=False)
        if not args.random_init:
            checkpoint_path = cfg.checkpoint.resume_checkpoint_path
            print(f"Loading checkpoint: {checkpoint_path}", flush=True)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
            state = checkpoint.get("model", checkpoint)
            incompatible = model.load_state_dict(state, strict=False)
            report["checkpoint"] = {"path": checkpoint_path,
                "missing_keys": list(incompatible.missing_keys), "unexpected_keys": list(incompatible.unexpected_keys)}
            allowed = ("smpl_multi_query_trans_rot_head.memory_layers.",
                       "smpl_multi_query_trans_rot_head.time_embedding.")
            bad_missing = [key for key in incompatible.missing_keys if not key.startswith(allowed)]
            if bad_missing or incompatible.unexpected_keys:
                raise RuntimeError(f"Unexpected checkpoint incompatibility: {bad_missing}, {incompatible.unexpected_keys}")
            del checkpoint, state
        else:
            report["checkpoint"] = "random initialization explicitly requested"
        for name, parameter in model.named_parameters():
            if any(fnmatch.fnmatch(name, pattern) for pattern in cfg.optim.frozen_module_names):
                parameter.requires_grad_(False)
        model = model.to(device).train()
        loss_module = instantiate(cfg.loss, _recursive_=False).to(device)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=float(cfg.optim.optimizer.lr),
                                      weight_decay=float(cfg.optim.optimizer.weight_decay))
        report["trainable_parameters"] = sum(p.numel() for p in parameters)
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(device)
            print(f"Step {step}: forward", flush=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.optim.amp.enabled):
                prediction = model(images=batch["images"], smpl_inputs=inputs)
                print(f"Step {step}: loss", flush=True)
                losses = loss_module(prediction, flat_batch)
            for name, value in losses.items():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    raise RuntimeError(f"Nonfinite loss: {name}")
            assert prediction["smpl_pose"].shape == (3, 20, 72)
            assert prediction["person_mask_logits"].shape == (3, 8, 20, 518, 518)
            print(f"Step {step}: backward", flush=True)
            losses["objective"].backward()
            gradients = {}
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():
                        raise RuntimeError(f"Nonfinite gradient: {name}")
                    for group in ("memory_layers", "time_embedding", "decoder", "camera_head", "person_mask_head"):
                        if group in name:
                            gradients[group] = gradients.get(group, 0.0) + float(parameter.grad.detach().abs().sum())
            for group in ("memory_layers", "time_embedding", "decoder", "camera_head", "person_mask_head"):
                if gradients.get(group, 0) <= 0:
                    raise RuntimeError(f"No nonzero gradient in {group}")
            assert all(p.grad is None for p in model.aggregator.parameters())
            # Same per-module clipping limits as the training config.
            for module in (model.camera_head, model.smpl_multi_query_trans_rot_head, model.person_mask_head):
                torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0, error_if_nonfinite=True)
            tracked = model.smpl_multi_query_trans_rot_head.time_embedding.weight
            before = tracked.detach().clone()
            optimizer.step()
            assert not torch.equal(before, tracked.detach()), "Temporal parameters did not update"
            torch.cuda.synchronize(device)
            metrics = {name: float(value.detach()) for name, value in losses.items()
                       if torch.is_tensor(value) and value.numel() == 1}
            item = {"step": step, "losses": metrics, "gradient_abs_sums": gradients,
                    "output_shapes": {name: list(value.shape) for name, value in prediction.items() if torch.is_tensor(value)},
                    "history_valid_counts": prediction["smpl_memory_valid"].sum((1, 2)).tolist(),
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                    "optimizer_step_passed": True}
            report["steps"].append(item)
            print(json.dumps({"step": step, "objective": metrics["objective"],
                              "history_valid_counts": item["history_valid_counts"],
                              "peak_allocated_gib": item["peak_allocated_gib"]}), flush=True)
            del prediction, losses
        report["passed"] = True
    finally:
        report["elapsed_seconds"] = time.time() - started
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if owns_dist:
            dist.destroy_process_group()
    print(f"PASS: {out / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
