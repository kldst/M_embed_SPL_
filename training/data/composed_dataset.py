# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC
from collections import defaultdict
import logging
import time

from hydra.utils import instantiate
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
import bisect
from .dataset_util import *
from .track_util import *
from .augmentation import get_image_augmentation


class ComposedDataset(Dataset, ABC):
    """
    Composes multiple base datasets and applies common configurations.

    This dataset provides a flexible way to combine multiple base datasets while
    applying shared augmentations, track generation, and other processing steps.
    It handles image normalization, tensor conversion, and other preparations
    needed for training computer vision models with sequences of images.
    """
    def __init__(self, dataset_configs: dict, common_config: dict, **kwargs):
        """
        Initializes the ComposedDataset.

        Args:
            dataset_configs (dict): List of Hydra configurations for base datasets.
            common_config (dict): Shared configurations (augs, tracks, mode, etc.).
            **kwargs: Additional arguments (unused).
        """
        base_dataset_list = []

        # Instantiate each base dataset with common configuration
        for baseset_dict in dataset_configs:
            baseset = instantiate(baseset_dict, common_conf=common_config)
            base_dataset_list.append(baseset)

        # Use custom concatenation class that supports tuple indexing
        self.base_dataset = TupleConcatDataset(base_dataset_list, common_config)

        # --- Augmentation Settings ---
        # Controls whether to apply identical color jittering across all frames in a sequence
        self.cojitter = common_config.augs.cojitter
        # Probability of using shared jitter vs. frame-specific jitter
        self.cojitter_ratio = common_config.augs.cojitter_ratio
        # Initialize image augmentations (color jitter, grayscale, gaussian blur)
        self.image_aug = get_image_augmentation(
            color_jitter=common_config.augs.color_jitter,
            gray_scale=common_config.augs.gray_scale,
            gau_blur=common_config.augs.gau_blur,
        )

        # --- Optional Fixed Settings (useful for debugging) ---
        # Force each sequence to have exactly this many images (if > 0)
        self.fixed_num_images = common_config.fix_img_num
        # Force a specific aspect ratio for all images
        self.fixed_aspect_ratio = common_config.fix_aspect_ratio

        # --- Track Settings ---
        # Whether to include point tracks in the output
        self.load_track = common_config.load_track
        # Number of point tracks to include per sequence
        self.track_num = common_config.track_num

        # --- Mode Settings ---
        # Whether the dataset is being used for training (affects augmentations)
        self.training = common_config.training
        self.common_config = common_config
        self.defer_image_normalization = bool(
            getattr(common_config, "defer_image_normalization", False)
        )
        self.profile_data_loading = bool(
            getattr(common_config, "profile_data_loading", False)
        )
        self.profile_data_loading_every = max(
            1, int(getattr(common_config, "profile_data_loading_every", 10))
        )
        self._profile_totals = defaultdict(float)
        self._profile_samples = 0

        self.total_samples = len(self.base_dataset)

    def __len__(self):
        """Returns the total number of sequences in the dataset."""
        return self.total_samples


    def __getitem__(self, idx_tuple):
        """
        Retrieves a data sample (sequence) from the dataset.

        Loads raw data, converts to PyTorch tensors, applies augmentations,
        and prepares tracks if enabled.

        Args:
            idx_tuple (tuple): a tuple of (seq_idx, num_images, aspect_ratio)

        Returns:
            dict: A dictionary containing the sequence data (images, poses, tracks, etc.).
        """
        sample_start = time.perf_counter()

        # If fixed settings are provided, override the tuple values
        if self.fixed_num_images > 0:
            seq_idx = idx_tuple[0] if isinstance(idx_tuple, tuple) else idx_tuple
            idx_tuple = (seq_idx, self.fixed_num_images, self.fixed_aspect_ratio)

        # Retrieve the raw data batch from the appropriate base dataset
        batch = self.base_dataset[idx_tuple]
        timings = batch.pop("_profile_timings", {})
        seq_name = batch["seq_name"]

        # --- Data Conversion and Preparation ---
        # Convert numpy arrays to tensors
        image_start = time.perf_counter()
        image_array = np.stack(batch["images"])
        if self.defer_image_normalization:
            # Keep the compact uint8 representation through collation and H2D.
            # The trainer converts it to [0, 1] float on the GPU.
            images = torch.from_numpy(image_array).permute(0, 3, 1, 2).contiguous()
        else:
            images = torch.from_numpy(image_array.astype(np.float32)).contiguous()
            images = images.permute(0, 3, 1, 2).to(torch.get_default_dtype()).div(255)
        timings["image_stack"] = time.perf_counter() - image_start

        # Convert other data to tensors with appropriate types
        basic_tensor_start = time.perf_counter()
        extrinsics = torch.from_numpy(np.stack(batch["extrinsics"]).astype(np.float32))
        intrinsics = torch.from_numpy(np.stack(batch["intrinsics"]).astype(np.float32))
        ids = torch.from_numpy(batch["ids"])    # Frame indices sampled from the original sequence
        timings["basic_tensor_pack"] = time.perf_counter() - basic_tensor_start


        # --- Apply Color Augmentation (training mode only) ---
        augmentation_start = time.perf_counter()
        if self.training and self.image_aug is not None:
            if self.cojitter and random.random() > self.cojitter_ratio:
                # Apply the same color jittering transformation to all frames
                images = self.image_aug(images)
            else:
                # Apply different color jittering to each frame individually
                for aug_img_idx in range(len(images)):
                    images[aug_img_idx] = self.image_aug(images[aug_img_idx])
        timings["image_augmentation"] = time.perf_counter() - augmentation_start
        tensor_pack_start = time.perf_counter()


        # --- Prepare Final Sample Dictionary ---
        sample = {
            "seq_name": seq_name,
            "ids": ids,
            "images": images,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
        }

        optional_dense_fields = {
            "depths": np.float32,
            "cam_points": np.float32,
            "world_points": np.float32,
            "point_masks": None,
        }
        for key, dtype in optional_dense_fields.items():
            if key not in batch:
                continue
            value = np.stack(batch[key])
            if dtype is not None:
                value = value.astype(dtype)
            sample[key] = torch.from_numpy(value)

        # File/camera identifiers are debug metadata, not model inputs.  Keep a
        # canonical schema when requested so heterogeneous datasets can still be
        # collated together: older datasets call the paths ``image_paths`` while
        # Harmony4D calls them ``image_filenames``.  Mixed training can disable
        # the metadata entirely to avoid moving unused strings through workers.
        include_metadata = bool(
            getattr(self.common_config, "include_metadata", True)
        )
        if include_metadata:
            image_filenames = batch.get(
                "image_filenames", batch.get("image_paths")
            )
            if image_filenames is not None:
                sample["image_filenames"] = list(image_filenames)

        if "background_masks" in batch:
            sample["background_masks"] = torch.from_numpy(
                np.stack(batch["background_masks"]).astype(np.bool_)
            )

        if "frame_num" in batch:
            # 單一 sequence 的 frame_num 一個 scalar
            sample["frame_num"] = torch.as_tensor(batch["frame_num"], dtype=torch.long)

        if "views_per_frame" in batch:
            sample["views_per_frame"] = torch.as_tensor(batch["views_per_frame"], dtype=torch.long)

        if "temporal_num_frames" in batch:
            sample["temporal_num_frames"] = torch.as_tensor(batch["temporal_num_frames"], dtype=torch.long)

        for key in ("history_smpl_pose", "history_smpl_beta", "history_smpl_trans",
                    "history_valid", "history_smpl_gender", "history_frame_ids"):
            if key in batch:
                dtype = torch.long if key in ("history_frame_ids", "history_smpl_gender") else torch.float32
                sample[key] = torch.as_tensor(np.asarray(batch[key]), dtype=dtype)

        if "frame_ids" in batch:
            sample["frame_ids"] = torch.from_numpy(
                np.asarray(batch["frame_ids"]).astype(np.int64)
            )

        if "view_ids" in batch:
            sample["view_ids"] = torch.from_numpy(
                np.asarray(batch["view_ids"]).astype(np.int64)
            )

        if include_metadata:
            # ``ids`` is the stable local view index shared by all current
            # datasets.  Harmony4D additionally exposes a camera-name-derived
            # integer; use it when available and otherwise retain the local view
            # identity.  Camera IDs are intentionally only meaningful within a
            # sequence/dataset, never across MAMMA and Harmony4D.
            cam_ids = batch.get("cam_ids", batch.get("ids"))
            if cam_ids is not None:
                sample["cam_ids"] = torch.from_numpy(
                    np.asarray(cam_ids).astype(np.int64)
                )

        # original_sizes: list of [H, W] per view
        if "original_sizes" in batch:
            sample["original_sizes"] = torch.from_numpy(
                np.stack(batch["original_sizes"]).astype(np.float32)
            )  # [S, 2]

        # SMPL pose: [S, 72] numpy → [S, 72] tensor
        if "smpl_pose" in batch:
            sample["smpl_pose"] = torch.from_numpy(
                batch["smpl_pose"].astype(np.float32)
            )  # [S, 72]

        # SMPL beta: [S, 10] numpy → [S, 10] tensor
        if "smpl_beta" in batch:
            sample["smpl_beta"] = torch.from_numpy(
                batch["smpl_beta"].astype(np.float32)
            )  # [S, 10]

        # SMPL translation: [S, 3] numpy → [S, 3] tensor
        if "smpl_trans" in batch:
            sample["smpl_trans"] = torch.from_numpy(
                batch["smpl_trans"].astype(np.float32)
            )  # [S, 3]

        # SMPL joints 2D: [S, 24, 2]
        if "smpl_joints2d" in batch:
            sample["smpl_joints2d"] = torch.from_numpy(
                batch["smpl_joints2d"].astype(np.float32)
            )  # [S, 24, 2]

        # SMPL joints 2D confidence: [S, 24] (1=in image, 0=out of bounds)
        if "smpl_joints2d_confidence" in batch:
            sample["smpl_joints2d_confidence"] = torch.from_numpy(
                np.asarray(batch["smpl_joints2d_confidence"], dtype=np.float32)
            )

        # SMPL joints 3D world: [S, 24, 3]
        if "smpl_joints3d_world" in batch:
            sample["smpl_joints3d_world"] = torch.from_numpy(
                batch["smpl_joints3d_world"].astype(np.float32)
            )  # [S, 24, 3]

        # Dense-landmark GT: [S, P, 512, 2] normalised 2D + [S, P, 512] visibility
        if "smpl_landmarks2d" in batch:
            sample["smpl_landmarks2d"] = torch.from_numpy(
                batch["smpl_landmarks2d"].astype(np.float32)
            )
        if "smpl_landmarks2d_visibility" in batch:
            sample["smpl_landmarks2d_visibility"] = torch.from_numpy(
                batch["smpl_landmarks2d_visibility"].astype(np.float32)
            )
        # Per-person mask GT: [S, P, patch_h, patch_w] occupancy in [0,1]
        if "person_mask" in batch:
            sample["person_mask"] = torch.from_numpy(
                batch["person_mask"].astype(np.float32)
            )
        # MAMMA-style contact GT: [S, P, 512] binary person-person + floor contact
        if "smpl_contact" in batch:
            sample["smpl_contact"] = torch.from_numpy(batch["smpl_contact"].astype(np.float32))
        if "smpl_floor_contact" in batch:
            sample["smpl_floor_contact"] = torch.from_numpy(batch["smpl_floor_contact"].astype(np.float32))

        if "ball_3d" in batch:
            sample["ball_3d"] = torch.from_numpy(
                batch["ball_3d"].astype(np.float32)
            )

        if "ball_2d" in batch:
            sample["ball_2d"] = torch.from_numpy(
                batch["ball_2d"].astype(np.float32)
            )

        if "ball_2d_confidence" in batch:
            sample["ball_2d_confidence"] = torch.from_numpy(
                np.asarray(batch["ball_2d_confidence"], dtype=np.float32)
            )

        if "ball_valid" in batch:
            sample["ball_valid"] = torch.from_numpy(
                np.asarray(batch["ball_valid"], dtype=np.float32)
            )

        # SMPL gender: [S], 數值 1=male, 0=female
        if "smpl_gender" in batch:
            sample["smpl_gender"] = torch.from_numpy(
                batch["smpl_gender"].astype(np.float32)
            )  # [S]

        if "has_smpl" in batch:
            sample["has_smpl"] = torch.from_numpy(
                np.asarray(batch["has_smpl"], dtype=np.float32)
            )

        if "num_people" in batch:
            sample["num_people"] = torch.as_tensor(batch["num_people"], dtype=torch.long)

        if "person_keys" in batch:
            sample["person_keys"] = "|".join(str(key) for key in batch["person_keys"])

        # --- Track Processing (if enabled) ---
        if self.load_track:
            if batch["tracks"] is not None:
                # Use pre-computed tracks from the dataset
                tracks = torch.from_numpy(np.stack(batch["tracks"]).astype(np.float32))
                track_vis_mask = torch.from_numpy(np.stack(batch["track_masks"]).astype(bool))

                # Sample a subset of tracks randomly
                valid_indices = torch.where(track_vis_mask[0])[0]
                if len(valid_indices) >= self.track_num:
                    # If we have enough tracks, sample without replacement
                    sampled_indices = valid_indices[torch.randperm(len(valid_indices))][:self.track_num]
                else:
                    # If not enough tracks, sample with replacement (allow duplicates)
                    sampled_indices = valid_indices[torch.randint(0, len(valid_indices),
                                                    (self.track_num,),
                                                    dtype=torch.int64,
                                                    device=valid_indices.device)]

                # Extract the sampled tracks and their masks
                tracks = tracks[:, sampled_indices, :]
                track_vis_mask = track_vis_mask[:, sampled_indices]
                track_positive_mask = torch.ones(track_vis_mask.shape[1]).bool()

            else:
                # Generate tracks on-the-fly using depth information
                # This creates synthetic tracks based on the 3D information available
                tracks, track_vis_mask, track_positive_mask = build_tracks_by_depth(
                    extrinsics, intrinsics, world_points, depths, point_masks, images,
                    target_track_num=self.track_num, seq_name=seq_name
                )

            # Add track information to the sample dictionary
            sample["tracks"] = tracks
            sample["track_vis_mask"] = track_vis_mask
            sample["track_positive_mask"] = track_positive_mask

        timings["tensor_pack"] = time.perf_counter() - tensor_pack_start
        timings["sample_total"] = time.perf_counter() - sample_start
        if self.profile_data_loading:
            sample["_profile_sample_seconds"] = torch.tensor(
                timings["sample_total"], dtype=torch.float64
            )
        self._update_data_profile(timings)
        return sample

    def _update_data_profile(self, timings):
        if not self.profile_data_loading:
            return
        for key, value in timings.items():
            self._profile_totals[key] += float(value)
        self._profile_samples += 1
        if self._profile_samples < self.profile_data_loading_every:
            return

        count = self._profile_samples
        averages = {
            key: value / count for key, value in self._profile_totals.items()
        }
        total = max(averages.get("sample_total", 0.0), 1e-12)
        measured_dataset = sum(
            averages.get(key, 0.0)
            for key in (
                "view_read", "smpl_decode", "crop_resize", "view_other",
                "frame_pack", "temporal_combine",
            )
        )
        averages["dataset_other"] = max(
            0.0, averages.get("dataset_total", 0.0) - measured_dataset
        )
        measured_preprocess = sum(
            averages.get(key, 0.0)
            for key in (
                "image_copy", "crop_initial", "image_resize", "crop_final",
                "image_rotate", "dense_geometry",
            )
        )
        averages["preprocess_other"] = max(
            0.0, averages.get("crop_resize", 0.0) - measured_preprocess
        )
        measured_sample = sum(
            averages.get(key, 0.0)
            for key in (
                "dataset_total", "image_stack", "basic_tensor_pack",
                "image_augmentation", "tensor_pack",
            )
        )
        averages["composed_other"] = max(0.0, total - measured_sample)
        dataset_keys = (
            "view_read", "smpl_decode", "image_copy", "crop_initial",
            "image_resize", "crop_final", "image_rotate", "dense_geometry",
            "preprocess_other", "view_other", "frame_pack",
            "temporal_combine", "dataset_other",
        )
        composed_keys = (
            "image_stack", "basic_tensor_pack", "image_augmentation",
            "tensor_pack", "composed_other",
        )

        def _format(keys):
            return " | ".join(
                f"{key}={averages.get(key, 0.0):.3f}s "
                f"({100.0 * averages.get(key, 0.0) / total:.1f}%)"
                for key in keys
            )
        logging.info(
            "Data profile dataset (avg over %d clips): clip_total=%.3fs | %s",
            count, total, _format(dataset_keys),
        )
        logging.info(
            "Data profile compose (avg over %d clips): clip_total=%.3fs | %s",
            count, total, _format(composed_keys),
        )
        self._profile_totals.clear()
        self._profile_samples = 0


class TupleConcatDataset(ConcatDataset):
    """
    A custom ConcatDataset that supports indexing with a tuple.

    Standard PyTorch ConcatDataset only accepts an integer index. This class extends
    that functionality to allow passing a tuple like (sample_idx, num_images, aspect_ratio),
    where the first element is used to determine which sample to fetch, and the full
    tuple is passed down to the selected dataset's __getitem__ method.

    It also supports an option to randomly sample across all datasets, ignoring the
    provided index. This is useful during training when shuffling the entire dataset
    might cause memory issues due to duplicating dictionaries. If doing this, you can
    set pytorch's dataloader shuffle to False.
    """
    def __init__(self, datasets, common_config):
        """
        Initialize the TupleConcatDataset.

        Args:
            datasets (iterable): An iterable of PyTorch Dataset objects to concatenate.
            common_config (dict): Common configuration dict, used to check for random sampling.
        """
        super().__init__(datasets)
        # If True, ignores the input index and samples randomly across all datasets
        # This provides an alternative to dataloader shuffling for large datasets
        self.inside_random = common_config.inside_random

    def __getitem__(self, idx):
        """
        Retrieves an item using either an integer index or a tuple index.

        Args:
            idx (int or tuple): The index. If tuple, the first element is the sequence
                               index across the concatenated datasets, and the rest are
                               passed down. If int, it's treated as the sequence index.

        Returns:
            The item returned by the underlying dataset's __getitem__ method.

        Raises:
            ValueError: If the index is out of range or the tuple doesn't have exactly 3 elements.
        """
        idx_tuple = None
        if isinstance(idx, tuple):
            idx_tuple = idx
            idx = idx_tuple[0]  # Extract the sequence index

        # Override index with random value if inside_random is enabled
        if self.inside_random:
            total_len = self.cumulative_sizes[-1]
            idx = random.randint(0, total_len - 1)

        # Handle negative indices
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx

        # Find which dataset the index belongs to
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        # Create the tuple to pass to the underlying dataset
        if len(idx_tuple) == 3:
            idx_tuple = (sample_idx,) + idx_tuple[1:]
        else:
            raise ValueError("Tuple index must have exactly three elements")

        # Pass the modified tuple to the appropriate dataset
        return self.datasets[dataset_idx][idx_tuple]
