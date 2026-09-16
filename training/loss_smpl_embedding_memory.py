"""Loss for causal SMPL embedding memory and its framewise spatial encoder."""

from training.loss import MultitaskLoss
from training.loss_smpl import compute_smpl_loss


class SMPLEmbeddingMemoryLoss(MultitaskLoss):
    """L = configured refined multitask loss + w_spatial * spatial SMPL loss.

    Refined outputs at all causal steps receive framewise supervision and the
    configured GT-relative motion losses. Spatial tokens receive independent
    pose/shape/translation/presence/joints/vertices supervision, ensuring useful
    embeddings even when history is detached. No GT embedding is an input.
    Spatial outputs have no mask head; their matching uses geometry/presence.
    Camera/mask and temporal losses are counted only in the refined branch.
    """

    def __init__(self, spatial_aux_weight=0.25, **kwargs):
        super().__init__(**kwargs)
        if spatial_aux_weight < 0:
            raise ValueError("spatial_aux_weight must be nonnegative")
        self.spatial_aux_weight = float(spatial_aux_weight)

    def forward(self, predictions, batch):
        if "spatial_smpl_outputs" not in predictions:
            raise ValueError("SMPLEmbeddingMemoryLoss requires the embedding-memory head")
        losses = super().forward(predictions, batch)
        if self.spatial_aux_weight:
            spatial = dict(predictions["spatial_smpl_outputs"])
            for key in ("pose_enc", "pose_enc_list"):
                if key in predictions:
                    spatial[key] = predictions[key]
            config = dict(self.smpl)
            config.update(use_temporal_training=False, weight_mask=0.0,
                          hungarian_cost_mask_weight=0.0)
            auxiliary = compute_smpl_loss(spatial, batch, **config)
            for key, value in auxiliary.items():
                losses[f"{key}_spatial"] = value
            weighted = self.spatial_aux_weight * config.get("weight", 1.0) * auxiliary["loss_smpl"]
        else:
            weighted = predictions["smpl_pose"].sum() * 0.0
        losses["loss_smpl_spatial_aux"] = weighted
        losses["objective"] = losses["objective"] + weighted
        losses["loss_objective"] = losses["objective"]
        losses["smpl_memory_valid_fraction"] = predictions["smpl_memory_valid"].float().mean().detach()
        return losses
