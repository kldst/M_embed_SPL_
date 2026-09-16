"""Current-frame body-only supervision for GT parameter history conditioning."""
from training.loss_smpl_embedding_memory import SMPLEmbeddingMemoryLoss


class SMPLGTBodyHistoryLoss(SMPLEmbeddingMemoryLoss):
    """Refined multitask + spatial auxiliary loss on the CURRENT frame only.

    With exact GT at t-1, matching (pred_t - GT_{t-1}) to (GT_t - GT_{t-1})
    reduces to current-frame error, so we do not double-count that as motion loss.
    The same invariance holds for bi-invariant SO(3) relative-rotation error.
    No reconstruction loss is necessary: current SMPL/geometry/mask losses train
    the parameter encoder end to end.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.smpl.get("use_temporal_training", False):
            raise ValueError("GT history predicts one frame; disable smpl.use_temporal_training")

    def forward(self, predictions, batch):
        if "temporal_shape" in batch:
            raise ValueError("GT body-history loss expects only current-frame targets")
        batch = dict(batch)
        pose = batch["smpl_pose"].clone()
        pose[..., 66:] = 0
        batch["smpl_pose"] = pose
        return super().forward(predictions, batch)
