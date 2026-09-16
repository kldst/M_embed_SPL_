"""Loss composition contract; real geometry is covered by the integration smoke."""
import unittest
from unittest.mock import patch

import torch

from training.loss_smpl_embedding_memory import SMPLEmbeddingMemoryLoss


class EmbeddingMemoryLossTest(unittest.TestCase):
    def test_auxiliary_weight_gradient_and_no_duplicate_temporal_mask(self):
        refined = torch.tensor(3.0, requires_grad=True)
        coarse = torch.tensor(2.0, requires_grad=True)
        config = {"weight": 2.0, "weight_mask": 1.0, "use_temporal_training": True,
                  "hungarian_cost_mask_weight": 1.0}
        criterion = SMPLEmbeddingMemoryLoss(smpl=config, spatial_aux_weight=0.25)
        predictions = {"smpl_pose": refined, "spatial_smpl_outputs": {"smpl_pose": coarse},
                       "smpl_memory_valid": torch.tensor([False, True])}
        with patch("training.loss.MultitaskLoss.forward", return_value={
            "objective": refined, "loss_objective": refined,
        }), patch("training.loss_smpl_embedding_memory.compute_smpl_loss", return_value={
            "loss_smpl": coarse,
        }) as auxiliary:
            result = criterion(predictions, {})
        self.assertEqual(result["objective"].item(), 4.0)
        self.assertIs(result["objective"], result["loss_objective"])
        result["objective"].backward()
        self.assertEqual(refined.grad.item(), 1.0)
        self.assertEqual(coarse.grad.item(), 0.5)
        kwargs = auxiliary.call_args.kwargs
        self.assertFalse(kwargs["use_temporal_training"])
        self.assertEqual(kwargs["weight_mask"], 0)
        self.assertEqual(kwargs["hungarian_cost_mask_weight"], 0)
        self.assertTrue(config["use_temporal_training"])
        self.assertEqual(config["weight_mask"], 1)

    def test_disabled_auxiliary_and_wrong_head(self):
        criterion = SMPLEmbeddingMemoryLoss(smpl={}, spatial_aux_weight=0)
        predictions = {"smpl_pose": torch.tensor(2.0), "spatial_smpl_outputs": {},
                       "smpl_memory_valid": torch.tensor([False])}
        with patch("training.loss.MultitaskLoss.forward", return_value={
            "objective": torch.tensor(1.0), "loss_objective": torch.tensor(1.0),
        }), patch("training.loss_smpl_embedding_memory.compute_smpl_loss") as auxiliary:
            result = criterion(predictions, {})
            auxiliary.assert_not_called()
            self.assertEqual(result["objective"].item(), 1)
        with self.assertRaisesRegex(ValueError, "requires"):
            criterion({"smpl_pose": torch.zeros(1)}, {})


if __name__ == "__main__":
    unittest.main()
