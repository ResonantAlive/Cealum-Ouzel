from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from model import GPT, ModelConfig


class LinearCrossEntropyTests(unittest.TestCase):
    def _model(self) -> GPT:
        return GPT(
            ModelConfig(
                vocab_size=64,
                block_size=16,
                n_layer=1,
                n_head=2,
                n_kv_head=1,
                n_embd=32,
                n_expert=2,
                expert_hidden_size=32,
                shared_expert_hidden_size=16,
                gradient_checkpointing=False,
                use_transformer_engine=False,
                use_te_rmsnorm=False,
                require_flash_attn_for_packing=False,
                linear_cross_entropy_backend="checkpointed",
                linear_cross_entropy_chunk_size=3,
            )
        )

    def test_checkpointed_linear_ce_matches_standard_loss_and_gradients(self) -> None:
        torch.manual_seed(7)
        model = self._model()
        targets = torch.tensor([1, 2, -100, 4, 5, 6, 7], dtype=torch.long)

        hidden_reference = torch.randn(7, 32, requires_grad=True)
        weight_reference = model.lm_head.weight.detach().clone().requires_grad_(True)
        reference = F.cross_entropy(
            F.linear(hidden_reference, weight_reference),
            targets,
            ignore_index=-100,
            reduction="sum",
        ) / targets.ne(-100).sum()
        reference.backward()

        hidden_checkpointed = hidden_reference.detach().clone().requires_grad_(True)
        model.lm_head.weight.grad = None
        actual = model._checkpointed_linear_cross_entropy(
            hidden_checkpointed,
            targets,
        )
        actual.backward()

        torch.testing.assert_close(actual, reference.detach(), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            hidden_checkpointed.grad,
            hidden_reference.grad,
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            model.lm_head.weight.grad,
            weight_reference.grad,
            rtol=1e-5,
            atol=1e-6,
        )


class ConfigurationSafetyTests(unittest.TestCase):
    def test_unknown_model_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown model configuration"):
            ModelConfig.from_dict({"n_layer": 1, "gradent_checkpointing": False})


if __name__ == "__main__":
    unittest.main()
