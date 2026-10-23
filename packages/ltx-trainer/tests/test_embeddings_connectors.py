"""The Gemma connectors are frozen during training: ``apply_embeddings_connectors`` records no autograd graph
through them, in either cached caption format, so the backward pass stops at the transformer's inputs."""

import pytest
import torch

from ltx_core.text_encoders.gemma.embeddings_connector import Embeddings1DConnector
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor
from ltx_trainer.trainer import apply_embeddings_connectors

BATCH, TOKENS, DIM = 2, 1024, 16
REAL_TOKENS = 200


def _processor() -> EmbeddingsProcessor:
    def connector() -> Embeddings1DConnector:
        return Embeddings1DConnector(attention_head_dim=8, num_attention_heads=2, num_layers=1)

    return EmbeddingsProcessor(video_connector=connector(), audio_connector=connector()).to(torch.bfloat16)


def _conditions(legacy: bool) -> dict[str, torch.Tensor]:
    mask = torch.zeros(BATCH, TOKENS, dtype=torch.int64)
    mask[:, -REAL_TOKENS:] = 1
    features = torch.randn(BATCH, TOKENS, DIM, dtype=torch.bfloat16)
    if legacy:
        return {"prompt_embeds": features, "prompt_attention_mask": mask}
    return {"video_prompt_embeds": features, "audio_prompt_embeds": features.clone(), "prompt_attention_mask": mask}


@pytest.mark.parametrize("legacy", [False, True])
def test_connectors_run_without_autograd(legacy: bool) -> None:
    torch.manual_seed(0)
    processor = _processor()  # parameters still trainable, as freshly loaded
    assert all(p.requires_grad for p in processor.parameters())

    conditions = apply_embeddings_connectors(processor, _conditions(legacy))

    assert set(conditions) >= {"video_prompt_embeds", "audio_prompt_embeds", "prompt_attention_mask"}
    for key in ("video_prompt_embeds", "audio_prompt_embeds"):
        assert conditions[key].shape[:2] == (BATCH, TOKENS)
        assert conditions[key].grad_fn is None
        assert not conditions[key].requires_grad

    # A downstream loss on the embeddings reaches no connector parameter.
    downstream = torch.nn.Linear(conditions["video_prompt_embeds"].shape[-1], 1, dtype=torch.bfloat16)
    downstream(conditions["video_prompt_embeds"]).float().mean().backward()
    assert all(p.grad is None for p in processor.parameters())
    assert downstream.weight.grad is not None
