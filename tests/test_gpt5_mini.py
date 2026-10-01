"""Tests for the --gpt5-mini main-LLM switch."""
import pytest

from coding_agent.config import Config


def test_apply_gpt5_mini_reuses_vision_config():
    cfg = Config(
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        api_key="deepseek-key",
        temperature=0.0,
        vision_model_url="https://garyliufoundry1.services.ai.azure.com/openai/v1",
        vision_model_name="gpt-5-mini",
        vision_model_api_key="azure-key",
    )
    cfg.apply_gpt5_mini()
    assert cfg.base_url == "https://garyliufoundry1.services.ai.azure.com/openai/v1"
    assert cfg.model == "gpt-5-mini"
    assert cfg.api_key == "azure-key"
    assert cfg.temperature is None  # gpt-5 rejects temperature=0


def test_apply_gpt5_mini_requires_url():
    cfg = Config(vision_model_url="", vision_model_name="gpt-5-mini", vision_model_api_key="k")
    with pytest.raises(ValueError):
        cfg.apply_gpt5_mini()


def test_apply_gpt5_mini_default_model_name():
    # vision_model_name falls back to the default (gpt-5-mini) when unset
    cfg = Config(
        vision_model_url="https://x.openai.azure.com/openai/v1",
        vision_model_name="",
        vision_model_api_key="k",
    )
    cfg.apply_gpt5_mini()
    assert cfg.model == "gpt-5-mini"
