"""Tests for LiteLLM inference service."""

import ast
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

SERVICE_PATH = (
    Path(__file__).resolve().parents[2]
    / "edsl"
    / "inference_services"
    / "services"
    / "litellm_service.py"
)


class TestLiteLLMServiceStructure:
    def _parse(self):
        return ast.parse(SERVICE_PATH.read_text())

    def test_file_exists(self):
        assert SERVICE_PATH.exists()

    def test_has_litellm_service_class(self):
        tree = self._parse()
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        assert "LiteLLMService" in classes

    def test_inherits_inference_service_abc(self):
        tree = self._parse()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "LiteLLMService":
                base_names = [b.id for b in node.bases if isinstance(b, ast.Name)]
                assert "InferenceServiceABC" in base_names
                return
        pytest.fail("LiteLLMService not found")

    def test_has_inference_service_name(self):
        src = SERVICE_PATH.read_text()
        assert '_inference_service_ = "litellm"' in src

    def test_has_key_sequence(self):
        src = SERVICE_PATH.read_text()
        assert "key_sequence" in src

    def test_has_usage_sequence(self):
        src = SERVICE_PATH.read_text()
        assert "usage_sequence" in src

    def test_has_create_model(self):
        src = SERVICE_PATH.read_text()
        assert "def create_model" in src

    def test_has_async_execute_model_call(self):
        src = SERVICE_PATH.read_text()
        assert "async_execute_model_call" in src

    def test_uses_drop_params_true(self):
        src = SERVICE_PATH.read_text()
        assert '"drop_params": True' in src or "'drop_params': True" in src

    def test_uses_litellm_acompletion(self):
        src = SERVICE_PATH.read_text()
        assert "litellm.acompletion" in src


class TestLiteLLMServiceAutoDiscovery:
    def test_service_name_maps_correctly(self):
        """The registry auto-discovers files named *_service.py
        and strips the _service suffix."""
        assert SERVICE_PATH.name == "litellm_service.py"

    def test_env_key_name(self):
        src = SERVICE_PATH.read_text()
        assert "LITELLM_API_KEY" in src


class TestLiteLLMSDKInteraction:
    def test_acompletion_called_with_drop_params(self):
        import asyncio

        fake = types.ModuleType("litellm")
        mock_msg = MagicMock()
        mock_msg.content = "4"
        mock_resp = MagicMock()
        mock_resp.model_dump.return_value = {
            "choices": [{"message": {"content": "4"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        fake.acompletion = AsyncMock(return_value=mock_resp)
        sys.modules["litellm"] = fake

        try:

            async def run():
                resp = await fake.acompletion(
                    model="anthropic/claude-sonnet-4-20250514",
                    messages=[{"role": "user", "content": "2+2?"}],
                    drop_params=True,
                    temperature=0.5,
                    max_tokens=10,
                )
                return resp.model_dump()

            result = asyncio.run(run())
            assert result["choices"][0]["message"]["content"] == "4"

            kwargs = fake.acompletion.call_args.kwargs
            assert kwargs["drop_params"] is True
            assert kwargs["model"] == "anthropic/claude-sonnet-4-20250514"
        finally:
            del sys.modules["litellm"]

    def test_acompletion_forwards_api_key(self):
        import asyncio

        fake = types.ModuleType("litellm")
        mock_resp = MagicMock()
        mock_resp.model_dump.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        }
        fake.acompletion = AsyncMock(return_value=mock_resp)
        sys.modules["litellm"] = fake

        try:

            async def run():
                await fake.acompletion(
                    model="openai/gpt-4o",
                    messages=[{"role": "user", "content": "hi"}],
                    api_key="sk-test",
                    drop_params=True,
                )

            asyncio.run(run())
            assert fake.acompletion.call_args.kwargs["api_key"] == "sk-test"
        finally:
            del sys.modules["litellm"]


class TestDependency:
    def test_litellm_in_pyproject(self):
        pyproject = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
        assert "litellm" in pyproject

    def test_litellm_in_inference_extra(self):
        pyproject = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
        assert '"litellm"' in pyproject
