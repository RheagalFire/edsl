"""LiteLLM inference service for EDSL.

Routes to 100+ LLM providers (OpenAI, Anthropic, Google, Azure, Bedrock,
Ollama, etc.) via the litellm SDK. No proxy server needed.

Model strings use the provider/model format, e.g.
anthropic/claude-sonnet-4-20250514, azure/gpt-4o, openai/gpt-4o.

See https://docs.litellm.ai/docs/providers for all supported models.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import litellm

from ..inference_service_abc import InferenceServiceABC
from ..decorators import report_errors_async

if TYPE_CHECKING:
    from ...scenarios.file_store import FileStore as Files
    from ...invigilators.invigilator_base import InvigilatorBase as InvigilatorAI


class LiteLLMService(InferenceServiceABC):
    """LiteLLM inference service for 100+ LLM providers."""

    _inference_service_ = "litellm"
    _env_key_name_ = "LITELLM_API_KEY"
    _base_url_ = None

    key_sequence = ["choices", 0, "message", "content"]
    usage_sequence = ["usage"]
    input_token_name = "prompt_tokens"
    output_token_name = "completion_tokens"

    _models_list_cache: List[str] = []

    @classmethod
    def available(cls) -> List[str]:
        if cls._models_list_cache:
            return cls._models_list_cache

        try:
            models = [m for m in sorted(litellm.model_cost.keys()) if "/" in m]
            cls._models_list_cache = models[:500]
        except Exception:
            cls._models_list_cache = [
                "openai/gpt-4o",
                "openai/gpt-4o-mini",
                "anthropic/claude-sonnet-4-20250514",
                "anthropic/claude-haiku-4-5-20251001",
                "google/gemini-2.5-flash",
            ]
        return cls._models_list_cache

    @classmethod
    def create_model(
        cls,
        model_name: str = "openai/gpt-4o",
        model_class_name: str | None = None,
    ):
        from ...language_models import LanguageModel

        api_token = cls.get_api_token()

        class LLM(LanguageModel):
            key_sequence = cls.key_sequence
            usage_sequence = cls.usage_sequence
            input_token_name = cls.input_token_name
            output_token_name = cls.output_token_name
            _inference_service_ = cls._inference_service_

            @report_errors_async
            async def async_execute_model_call(
                self,
                user_prompt: str,
                system_prompt: str = "",
                question_name: Optional[str] = None,
                files_list: Optional[List["Files"]] = None,
                invigilator: Optional["InvigilatorAI"] = None,
                cache_key: Optional[str] = None,
                response_schema: Optional[dict] = None,
                response_schema_name: Optional[str] = None,
            ) -> Dict[str, Any]:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_prompt})

                params: Dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "top_p": self.top_p,
                    "drop_params": True,
                }

                if api_token:
                    params["api_key"] = api_token

                response = await litellm.acompletion(**params)
                return response.model_dump()

        LLM.__name__ = "LanguageModel"
        LLM.__qualname__ = "LanguageModel"
        return LLM
