"""LLM provider implementations for quantum katas benchmark."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import time

from .config import ModelConfig, ProviderType


@dataclass
class GenerationResult:
    """Result from an LLM generation."""

    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_id: str
    error: Optional[str] = None


class Provider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> GenerationResult:
        """Generate a response from the LLM."""
        pass

    @property
    def name(self) -> str:
        """Provider name for display."""
        return f"{self.config.provider.value}:{self.config.model_id}"


class AnthropicProvider(Provider):
    """Anthropic Claude provider."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

        self.client = Anthropic(api_key=config.api_key)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> GenerationResult:
        start_time = time.time()
        error = None
        content = ""
        input_tokens = 0
        output_tokens = 0

        try:
            kwargs = {
                "model": self.config.model_id,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            response = self.client.messages.create(**kwargs)
            content = response.content[0].text
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
        except Exception as e:
            error = str(e)

        latency_ms = (time.time() - start_time) * 1000
        return GenerationResult(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            model_id=self.config.model_id,
            error=error,
        )


class OpenAIProvider(Provider):
    """OpenAI GPT provider."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        kwargs = {"api_key": config.api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url

        self.client = OpenAI(**kwargs)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> GenerationResult:
        start_time = time.time()
        error = None
        content = ""
        input_tokens = 0
        output_tokens = 0

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.config.model_id,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=messages,
            )
            content = response.choices[0].message.content or ""
            if response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
        except Exception as e:
            error = str(e)

        latency_ms = (time.time() - start_time) * 1000
        return GenerationResult(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            model_id=self.config.model_id,
            error=error,
        )


class VLLMProvider(OpenAIProvider):
    """vLLM provider using OpenAI-compatible API."""

    def __init__(self, config: ModelConfig):
        if not config.base_url:
            raise ValueError("VLLMProvider requires base_url (e.g., http://localhost:8000/v1)")
        if not config.api_key:
            config.api_key = "dummy"  # vLLM doesn't require real API key
        super().__init__(config)


class GoogleProvider(Provider):
    """Google Gemini provider."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai package required: pip install google-generativeai")

        genai.configure(api_key=config.api_key)
        self.genai = genai
        self.model = genai.GenerativeModel(config.model_id)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> GenerationResult:
        start_time = time.time()
        error = None
        content = ""
        input_tokens = 0
        output_tokens = 0

        try:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            generation_config = self.genai.GenerationConfig(
                max_output_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config,
            )
            content = response.text
            if response.usage_metadata:
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count
        except Exception as e:
            error = str(e)

        latency_ms = (time.time() - start_time) * 1000
        return GenerationResult(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            model_id=self.config.model_id,
            error=error,
        )


class QiskitAssistantProvider(Provider):
    """IBM Qiskit Code Assistant provider."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            import requests
        except ImportError:
            raise ImportError("requests package required: pip install requests")

        self.requests = requests
        self.base_url = config.base_url or "https://qiskit-code-assistant.quantum.ibm.com"

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> GenerationResult:
        start_time = time.time()
        error = None
        content = ""
        input_tokens = 0
        output_tokens = 0

        try:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            response = self.requests.post(
                f"{self.base_url}/v1/completions",
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "prompt": full_prompt,
                    "model": self.config.model_id,
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()

            if "choices" in data and data["choices"]:
                content = data["choices"][0].get("text", "")
            if "usage" in data:
                input_tokens = data["usage"].get("prompt_tokens", 0)
                output_tokens = data["usage"].get("completion_tokens", 0)
        except Exception as e:
            error = str(e)

        latency_ms = (time.time() - start_time) * 1000
        return GenerationResult(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            model_id=self.config.model_id,
            error=error,
        )


def create_provider(config: ModelConfig) -> Provider:
    """Create a provider instance from configuration."""
    provider_map = {
        ProviderType.ANTHROPIC: AnthropicProvider,
        ProviderType.OPENAI: OpenAIProvider,
        ProviderType.GOOGLE: GoogleProvider,
        ProviderType.VLLM: VLLMProvider,
        ProviderType.QISKIT_ASSISTANT: QiskitAssistantProvider,
    }

    provider_class = provider_map.get(config.provider)
    if not provider_class:
        raise ValueError(f"Unsupported provider: {config.provider}")

    return provider_class(config)
