"""LLM provider implementations for quantum katas benchmark."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional
import time

from .config import ModelConfig, ProviderType

# Identifier for requests from this benchmark
USER_AGENT = "quantum-katas-benchmark/0.1.0"
X_CALLER = "quantum-katas-benchmark"

# Retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0  # seconds
DEFAULT_RETRY_MULTIPLIER = 2.0  # exponential backoff multiplier


def resolve_header_value(value: str) -> str:
    """Resolve header value, substituting environment variables.

    If the value matches an environment variable name, return the env var value.
    This allows configs like {"API_KEY": "VLLM_API_KEY"} to resolve to the actual key.
    """
    import os
    env_value = os.environ.get(value)
    if env_value:
        return env_value
    return value

# Error patterns that trigger retries
RETRYABLE_ERROR_PATTERNS = [
    "rate limit",
    "rate_limit",
    "429",
    "too many requests",
    "overloaded",
    "capacity",
    "500",
    "502",
    "503",
    "504",
    "internal server error",
    "bad gateway",
    "service unavailable",
    "gateway timeout",
    "timeout",
    "timed out",
    "connection error",
    "connection reset",
]


def is_retryable_error(error: str) -> bool:
    """Check if an error message indicates a retryable condition."""
    error_lower = error.lower()
    return any(pattern in error_lower for pattern in RETRYABLE_ERROR_PATTERNS)


@dataclass
class GenerationResult:
    """Result from an LLM generation."""

    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_id: str
    error: Optional[str] = None
    retries: int = 0


class Provider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(
        self,
        config: ModelConfig,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        retry_multiplier: float = DEFAULT_RETRY_MULTIPLIER,
    ):
        self.config = config
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_multiplier = retry_multiplier

    @abstractmethod
    def _generate_impl(self, prompt: str, system_prompt: Optional[str] = None) -> GenerationResult:
        """Implementation of generate - subclasses override this."""
        pass

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> GenerationResult:
        """Generate a response with automatic retries for transient errors."""
        last_result = None
        delay = self.retry_delay

        for attempt in range(self.max_retries + 1):
            result = self._generate_impl(prompt, system_prompt)
            result.retries = attempt

            if not result.error or not is_retryable_error(result.error):
                return result

            last_result = result

            if attempt < self.max_retries:
                time.sleep(delay)
                delay *= self.retry_multiplier

        return last_result

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

        # Build headers, merging defaults with any custom headers from config
        # Resolve env var references in header values
        headers = {
            "User-Agent": USER_AGENT,
            "X-Caller": X_CALLER,
        }
        if config.headers:
            resolved_headers = {k: resolve_header_value(v) for k, v in config.headers.items()}
            headers.update(resolved_headers)

        self.client = Anthropic(
            api_key=config.api_key,
            default_headers=headers,
        )

    def _generate_impl(self, prompt: str, system_prompt: Optional[str] = None) -> GenerationResult:
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

        # Build headers, merging defaults with any custom headers from config
        # Resolve env var references in header values
        headers = {
            "User-Agent": USER_AGENT,
            "X-Caller": X_CALLER,
        }
        if config.headers:
            resolved_headers = {k: resolve_header_value(v) for k, v in config.headers.items()}
            headers.update(resolved_headers)

        kwargs = {
            "api_key": config.api_key,
            "default_headers": headers,
        }
        if config.base_url:
            kwargs["base_url"] = config.base_url

        self.client = OpenAI(**kwargs)

    def _generate_impl(self, prompt: str, system_prompt: Optional[str] = None) -> GenerationResult:
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
    """vLLM provider using OpenAI-compatible API.

    Environment variables:
        VLLM_API_KEY: API key (optional, defaults to "dummy")
        VLLM_BASE_URL: Base URL (e.g., http://localhost:8000/v1)
    """

    def __init__(self, config: ModelConfig):
        import os

        if not config.base_url:
            config.base_url = os.environ.get("VLLM_BASE_URL")
        if not config.base_url:
            raise ValueError(
                "VLLMProvider requires base_url via config or VLLM_BASE_URL env var"
            )
        if not config.api_key:
            config.api_key = os.environ.get("VLLM_API_KEY", "dummy")
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

    def _generate_impl(self, prompt: str, system_prompt: Optional[str] = None) -> GenerationResult:
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


class LiteLLMProvider(OpenAIProvider):
    """LiteLLM provider for unified access to multiple LLM providers.

    LiteLLM provides a unified interface to 100+ LLMs including OpenAI, Anthropic,
    Google, and open-source models. It exposes an OpenAI-compatible API.

    Environment variables:
        LITELLM_API_KEY: API key for LiteLLM proxy
        LITELLM_BASE_URL: Base URL (default: http://localhost:4000/v1)
    """

    def __init__(self, config: ModelConfig):
        import os

        if not config.base_url:
            config.base_url = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000/v1")
        if not config.api_key:
            config.api_key = os.environ.get("LITELLM_API_KEY", "dummy")
        super().__init__(config)


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

    def _generate_impl(self, prompt: str, system_prompt: Optional[str] = None) -> GenerationResult:
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
                    "User-Agent": USER_AGENT,
                    "X-Caller": X_CALLER,
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
        ProviderType.LITELLM: LiteLLMProvider,
        ProviderType.VLLM: VLLMProvider,
        ProviderType.QISKIT_ASSISTANT: QiskitAssistantProvider,
    }

    provider_class = provider_map.get(config.provider)
    if not provider_class:
        raise ValueError(f"Unsupported provider: {config.provider}")

    return provider_class(config)
