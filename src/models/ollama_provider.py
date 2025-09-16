
"""Ollama provider implementation for local model integration."""

import asyncio
import aiohttp
import logging
from typing import Dict, Any, List

from .model_manager import ModelProvider, ModelConfig, ModelResponse

logger = logging.getLogger(__name__)


class OllamaProvider(ModelProvider):
    """Ollama local model provider implementation."""

    def __init__(self, config: ModelConfig, base_url: str = "http://localhost:11434"):
        super().__init__(config)
        self.base_url = base_url.rstrip('/')
        self._session = None
        self._session_lock = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with proper connection limits."""
        if self._session_lock is None:
            import asyncio
            self._session_lock = asyncio.Lock()

        async with self._session_lock:
            if self._session is None or self._session.closed:
                timeout = aiohttp.ClientTimeout(total=self.config.timeout)
                # Limit concurrent connections to prevent resource exhaustion
                connector = aiohttp.TCPConnector(
                    limit=10,  # Max 10 concurrent connections
                    limit_per_host=5,  # Max 5 per host
                    ttl_dns_cache=300,
                    use_dns_cache=True,
                )
                self._session = aiohttp.ClientSession(
                    timeout=timeout,
                    connector=connector
                )
            return self._session

    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response using Ollama API.

        Args:
            prompt: Input prompt for the model
            **kwargs: Additional parameters (stream, format, etc.)

        Returns:
            ModelResponse with generated content
        """
        session = await self._get_session()

        # Prepare request payload
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,  # We want complete response
            "options": {
                "temperature": self.config.temperature,
                "num_ctx": self.config.max_ctx,
                "num_predict": self.config.max_predict,
                **(self.config.provider_options or {})
            }
        }

        # Add any additional kwargs to options
        for key, value in kwargs.items():
            if key in ["stream", "format"]:
                payload[key] = value
            else:
                payload["options"][key] = value

        try:
            async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                response.raise_for_status()
                result = await response.json()

                # Ensure we got a proper response
                if not result.get("response"):
                    logger.warning("Empty response from Ollama")
                    return ModelResponse(
                        content="",
                        model=self.config.model_name,
                        metadata={"error": "empty_response"}
                    )

                return ModelResponse(
                    content=result.get("response", ""),
                    model=self.config.model_name,
                    metadata={
                        "total_duration": result.get("total_duration"),
                        "load_duration": result.get("load_duration"),
                        "prompt_eval_count": result.get("prompt_eval_count"),
                        "eval_count": result.get("eval_count"),
                        "eval_duration": result.get("eval_duration")
                    }
                )
        except asyncio.TimeoutError as e:
            logger.error(f"Ollama request timeout: {e}")
            raise RuntimeError(f"Ollama request timed out after {self.config.timeout}s")
        except aiohttp.ClientError as e:
            logger.error(f"Ollama API error: {e}")
            raise RuntimeError(f"Failed to generate response from Ollama: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in Ollama generation: {e}")
            raise

    def is_available(self) -> bool:
        """Check if Ollama server is running and model is available.

        Returns:
            True if Ollama is accessible and model exists
        """
        try:
            # Use synchronous check for availability
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()

            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]

            return self.config.model_name in model_names
        except Exception as e:
            logger.debug(f"Ollama availability check failed: {e}")
            return False

    def get_supported_models(self) -> List[str]:
        """Get list of available models from Ollama.

        Returns:
            List of model names available in Ollama
        """
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()

            models = response.json().get("models", [])
            return [model["name"] for model in models]
        except Exception as e:
            logger.error(f"Failed to get Ollama models: {e}")
            return []

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
