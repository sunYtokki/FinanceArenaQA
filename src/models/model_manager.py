"""Model management interface and implementations for LLM providers."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    """Response from a model provider."""
    content: str
    model: str
    usage: Optional[Dict[str, Any]] = None  # Token usage info
    metadata: Optional[Dict[str, Any]] = None  # Provider-specific metadata


@dataclass
class ModelConfig:
    """Configuration for model providers."""
    model_name: str
    temperature: float = 0.1  # Lower temperature for financial reasoning
    max_ctx: Optional[int] = None
    max_predict: Optional[int] = None
    timeout: int = 30
    provider_options: Optional[Dict[str, Any]] = None


class ModelProvider(ABC):
    """Abstract base class for LLM model providers."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model_name = config.model_name

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response from the model.

        Args:
            prompt: Input prompt for the model
            **kwargs: Additional provider-specific parameters

        Returns:
            ModelResponse containing the generated content and metadata
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model provider is available and configured.

        Returns:
            True if the provider can be used, False otherwise
        """
        pass

    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """Get list of supported model names for this provider.

        Returns:
            List of model names that can be used with this provider
        """
        pass

    def validate_config(self) -> bool:
        """Validate the provider configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        return (
            self.config.model_name is not None and
            0.0 <= self.config.temperature <= 2.0 and
            self.config.timeout > 0
        )


class ModelManager:
    """Manages multiple model providers and handles switching logic."""

    def __init__(self):
        self._providers: Dict[str, ModelProvider] = {}
        self._default_provider: Optional[str] = None

    def register_provider(self, name: str, provider: ModelProvider):
        """Register a model provider.

        Args:
            name: Unique name for the provider
            provider: ModelProvider instance
        """
        if not provider.validate_config():
            raise ValueError(f"Invalid configuration for provider '{name}'")

        self._providers[name] = provider

        # Set as default if it's the first available provider
        if self._default_provider is None and provider.is_available():
            self._default_provider = name

    def get_provider(self, name: Optional[str] = None) -> ModelProvider:
        """Get a model provider by name or return default.

        Args:
            name: Provider name, if None uses default

        Returns:
            ModelProvider instance

        Raises:
            ValueError: If provider not found or not available
        """
        if name is None:
            name = self._default_provider

        if name is None:
            raise ValueError("No default provider available")

        if name not in self._providers:
            raise ValueError(f"Provider '{name}' not registered")

        provider = self._providers[name]
        if not provider.is_available():
            raise ValueError(f"Provider '{name}' is not available")

        return provider

    def list_providers(self) -> Dict[str, bool]:
        """List all registered providers and their availability status.

        Returns:
            Dict mapping provider names to availability status
        """
        return {name: provider.is_available()
                for name, provider in self._providers.items()}

    def get_available_providers(self) -> List[str]:
        """Get list of currently available provider names.

        Returns:
            List of provider names that are currently available
        """
        return [name for name, provider in self._providers.items()
                if provider.is_available()]

    def set_default_provider(self, name: str):
        """Set the default provider.

        Args:
            name: Provider name to set as default

        Raises:
            ValueError: If provider not found or not available
        """
        if name not in self._providers:
            raise ValueError(f"Provider '{name}' not registered")

        provider = self._providers[name]
        if not provider.is_available():
            raise ValueError(f"Provider '{name}' is not available")

        self._default_provider = name

    async def generate(self, prompt: str, provider_name: Optional[str] = None, **kwargs) -> ModelResponse:
        """Generate response using specified or default provider.

        Args:
            prompt: Input prompt
            provider_name: Specific provider to use, if None uses default
            **kwargs: Additional parameters for generation

        Returns:
            ModelResponse from the selected provider
        """
        provider = self.get_provider(provider_name)
        return await provider.generate(prompt, **kwargs)

    def get_all_supported_models(self) -> Dict[str, List[str]]:
        """Get supported models for all providers.

        Returns:
            Dict mapping provider names to their supported models
        """
        return {name: provider.get_supported_models()
                for name, provider in self._providers.items()}

    async def close_all(self):
        """Close all provider connections."""
        for provider in self._providers.values():
            if hasattr(provider, 'close'):
                await provider.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.close_all()
        return False  # Don't suppress exceptions


def create_model_manager_from_config(config_dict: Dict[str, Any]) -> ModelManager:
    """Create a ModelManager from configuration dictionary.

    Args:
        config_dict: Configuration with provider settings

    Returns:
        Configured ModelManager instance

    Example config:
        {
            "providers": {
                "ollama": {
                    "type": "ollama",
                    "model_name": "llama2",
                    "base_url": "http://localhost:11434",
                    "temperature": 0.1
                }
            },
            "default_provider": "ollama"
        }
    """
    manager = ModelManager()

    providers_config = config_dict.get("providers", {})

    for name, provider_config in providers_config.items():
        provider_type = provider_config.get("type")

        # Create model config
        model_config = ModelConfig(
            model_name=provider_config["model_name"],
            temperature=provider_config.get("temperature", 0.1),
            max_ctx=provider_config.get("max_ctx"),
            max_predict=provider_config.get("max_predict"),
            timeout=provider_config.get("timeout", 30),
            provider_options=provider_config.get("options", {})
        )

        # Create provider based on type
        if provider_type == "ollama":
            from .ollama_provider import OllamaProvider
            base_url = provider_config.get("base_url", "http://localhost:11434")
            provider = OllamaProvider(model_config, base_url)
        else:
            logger.warning(f"Unknown provider type '{provider_type}' for '{name}'")
            continue

        manager.register_provider(name, provider)

    # Set default provider if specified
    default_provider = config_dict.get("default_provider")
    if default_provider and default_provider in manager._providers:
        try:
            manager.set_default_provider(default_provider)
        except ValueError as e:
            logger.warning(f"Could not set default provider: {e}")

    return manager