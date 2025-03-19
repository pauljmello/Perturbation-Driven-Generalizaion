import logging
from typing import Dict, Type, Callable, Set

logger = logging.getLogger('model_registry')

ModelClass = Type
ModelFactory = Callable


class ModelRegistry:
    """
    Registry of model architectures and factories.
    """

    _model_classes: Dict[str, Dict[str, ModelClass]] = {}
    _model_factories: Dict[str, ModelFactory] = {}
    _supported_types: Set[str] = set()
    _supported_sizes: Set[str] = {'small', 'medium', 'large'}

    @classmethod
    def register_model_class(cls, model_type: str, model_size: str, model_class: ModelClass) -> None:
        """
        Register a model class for a specific type and size.
        """
        if model_type not in cls._model_classes:
            cls._model_classes[model_type] = {}
            cls._supported_types.add(model_type)
        cls._model_classes[model_type][model_size] = model_class
        logger.debug(f"Registered {model_type} {model_size} model: {model_class.__name__}")

    @classmethod
    def register_model_factory(cls, model_type: str, factory: ModelFactory) -> None:
        """
        Register a model factory function for a specific model type.
        """
        cls._model_factories[model_type] = factory
        cls._supported_types.add(model_type)
        logger.debug(f"Registered factory for {model_type} models")

    @classmethod
    def get_model_class(cls, model_type: str, model_size: str) -> ModelClass:
        """
        Get the model class for a specific type and size.
        """
        if model_type not in cls._model_classes:
            raise ValueError(f"Model type '{model_type}' not registered")
        if model_size not in cls._model_classes[model_type]:
            raise ValueError(f"Size '{model_size}' not registered for model type '{model_type}'")
        return cls._model_classes[model_type][model_size]

    @classmethod
    def get_model_factory(cls, model_type: str) -> ModelFactory:
        """
        Get the factory function for a specific model type.
        """
        if model_type not in cls._model_factories:
            raise ValueError(f"No factory registered for model type '{model_type}'")
        return cls._model_factories[model_type]

    @classmethod
    def get_supported_types(cls) -> Set[str]:
        """
        Get all supported model types.
        """
        return cls._supported_types

    @classmethod
    def get_supported_sizes(cls) -> Set[str]:
        """
        Get all supported model sizes.
        """
        return cls._supported_sizes

    @classmethod
    def is_model_available(cls, model_type: str, model_size: str) -> bool:
        """
        Check if a specific model type and size is available.
        """
        if model_type not in cls._model_classes:
            return False
        return model_size in cls._model_classes[model_type]