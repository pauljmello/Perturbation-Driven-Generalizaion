import logging
from typing import Type, Callable, Set

logger = logging.getLogger('model_registry')

ModelClass = Type
ModelFactory = Callable


class ModelRegistry:
    """
    Registry with error handling and caching.
    """
    _model_classes = {}
    _model_factories = {}
    _supported_types = set()
    _supported_sizes = {'small', 'medium', 'large'}
    _model_cache = {}

    @classmethod
    def get_available_models(cls):
        """
        Get all available models efficiently.
        """
        if not hasattr(cls, '_available_models_cache'):
            cls._available_models_cache = [(model_type, model_size)
                for model_type in cls.get_supported_types()
                for model_size in cls.get_supported_sizes()
                if cls.is_model_available(model_type, model_size)
            ]
        return cls._available_models_cache

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
    def is_model_available(cls, model_type, model_size):
        """
        Check model availability with efficient caching.
        """
        cache_key = f"{model_type}_{model_size}"
        if cache_key in cls._model_cache:
            return cls._model_cache[cache_key]

        result = (model_type in cls._model_classes and
                  model_size in cls._model_classes.get(model_type, {}))
        cls._model_cache[cache_key] = result
        return result
