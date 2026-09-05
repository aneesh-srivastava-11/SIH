"""
Method Registry for dynamic registration and discovery of benchmark algorithms.
"""

from typing import Dict, Type, List, Optional, Tuple, Any
import importlib
import logging
from methods.base import RegistrationMethod

logger = logging.getLogger(__name__)


class MethodRegistry:
    """Singleton registry holding available image registration method classes."""

    _instance: Optional["MethodRegistry"] = None
    _registry: Dict[str, Type[RegistrationMethod]] = {}

    def __new__(cls) -> "MethodRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry = {}
        return cls._instance

    @classmethod
    def register(cls, name: str, method_cls: Type[RegistrationMethod]) -> None:
        """Register a new registration method class under a given name key."""
        key = name.lower().strip()
        cls()._registry[key] = method_cls
        logger.debug(f"Registered method: {key} -> {method_cls.__name__}")

    @classmethod
    def get(cls, name: str) -> Optional[Type[RegistrationMethod]]:
        """Retrieve registered method class by name."""
        return cls()._registry.get(name.lower().strip())

    @classmethod
    def list_all(cls) -> Dict[str, Tuple[bool, str]]:
        """
        List all registered methods along with their availability status.
        Returns dict: {method_key: (is_available: bool, reason: str)}
        """
        cls.discover()
        status = {}
        for name, method_cls in cls()._registry.items():
            avail, reason = method_cls.is_available()
            status[name] = (avail, reason)
        return status

    @classmethod
    def get_enabled(cls, config: Any) -> List[RegistrationMethod]:
        """
        Instantiate and return all enabled & available method objects based on config.
        """
        cls.discover()
        enabled_keys = [k.lower().strip() for k in config.enabled_methods]
        instances: List[RegistrationMethod] = []

        for key in enabled_keys:
            method_cls = cls()._registry.get(key)
            if method_cls is None:
                logger.warning(f"Method '{key}' is enabled in config but not registered.")
                continue

            avail, reason = method_cls.is_available()
            if not avail:
                logger.warning(f"Method '{key}' is disabled (unavailable): {reason}")
                continue

            try:
                instance = method_cls()
                instances.append(instance)
                logger.info(f"Initialized method baseline: {instance.name}")
            except Exception as e:
                logger.error(f"Failed to instantiate method '{key}': {e}")

        return instances

    @classmethod
    def discover(cls) -> None:
        """Dynamically import and register method modules in the `methods` package."""
        modules = [
            ("sift", "methods.sift", "SIFTMethod"),
            ("akaze", "methods.akaze", "AKAZEMethod"),
            ("rift2", "methods.rift2", "RIFT2Method"),
            ("superpoint_lightglue", "methods.superpoint_lightglue", "SuperPointLightGlueMethod"),
            ("efficient_loftr", "methods.efficient_loftr", "EfficientLoFTRMethod"),
            ("arosics", "methods.arosics_method", "AROSICSMethod"),
        ]

        for key, mod_name, class_name in modules:
            try:
                mod = importlib.import_module(mod_name)
                method_cls = getattr(mod, class_name)
                cls.register(key, method_cls)
            except (ImportError, AttributeError) as e:
                logger.debug(f"Module {mod_name} ({class_name}) import check: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error importing {mod_name}: {e}")

