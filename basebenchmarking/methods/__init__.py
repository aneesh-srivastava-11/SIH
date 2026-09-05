"""Methods package initialization."""

from methods.base import RegistrationMethod, RegistrationResult
from methods.sift import SIFTMethod
from methods.akaze import AKAZEMethod
from methods.rift2 import RIFT2Method
from methods.superpoint_lightglue import SuperPointLightGlueMethod
from methods.efficient_loftr import EfficientLoFTRMethod
from methods.arosics_method import AROSICSMethod

__all__ = [
    "RegistrationMethod",
    "RegistrationResult",
    "SIFTMethod",
    "AKAZEMethod",
    "RIFT2Method",
    "SuperPointLightGlueMethod",
    "EfficientLoFTRMethod",
    "AROSICSMethod",
]
