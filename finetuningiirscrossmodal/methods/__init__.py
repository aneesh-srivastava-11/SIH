"""
Method Adapters for MatchAnything and Fine-Tuned Matchers.
"""

from finetuningiirscrossmodal.methods.base import RegistrationResult, RegistrationMethod
from finetuningiirscrossmodal.methods.matchanything import MatchAnythingMethod
from finetuningiirscrossmodal.methods.finetuned_loftr import FineTunedLoFTRMethod
from finetuningiirscrossmodal.methods.finetuned_roma import FineTunedRoMaMethod

__all__ = [
    "RegistrationResult",
    "RegistrationMethod",
    "MatchAnythingMethod",
    "FineTunedLoFTRMethod",
    "FineTunedRoMaMethod",
]
