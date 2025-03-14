"""
Models package for Babeltron.

This package contains the translation model implementations.
"""

from babeltron.app.models.translation.base import TranslationModelBase
from babeltron.app.models.translation.factory import get_translation_model
from babeltron.app.models.translation.m2m100 import M2M100TranslationModel
from babeltron.app.models.translation.m2m100 import (
    ModelArchitecture as M2MModelArchitecture,
)
from babeltron.app.models.translation.m2m100 import (
    get_translation_model as get_m2m_model,
)
from babeltron.app.models.translation.nllb import (
    ModelArchitecture as NLLBModelArchitecture,
)
from babeltron.app.models.translation.nllb import NLLBTranslationModel
from babeltron.app.models.translation.nllb import (
    get_translation_model as get_nllb_model,
)

__all__ = [
    "TranslationModelBase",
    "get_translation_model",
    "M2M100TranslationModel",
    "M2MModelArchitecture",
    "get_m2m_model",
    "NLLBTranslationModel",
    "NLLBModelArchitecture",
    "get_nllb_model",
]

# For backward compatibility
get_model = get_translation_model
