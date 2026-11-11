"""
AutoEdit Pro — Learning Loop
Preference learning, skill updates, embedding management.
"""

from .loop import learn_from_session, LearningResult
from .embeddings import PreferenceEmbeddings, EmbeddingStore
from .preference_model import PreferenceModel, UserPreferences
from .skill_updater import update_skill_pack, SkillUpdateResult

__all__ = [
    "learn_from_session",
    "LearningResult",
    "PreferenceEmbeddings",
    "EmbeddingStore",
    "PreferenceModel",
    "UserPreferences",
    "update_skill_pack",
    "SkillUpdateResult",
]