"""
AutoEdit Pro — Skill System
Load, discover, and manage editing skill packs.
"""

from .loader import load_skill_pack, SkillPack
from .registry import SkillRegistry, get_registry
from .builder import build_skill_from_references, ReferenceVideo
from .discovery.fetcher import discover_skills, SkillSource
from .discovery.extractor import extract_patterns, ExtractedPattern
from .discovery.normalizer import normalize_to_skill_pack
from .embedder import SkillEmbedder, embed_skill_pack
from .few_shot import FewShotSelector, select_few_shots
from .lora.trainer import train_lora, LoRAConfig
from .lora.adapter import LoRAAdapter

__all__ = [
    "load_skill_pack",
    "SkillPack",
    "SkillRegistry",
    "get_registry",
    "build_skill_from_references",
    "ReferenceVideo",
    "discover_skills",
    "SkillSource",
    "extract_patterns",
    "ExtractedPattern",
    "normalize_to_skill_pack",
    "SkillEmbedder",
    "embed_skill_pack",
    "FewShotSelector",
    "select_few_shots",
    "train_lora",
    "LoRAConfig",
    "LoRAAdapter",
]