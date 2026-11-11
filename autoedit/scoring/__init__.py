"""
AutoEdit Pro — Viral Scoring
Heuristic-based viral potential analysis.
"""

from .viral_scorer import ViralScorer, ViralScore, ScoreBreakdown
from .metrics import (
    calculate_hook_strength,
    calculate_pacing_health,
    estimate_retention_curve,
    check_platform_compliance,
    analyze_audio_quality,
)
from .reporter import generate_score_report, ScoreReport

__all__ = [
    "ViralScorer",
    "ViralScore",
    "ScoreBreakdown",
    "calculate_hook_strength",
    "calculate_pacing_health",
    "estimate_retention_curve",
    "check_platform_compliance",
    "analyze_audio_quality",
    "generate_score_report",
    "ScoreReport",
]