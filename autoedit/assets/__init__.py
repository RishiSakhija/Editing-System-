"""
AutoEdit Pro — Asset Management
Search, rank, download, and cache assets from free APIs.
"""

from .manager import AssetManager, AssetSearchResult
from .providers.pexels import PexelsProvider
from .providers.pixabay import PixabayProvider
from .providers.unsplash import UnsplashProvider
from .providers.freesound import FreesoundProvider
from .providers.yt_audio import YouTubeAudioProvider
from .ranker import rank_assets, AssetRanker
from .cache import AssetCache, CachedAsset

__all__ = [
    "AssetManager",
    "AssetSearchResult",
    "PexelsProvider",
    "PixabayProvider",
    "UnsplashProvider",
    "FreesoundProvider",
    "YouTubeAudioProvider",
    "rank_assets",
    "AssetRanker",
    "AssetCache",
    "CachedAsset",
]