"""Phase 5: Dimensionality reduction and mapping."""

from doc_visualizer.phase5.clusterers import HdbscanClusterer, build_clusterer
from doc_visualizer.phase5.models import MapPoint, PhaseFiveRunResult, StrategyMap
from doc_visualizer.phase5.reducers import UmapReducer, build_reducer
from doc_visualizer.phase5.service import build_strategy_map

__all__ = [
    "HdbscanClusterer",
    "MapPoint",
    "PhaseFiveRunResult",
    "StrategyMap",
    "UmapReducer",
    "build_clusterer",
    "build_reducer",
    "build_strategy_map",
]
