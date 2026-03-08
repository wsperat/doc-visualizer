from __future__ import annotations

from doc_visualizer.phase5.clusterers import HdbscanClusterer


def test_hdbscan_clusterer_marks_small_samples_as_noise() -> None:
    clusterer = HdbscanClusterer(min_cluster_size=5)
    labels = clusterer.cluster([(0.0, 0.0), (1.0, 1.0)])
    assert labels == [-1, -1]
