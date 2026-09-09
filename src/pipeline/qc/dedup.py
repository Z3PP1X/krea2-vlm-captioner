"""Perceptual hashing and deduplication clustering using pHash."""

from __future__ import annotations

import imagehash
from PIL import Image
from typing import Dict, List, Tuple, Any


def compute_phash(pil_img: Image.Image) -> str:
    """Calculates perceptual hash (pHash) for an image and returns hexadecimal string."""
    ph = imagehash.phash(pil_img)
    return str(ph)


def hamming_distance(hex_hash1: str, hex_hash2: str) -> int:
    """Calculates Hamming distance between two hex pHash strings."""
    h1 = imagehash.hex_to_hash(hex_hash1)
    h2 = imagehash.hex_to_hash(hex_hash2)
    return int(h1 - h2)


def cluster_by_phash(
    items: List[Dict[str, Any]],
    threshold: int = 6,
) -> List[List[Dict[str, Any]]]:
    """Clusters items by pHash Hamming distance <= threshold using disjoint-set / union-find.
    
    Each item must contain at least 'image_id' and 'phash'.
    Returns list of clusters (each cluster is a list of items).
    """
    n = len(items)
    if n == 0:
        return []

    parent = list(range(n))

    def find(i: int) -> int:
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i: int, j: int) -> None:
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    parsed_hashes = [imagehash.hex_to_hash(item["phash"]) for item in items]

    for i in range(n):
        for j in range(i + 1, n):
            dist = parsed_hashes[i] - parsed_hashes[j]
            if dist <= threshold:
                union(i, j)

    clusters_map: Dict[int, List[Dict[str, Any]]] = {}
    for i in range(n):
        root = find(i)
        if root not in clusters_map:
            clusters_map[root] = []
        clusters_map[root].append(items[i])

    return list(clusters_map.values())
