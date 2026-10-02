from __future__ import annotations

def reciprocal_rank_fusion(rankings: list[list[str]],
                           *,
                           k: int = 60,
                           )-> list[tuple[str, float]]:
    if k <= 0:
        raise ValueError(f"rrf k must be positive, got {k}")

    fused: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    order = 0

    for ranking in rankings:
        seen_in_list: set[str] = set()
        for position, chunk_id in enumerate(ranking):
            if chunk_id in seen_in_list:
                continue
            seen_in_list.add(chunk_id)

            rank = position + 1
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)

            if chunk_id not in first_seen:
                first_seen[chunk_id] = order
                order += 1

    return sorted(
        fused.items(),
        key=lambda item: (-item[1], first_seen[item[0]]),
    )