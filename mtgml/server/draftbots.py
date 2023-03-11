import math

import numpy as np
import numpy.ma as ma

from mtgml.constants import MAX_BASICS, MAX_CARDS_IN_PACK, MAX_PICKED, MAX_SEEN_PACKS
from mtgml.utils.grid import interpolate

POOL_ORACLE_METADATA = {
    "title": "Pool Synergy",
    "tooltip": "How well the card goes with the cards that have already been picked.",
}
SEEN_ORACLE_METADATA = {
    "title": "Seen Synergy",
    "tooltip": "How well the card goes with the cards that have already been seen, looking for things like openness and combos.",
}
RATING_ORACLE_METADATA = {
    "title": "Card Rating",
    "tooltip": "How good the card is in a vacuum.",
}
METADATA = [POOL_ORACLE_METADATA, SEEN_ORACLE_METADATA, RATING_ORACLE_METADATA]


def get_draft_scores(drafter_state, model, card_to_int, tracer, original_to_new_index):
    basics = np.zeros((1, MAX_BASICS), dtype=np.int16)
    idx = 0
    for card_id in drafter_state["basics"]:
        if card_id in card_to_int and original_to_new_index[card_to_int[card_id] + 1] != 0 and idx < MAX_BASICS:
            basics[0][idx] = card_to_int[card_id] + 1
            idx += 1
    pool = np.zeros((1, MAX_PICKED), dtype=np.int16)
    idx = 0
    for card_id in drafter_state["picked"][-MAX_SEEN_PACKS:]:
        if card_id in card_to_int and original_to_new_index[card_to_int[card_id] + 1] != 0:
            pool[0][idx] = original_to_new_index[card_to_int[card_id] + 1]
            idx += 1
    seen_packs = np.zeros((1, MAX_SEEN_PACKS, MAX_CARDS_IN_PACK), dtype=np.int16)
    seen_coords = np.zeros((1, MAX_SEEN_PACKS, 4, 2), dtype=np.int8)
    seen_weights = np.zeros((1, MAX_SEEN_PACKS, 4), dtype=np.float32)
    idx_pack = 0
    original_cards_idx = []
    seen_idx = 0
    for original_idx, pack in list(enumerate(drafter_state["seen"]))[-MAX_SEEN_PACKS:]:
        original_cards_idx = []
        idx = 0
        seen_coords[0][idx_pack], seen_weights[0][idx_pack] = interpolate(
            pack["pickNum"], pack["numPicks"], pack["packNum"], drafter_state["numPacks"]
        )
        for i, card_id in enumerate(pack["pack"]):
            if (
                card_id in card_to_int
                and original_to_new_index[card_to_int[card_id] + 1] != 0
                and idx < MAX_CARDS_IN_PACK
            ):
                seen_packs[0][idx_pack][idx] = original_to_new_index[card_to_int[card_id] + 1]
                original_cards_idx.append(i)
                idx += 1
        idx_pack += 1
        seen_idx = original_idx
    # 1 is a bad number for dimensions
    trim = max(2, seen_idx + 1)
    seen_packs = seen_packs[:, :trim]
    seen_coords = seen_coords[:, :trim]
    seen_weights = seen_weights[:, :trim]
    pool = pool[:, :trim]
    with tracer.start_as_current_span("call_draftbots"):
        call_draftbots = model.get_signature_runner("call_draftbots")
        result = call_draftbots(
            basics=basics, pool=pool, seen=seen_packs, seen_coords=seen_coords, seen_coord_weights=seen_weights
        )
        oracle_scores = result["sublayer_scores"][0][idx_pack - 1]
        oracle_weights = result["sublayer_weights"][0][idx_pack - 1]
    oracle_scores = ma.masked_array(
        oracle_scores, mask=np.broadcast_to(np.expand_dims(seen_packs[0][idx_pack - 1] == 0, -1), oracle_scores.shape)
    )
    oracle_scores_min = np.amin(oracle_scores, axis=0)
    oracle_scores = oracle_scores - oracle_scores_min
    oracle_max_scores = np.max(oracle_scores, axis=0) / 10
    oracle_max_scores = np.where(oracle_max_scores > 0, oracle_max_scores, np.ones_like(oracle_max_scores))
    oracle_scores = oracle_scores / oracle_max_scores
    oracle_weights = oracle_weights * oracle_max_scores
    oracle_weights = oracle_weights / oracle_weights.sum()
    scores = (oracle_weights * oracle_scores).sum(axis=1)
    oracles = [
        [
            dict(metadata)
            | {
                "weight": float(weight) if not math.isnan(float(weight)) else -1,
                "score": float(score) if not math.isnan(float(score)) else -1,
            }
            for metadata, weight, score in zip(METADATA, oracle_weights, card_scores)
        ]
        for card_scores in oracle_scores
    ]
    returned_scores = [0.0 for _ in drafter_state["seen"][seen_idx]["pack"]]
    returned_oracles = [
        [
            metadata | {"weight": float(weight) if not math.isnan(float(weight)) else -1, "score": 0}
            for metadata, weight in zip(METADATA, oracle_weights)
        ]
        for _ in drafter_state["seen"][seen_idx]["pack"]
    ]
    for i, score, oracle in zip(original_cards_idx, scores, oracles):
        if 0 <= i < len(returned_scores):
            returned_scores[i] = float(score)
            returned_oracles[i] = oracle
    return {
        "scores": returned_scores,
        "oracles": returned_oracles,
        "success": True,
    }
