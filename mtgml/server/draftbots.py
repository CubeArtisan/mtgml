import numpy as np

from mtgml.constants import MAX_BASICS, MAX_CARDS_IN_PACK, MAX_PICKED, MAX_SEEN_PACKS
from mtgml.utils.grid import interpolate

def get_draft_scores(drafter_state, model, card_to_int):
    cards_in_pack = np.zeros((1, MAX_CARDS_IN_PACK), dtype=np.int32)
    original_cards_idx = []
    idx = 0
    for i, card_id in enumerate(drafter_state['cardsInPack']):
        if card_id in card_to_int:
            cards_in_pack[0][idx] = card_to_int[card_id]
            original_cards_idx.append(i)
            idx += 1
    basics = np.zeros((1, MAX_BASICS), dtype=np.int32)
    idx = 0
    for card_id in drafter_state['basics']:
        if card_id in card_to_int:
            basics[0][idx] = card_to_int[card_id]
            idx += 1
    picked = np.zeros((1, MAX_PICKED), dtype=np.int32)
    idx = 0
    for card_id in drafter_state['picked']:
        if card_id in card_to_int:
            picked[0][idx] = card_to_int[card_id]
            idx += 1
    seen_packs = np.zeros((1, MAX_SEEN_PACKS, MAX_CARDS_IN_PACK), dtype=np.int32)
    seen_coords = np.zeros((1, MAX_SEEN_PACKS, 4, 2), dtype=np.int32)
    seen_weights = np.zeros((1, MAX_SEEN_PACKS, 4), dtype=np.float32)
    idx_pack = 0
    for pack in drafter_state['seen']:
        idx = 0
        seen_coords[0][idx_pack], seen_weights[0][idx_pack] = interpolate(pack['pickNum'], pack['numPicks'],
                                                                          pack['packNum'], drafter_state['numPacks'])
        for card_id in pack['pack']:
            if card_id in card_to_int and idx < MAX_CARDS_IN_PACK:
                seen_packs[0][idx_pack][idx] = card_to_int[card_id]
                idx += 1
        idx_pack += 1
    coords, weights = interpolate(drafter_state['pickNum'], drafter_state['numPicks'], drafter_state['packNum'],
                                  drafter_state['numPacks'])
    results = model.draftbots(((cards_in_pack, basics, picked, seen_packs, seen_coords, seen_weights,
                                coords, weights), np.zeros((1,), dtype=np.int32)), training=False).numpy()[0]
    scores = [0 for _ in drafter_state["cardsInPack"]]
    for i, score in zip(original_cards_idx, results):
        scores[i] = score
    return {
        "scores": results,
        "success": True,
    }
