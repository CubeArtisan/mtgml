from collections import Counter, defaultdict

import numpy as np

from mtgml.constants import MAX_COPIES


def get_deck_scores(deck_state, model, card_to_int, int_to_card, tracer, original_to_new_index):
    pool = [original_to_new_index[card_to_int[card_id] + 1] for card_id in deck_state["pool"]]
    basics = [original_to_new_index[card_to_int[card_id] + 1] for card_id in deck_state["basics"]]
    pool = pool + MAX_COPIES * basics
    pool_counter = Counter(pool)
    for k, v in pool_counter.items():
        if v > MAX_COPIES:
            pool_counter[k] = MAX_COPIES
    for i, basic in enumerate(basics):
        max_to_add = MAX_COPIES - pool_counter.get(basic, 0)
        if max_to_add < 0:
            continue
        pool += [
            basic
            for _ in range(max_to_add)
        ]
    pool = sorted(pool_counter.elements())
    pool_counter = defaultdict(int)
    copy_counts = []
    for card in pool:
        copy_counts.append(pool_counter[card])
        pool_counter[card] += 1
    with tracer.start_as_current_span("call_deck_builder"):
        call_deck_builder = model.get_signature_runner("call_deck_builder")
        result = call_deck_builder(
            pool=np.array([pool], dtype=np.int16), copy_counts=np.array([copy_counts], dtype=np.int16)
        )
        scores = result["card_scores"][0]
    return {
        "scores": sorted(
            [(int_to_card[card], score) for card, score in zip(pool, scores)], key=lambda x: x[1], reverse=True
        )
    }
