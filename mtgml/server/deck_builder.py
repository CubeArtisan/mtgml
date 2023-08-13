from collections import Counter, defaultdict

import numpy as np

from mtgml.constants import MAX_COPIES


def get_deck_scores(deck_state, model, card_to_int, int_to_card, tracer, original_to_new_index):
    new_to_original_index = {v: k for k, v in enumerate(original_to_new_index)}
    new_to_original_index = [new_to_original_index.get(i, 0) for i in range(len(int_to_card) + 1)]
    missing = [
        card_id
        for card_id in deck_state["pool"] + deck_state["basics"]
        if original_to_new_index[card_to_int[card_id] + 1] == 0
    ]
    pool = [
        original_to_new_index[card_to_int[card_id] + 1]
        for card_id in deck_state["pool"]
        if original_to_new_index[card_to_int[card_id] + 1] != 0
    ]
    basics = [
        original_to_new_index[card_to_int[card_id] + 1]
        for card_id in deck_state["basics"]
        if original_to_new_index[card_to_int[card_id] + 1] != 0
    ]
    pool = pool + MAX_COPIES * basics
    pool_counter = Counter(pool)
    removed = Counter()
    for k, v in pool_counter.items():
        if v > MAX_COPIES:
            removed[k] = v - MAX_COPIES
            pool_counter[k] = MAX_COPIES
    for i, basic in enumerate(basics):
        max_to_add = MAX_COPIES - pool_counter.get(basic, 0)
        if max_to_add < 0:
            continue
        pool += [basic for _ in range(max_to_add)]
    pool = sorted(pool_counter.elements())
    pool_counter = defaultdict(int)
    copy_counts = []
    for card in pool:
        copy_counts.append(pool_counter[card])
        pool_counter[card] += 1
    with tracer.start_as_current_span("call_deck_builder"):
        call_deck_builder = model.get_signature_runner("call_deck_builder")
        result = call_deck_builder(
            pool=np.array([pool], dtype=np.int16), instance_nums=np.array([copy_counts], dtype=np.int16)
        )
        scores = result["card_scores"][0]
    sorted_scores = sorted(
        [(int_to_card[new_to_original_index[card] - 1], float(score)) for card, score in zip(pool, scores)]
        + [(int_to_card[new_to_original_index[card] - 1], 0.0) for card in removed.elements()],
        key=lambda x: x[1],
        reverse=True,
    )
    return {"main": sorted_scores[:40], "side": sorted_scores[40:], "unknown": missing}
