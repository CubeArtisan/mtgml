import numpy as np

from mtgml.constants import MAX_CUBE_SIZE


def get_cube_recomendations(cube: list[str], model, card_to_int: dict[str, int], int_to_card: dict[int, dict[str, str]],
                            tracer, original_to_new_index, num_recs=10):
    new_to_original_index = {v: k for k,v in enumerate(original_to_new_index)}
    new_to_original_index = [new_to_original_index.get(i, 0) for i in range(len(int_to_card) + 1)]
    original_cube_idxs = [card_to_int[card] + 1 for card in cube if card in card_to_int]
    cube_idxs = [original_to_new_index[card_to_int[card] + 1] for card in cube if card in card_to_int]
    inverted_cube_idxs = [new_to_original_index[x] for x in cube_idxs]
    missing = [int_to_card[x - 1] for x, y in zip(original_cube_idxs, inverted_cube_idxs) if x != y]
    cube_ids = np.array([cube_idxs[:MAX_CUBE_SIZE] + [0 for _ in range(MAX_CUBE_SIZE - len(cube_idxs))]], dtype=np.int16)
    with tracer.start_as_current_span('call_cube_recommender'):
        call_recommender = model.get_signature_runner('call_recommender')
        results = call_recommender(cube=cube_ids)
        results = results['decoded_noisy_cube'][0]
    sorted_indices = results.argsort()
    adds = []
    idx = 1
    while len(adds) < num_recs and idx <= len(sorted_indices):
        card_idx = new_to_original_index[sorted_indices[-idx] + 1] - 1
        oracle_id = int_to_card[card_idx]
        if original_to_new_index[card_idx + 1] != 0 and oracle_id not in cube:
            adds.append((oracle_id, float(results[sorted_indices[-idx]])))
        idx += 1
    cuts = []
    idx = 0
    while len(cuts) < num_recs and idx < len(sorted_indices):
        card_idx = new_to_original_index[sorted_indices[-idx] + 1] - 1
        oracle_id = int_to_card[card_idx]
        if original_to_new_index[card_idx + 1] != 0 and oracle_id in cube:
            cuts.append((oracle_id, float(results[sorted_indices[idx]])))
        idx += 1
    return {
        "adds": adds,
        "cuts": cuts,
        "missing": missing,
        "success": True,
    }
