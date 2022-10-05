import numpy as np

from mtgml.constants import MAX_CUBE_SIZE


def get_cube_recomendations(cube: list[str], model, card_to_int: dict[str, int], int_to_card: dict[int, dict[str, str]], tracer, num_recs=10):
    cube_idxs = [card_to_int[card] + 1 for card in cube if card in card_to_int]
    cube_ids = np.array([cube_idxs[:MAX_CUBE_SIZE] + [0 for _ in range(MAX_CUBE_SIZE - len(cube_idxs))]], dtype=np.int32)
    with tracer.start_as_current_span('call_cube_recommender'):
        results = model.cube_recommender((cube_ids, model.embed_cards.embeddings), training=False)[0].numpy()[0]
    sorted_indices = results.argsort()
    adds = []
    idx = 1
    while len(adds) < num_recs and idx <= len(sorted_indices):
        oracle_id = int_to_card[sorted_indices[-idx]]['oracle_id']
        if oracle_id not in cube:
            adds.append((oracle_id, float(results[sorted_indices[-idx]])))
        idx += 1
    cuts = []
    idx = 0
    while len(cuts) < num_recs and idx < len(sorted_indices):
        oracle_id = int_to_card[sorted_indices[idx]]['oracle_id']
        if oracle_id in cube:
            cuts.append((oracle_id, float(results[sorted_indices[idx]])))
        idx += 1
    return {
        "adds": adds,
        "cuts": cuts,
        "success": True,
    }
