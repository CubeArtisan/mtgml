import numpy as np


def get_cube_recomendations(cube: list[str], model, card_to_int: dict[str, int], int_to_card: dict[int, str], num_recs=10):
    cube_ids = np.array([[card_to_int[card] for card in cube if card in card_to_int]], dtype=np.int32)
    results = model.cube_recommender(cube_ids, training=False).numpy()[0]
    sorted_indices = results.argsort()
    adds = []
    idx = 1
    while len(adds) < num_recs and idx <= len(sorted_indices):
        if sorted_indices[-idx] not in cube:
            adds.append(int_to_card[sorted_indices[-idx]])
        idx += 1
    cuts = []
    idx = 0
    while len(cuts) < num_recs and idx < len(sorted_indices):
        if sorted_indices[idx] in cube:
            cuts.append(int_to_card[sorted_indices[idx]])
        idx += 1
    return {
        "adds": adds,
        "cuts": cuts,
        "success": True,
    }
