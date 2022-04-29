import sys

import numpy as np
import tensorflow as tf
import yaml

from mtgml.constants import MAX_CUBE_SIZE
from mtgml.models.recommender import CubeRecommender
from mtgml.config.hyper_config import HyperConfig

MODEL: CubeRecommender = None
INT_TO_CARD = []
CARD_TO_INT: dict = {}
LOGGER = None


def initialize_recommender(MODEL_PATH, EMBEDDINGS, int_to_card, card_to_int, logger):
    global MODEL, INT_TO_CARD, CARD_TO_INT, LOGGER
    INT_TO_CARD = int_to_card
    CARD_TO_INT = card_to_int
    LOGGER = logger
    with open(MODEL_PATH/'cube_hyper_config.yaml', 'r') as config_file:
        data = yaml.load(config_file, yaml.Loader)
    hyper_config = HyperConfig(layer_type=CubeRecommender, data=data, fixed={
        'num_cards': len(INT_TO_CARD) + 1,
        'card_embeddings': EMBEDDINGS,
    })
    tf.config.optimizer.set_jit(bool(hyper_config.get_bool('use_xla', default=False, help='Whether to use xla to speed up calculations.')))
    MODEL = hyper_config.build(name='CombinedModel', dynamic=False)
    if MODEL is None:
        logger.error('Failed to load model')
        sys.exit(1)
    MODEL.compile()
    latest = MODEL_PATH / 'cube_model'
    MODEL.load_weights(latest).expect_partial()
    MODEL.build(())

def get_cube_recomendations(cube: list[str], num_recs=10):
    cube_idxs = [CARD_TO_INT[card] + 1 for card in cube if card in CARD_TO_INT]
    cube_ids = np.array([cube_idxs[:MAX_CUBE_SIZE] + [0 for _ in range(MAX_CUBE_SIZE - len(cube_idxs))]], dtype=np.int32)
    assert MODEL != None
    results = MODEL((cube_ids,), training=False).numpy()[0]
    sorted_indices = results.argsort()
    adds = []
    idx = 1
    while len(adds) < num_recs and idx <= len(sorted_indices):
        oracle_id = INT_TO_CARD[sorted_indices[-idx]]['oracle_id']
        if oracle_id not in cube:
            adds.append((oracle_id, float(results[sorted_indices[-idx]])))
        idx += 1
    cuts = []
    idx = 0
    while len(cuts) < num_recs and idx < len(sorted_indices):
        oracle_id = INT_TO_CARD[sorted_indices[idx]]['oracle_id']
        if oracle_id in cube:
            cuts.append((oracle_id, float(results[sorted_indices[idx]])))
        idx += 1
    return {
        "adds": adds,
        "cuts": cuts,
        "success": True,
    }
