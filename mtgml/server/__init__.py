import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from flask import Flask, request

from mtgml.config.hyper_config import HyperConfig
from mtgml.models.card_embeddings import CombinedCardModel
from mtgml.preprocessing.tokens import tokens
from mtgml.server.cube_recommender import get_cube_recomendations, initialize_recommender
from mtgml.server.draftbots import get_draft_scores, initialize_draftbots
from mtgml.utils.grid import pad
from mtgml.preprocessing.tokenize_card import tokenize_card


app = Flask(__name__)

if __name__ != "__main__":
    gunicorn_error_logger = logging.getLogger("gunicorn.error")
    app.logger.handlers.extend(gunicorn_error_logger.handlers)
    app.logger.setLevel(logging.INFO)



MODEL_PATH = Path('ml_files/latest')
EMBEDDINGS = None
with open(MODEL_PATH / 'int_to_card.json', 'rb') as map_file:
    INT_TO_CARD = json.load(map_file)
CARD_TO_INT = {v['oracle_id']: k for k, v in enumerate(INT_TO_CARD)}

def get_embeddings():
    return np.load(MODEL_PATH/'card_embeddings.npy')
    with open(MODEL_PATH/'nlp_hyper_config.yaml', 'r') as config_file:
        data = yaml.load(config_file, yaml.Loader)
    tokenized = [tokenize_card(c) for c in INT_TO_CARD]
    max_length = max(len(c) for c in tokenized)
    tokenized = np.array([pad(c, max_length) for c in tokenized], dtype=np.int32)
    hyper_config = HyperConfig(layer_type=CombinedCardModel, data=data, fixed={
        'card_token_map': tokenized,
        'deck_adj_mtx': [],
        'cube_adj_mtx': [],
        'num_tokens': len(tokens),
    })
    tf.config.optimizer.set_jit(bool(hyper_config.get_bool('use_xla', default=False, help='Whether to use xla to speed up calculations.')))
    EMBEDDINGS = hyper_config.build(name='CombinedModel', dynamic=False)
    if EMBEDDINGS is None:
        app.logger.error('Failed to load model')
        sys.exit(1)
    EMBEDDINGS.compile()
    latest = MODEL_PATH / 'nlp_model'
    EMBEDDINGS.load_weights(latest).expect_partial()
    train_generator = ((np.arange(len(tokenized)), [0 for _ in tokenized]),)
    dataset = tf.data.Dataset.from_tensor_slices(train_generator).batch(1024)
    EMBEDDINGS = EMBEDDINGS.predict(dataset)
    return EMBEDDINGS


# This is a POST request since it needs to take in a cube object
@app.route("/cube", methods=['POST'])
def cube_recommendations():
    num_recs = int(request.args.get("num_recs", 100))
    json_data = request.get_json()
    if json_data is None or "cube" not in json_data:
        error = "Must have a valid cube at the cube key in the json body."
        app.logger.error(error)
        return { "success": False, "error": error }, 400
    cube = json_data.get("cube")
    try:
        results = get_cube_recomendations(cube=cube, num_recs=num_recs)
    except Exception as e:
        app.logger.error(e)
        raise
    return results


# This is a POST request since it needs to take in a cube object
@app.route("/draft", methods=['POST'])
def draft_recommendations():
    app.logger.info('handling draftbot request')
    json_data = request.get_json()
    if json_data is None or "drafterState" not in json_data:
        error = "Must have a valid drafter state at the drafterState key in the json body."
        app.logger.error(error)
        return { "success": False, "error": error }, 400
    drafter_state = json_data.get("drafterState")
    try:
        results = get_draft_scores(drafter_state)
    except Exception as e:
        app.logger.error(e)
        raise
    return results

@app.route('/version', methods=['GET'])
def get_version():
    return { 'version': os.environ.get('MTGML_VERSION', 'alpha'), 'success': True }

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin',  "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response

app.logger.info('Started initialization.')
embeddings = get_embeddings()
app.logger.info('Got embeddings')
initialize_draftbots(MODEL_PATH, embeddings / 10_000, INT_TO_CARD, CARD_TO_INT, app.logger)
app.logger.info('Initialized draftbots')
initialize_recommender(MODEL_PATH, embeddings / 10_000, INT_TO_CARD, CARD_TO_INT, app.logger)
app.logger.info('Initialized recommender')
embeddings = []

app.run(host="0.0.0.0", port=8000, threaded=True)
