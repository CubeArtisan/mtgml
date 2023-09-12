from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import flask

try:
    import tflite_runtime.interpreter as tflite
except ModuleNotFoundError:
    import tensorflow.lite as tflite

from opentelemetry import trace
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.cloud_trace_propagator import CloudTraceFormatPropagator
from opentelemetry.resourcedetector.gcp_resource_detector import (
    GoogleCloudResourceDetector,
)
from opentelemetry.sdk.resources import get_aggregated_resources
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from mtgml.server.cube_recommender import get_cube_recomendations
from mtgml.server.deck_builder import get_deck_scores
from mtgml.server.draftbots import get_draft_scores

Flask = flask.Flask
request = flask.request

try:
    resources = get_aggregated_resources([GoogleCloudResourceDetector(raise_on_error=True)])

    # set_global_textmap(CloudTraceFormatPropagator())

    tracer_provider = TracerProvider(resource=resources)
    cloud_trace_exporter = CloudTraceSpanExporter()
    tracer_provider.add_span_processor(
        # BatchSpanProcessor buffers spans and sends them in batches in a
        # background thread. The default parameters are sensible, but can be
        # tweaked to optimize your performance
        BatchSpanProcessor(cloud_trace_exporter)
    )
except Exception:
    tracer_provider = TracerProvider()
    # tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    logging.error("Failed to initialize cloud tracing")

trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer(__name__)

app = Flask(__name__)


def stdout_log_filter(record):
    return record.levelno < logging.WARNING


stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.addFilter(stdout_log_filter)
stdout_formatter = logging.Formatter("%(asctime)s - mtgml-server-stdout - %(levelname)s - %(message)s")
stdout_handler.setFormatter(stdout_formatter)
stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.WARNING)
stderr_formatter = logging.Formatter("%(asctime)s - mtgml-server-stderr - %(levelname)s - %(message)s")
stderr_handler.setFormatter(stderr_formatter)

logger = app.logger
logger.propagate = False
logger.handlers.extend((stdout_handler, stderr_handler))
logger.setLevel(logging.INFO)
logger.info("Starting up the server")

try:
    FlaskInstrumentor().instrument_app(app)
except Exception:
    logger.error("Failed to attach instrumenter to flask app.")

MODELS = {}
AUTH_TOKENS = os.environ.get("MTGML_AUTH_TOKENS", "testing").split(";")
AUTH_ENABLED = os.environ.get("MTGML_AUTH_ENABLED", "False") == "True"


@app.before_request
def verify_request():
    if not AUTH_ENABLED or "/version" in request.path:
        return None
    else:
        token = request.args.get("auth_token", None)
        if token is None or token not in AUTH_TOKENS:
            return {"success": False, "error": "Not authorized"}, 401
        else:
            trace.get_current_span().set_attribute("authorized", True)
            return None


AVAILABLE_MODELS = [
    str(path.name).split("_model.tflite")[0] for path in Path(f"ml_files/tflite").glob("*_model.tflite")
]


def get_model_dict(name="prod"):
    global INT_TO_CARD
    global CARD_TO_INT
    if name not in MODELS:
        model_path = Path(f"ml_files/tflite")
        model = tflite.Interpreter(model_path=str(model_path / f"{name}_model.tflite"), num_threads=2)
        model.allocate_tensors()
        with open(model_path / f"{name}_int_to_oracle_id.json", "r") as map_file:
            int_to_card = json.load(map_file)
        card_to_int = {v: k for k, v in enumerate(int_to_card)}
        original_to_new_path = model_path / f"{name}_original_to_new_index.json"
        if original_to_new_path.exists():
            with original_to_new_path.open("r") as fp:
                original_to_new_index = json.load(fp)
        else:
            logger.warn("No original_to_new_index.json file found.")
            original_to_new_index = tuple(range(len(int_to_card) + 1))
        MODELS[name] = {
            "model": model,
            "original_to_new_index": original_to_new_index,
            "int_to_card": int_to_card,
            "card_to_int": card_to_int,
        }
    return MODELS[name]


# This is a POST request since it needs to take in a cube object
@app.route("/cube", methods=["POST"])
def cube_recommendations():
    num_recs = int(request.args.get("num_recs", 100))
    trace.get_current_span().set_attribute("num_recs", num_recs)
    json_data = request.get_json()
    if json_data is None or "cube" not in json_data:
        error = "Must have a valid cube at the cube key in the json body."
        app.logger.error(error)
        return {"success": False, "error": error}, 400
    cube = json_data.get("cube")
    model_dict = get_model_dict("prod")
    results = get_cube_recomendations(
        cube=cube,
        model=model_dict["model"],
        card_to_int=model_dict["card_to_int"],
        int_to_card=model_dict["int_to_card"],
        tracer=tracer,
        num_recs=num_recs,
        original_to_new_index=model_dict["original_to_new_index"],
    )
    return results, 200


# This is a POST request since it needs to take in a drafter state object
@app.route("/draft", methods=["POST"])
def draft_recommendations():
    json_data = request.get_json()
    if json_data is None or "drafterState" not in json_data:
        error = "Must have a valid drafter state at the drafterState key in the json body."
        app.logger.error(error)
        return {"success": False, "error": error}, 400
    name = request.args.get("model_type", "prod")
    trace.get_current_span().set_attribute("model_name", name)
    drafter_state = json_data.get("drafterState")
    model_dict = get_model_dict(name)
    results = get_draft_scores(
        drafter_state,
        model=model_dict["model"],
        card_to_int=model_dict["card_to_int"],
        tracer=tracer,
        original_to_new_index=model_dict["original_to_new_index"],
    )
    return results, 200


# This is a POST request since it needs to take in a pool object
@app.route("/deck", methods=["POST"])
def deck_building_recommendations():
    json_data = request.get_json()
    if json_data is None or "deckState" not in json_data:
        error = "Must have a valid drafter state at the drafterState key in the json body."
        app.logger.error(error)
        return {"success": False, "error": error}, 400
    name = request.args.get("model_type", "prod")
    trace.get_current_span().set_attribute("model_name", name)
    deck_state = json_data.get("deckState")
    model_dict = get_model_dict(name)
    results = get_deck_scores(
        deck_state,
        model=model_dict["model"],
        card_to_int=model_dict["card_to_int"],
        int_to_card=model_dict["int_to_card"],
        tracer=tracer,
        original_to_new_index=model_dict["original_to_new_index"],
    )
    return results, 200


@app.route("/oracle-ids", methods=["GET"])
def get_oracle_ids():
    name = request.args.get("model_type", "prod")
    trace.get_current_span().set_attribute("model_name", name)
    model_dict = get_model_dict(name)
    card_to_int = model_dict["card_to_int"]
    original_to_new_index = model_dict["original_to_new_index"]
    return {"oracleIds": [oracle for oracle, idx in card_to_int.items() if original_to_new_index[idx] > 0]}, 200


@app.route("/version", methods=["GET"])
def get_version():
    return {
        "version": os.environ.get("MTGML_VERSION", "development"),
        "success": True,
        "models": AVAILABLE_MODELS,
    }, 200


@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)
