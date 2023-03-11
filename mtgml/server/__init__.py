import json
import logging
import os
import sys
from pathlib import Path

import flask
import tflite_runtime.interpreter as tflite
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

MODEL_PATH = Path("ml_files/testing_tflite")
MODEL = None
with open(MODEL_PATH / "int_to_oracle_id.json", "r") as map_file:
    int_to_card = json.load(map_file)
original_to_new_path = MODEL_PATH / "original_to_new_index.json"
if original_to_new_path.exists():
    with original_to_new_path.open("r") as fp:
        original_to_new_index = json.load(fp)
else:
    logger.warn("No original_to_new_index.json file found.")
    original_to_new_index = tuple(range(len(int_to_card) + 1))
CARD_TO_INT = {v: k for k, v in enumerate(int_to_card)}
INT_TO_CARD = int_to_card
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
            return None


def get_model():
    global MODEL
    if MODEL is None:
        MODEL = tflite.Interpreter(model_path=str(MODEL_PATH / "combined_model.tflite"))
        MODEL.allocate_tensors()
    return MODEL


# This is a POST request since it needs to take in a cube object
@app.route("/cube", methods=["POST"])
def cube_recommendations():
    num_recs = int(request.args.get("num_recs", 100))
    json_data = request.get_json()
    if json_data is None or "cube" not in json_data:
        error = "Must have a valid cube at the cube key in the json body."
        app.logger.error(error)
        return {"success": False, "error": error}, 400
    cube = json_data.get("cube")
    model = get_model()
    results = get_cube_recomendations(
        cube=cube,
        model=model,
        card_to_int=CARD_TO_INT,
        int_to_card=INT_TO_CARD,
        tracer=tracer,
        num_recs=num_recs,
        original_to_new_index=original_to_new_index,
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
    drafter_state = json_data.get("drafterState")
    model = get_model()
    results = get_draft_scores(
        drafter_state, model=model, card_to_int=CARD_TO_INT, tracer=tracer, original_to_new_index=original_to_new_index
    )
    return results, 200


@app.route("/version", methods=["GET"])
def get_version():
    return {"version": os.environ.get("MTGML_VERSION", "alpha"), "success": True}, 200


@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response


get_model()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)
