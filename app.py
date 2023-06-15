import os
import flask
from flask import Flask, request
from werkzeug.middleware.proxy_fix import ProxyFix
import json
from src.DocuDive import loadModel, getResult


def model_fn():
    return loadModel()

def predict(body: dict, llm):
    query = body["query"]
    answer, documents = getResult(query, llm)
    result = {
        "answer":answer,
        "Documents":{}
        }
    for doc in documents:
        result['Documents'][doc.metadata["source"]] = doc.page_content

    return result

app = Flask(__name__)

# Load the model by reading the `SM_MODEL_DIR` environment variable
# which is passed to the container by SageMaker (usually /opt/ml/model).
llm = model_fn()

# Since the web application runs behind a proxy (nginx), we need to
# add this setting to our app.
app.wsgi_app = ProxyFix(
    app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1
)


@app.route("/ping", methods=["GET"])
def ping():
    """
    Healthcheck function.
    """
    return "pong"


@app.route("/invocations", methods=["POST"])
def invocations():
    """
    Function which responds to the invocations requests.
    """
    body = request.json
    result = predict(body, llm)
    resultjson = json.dumps(result, indent=4)
    return flask.Response(response=resultjson, status=200, mimetype='application/json')

# Sample response to POST request
# {
#     "answer": " The context does not provide an answer to this question as it deals with Russia's actions, rather than a specific statement made by President Zelenskyy or any other person mentioned in the text.",
#     "Documents": {
#         "source_documents\\state_of_the_union.txt": "The United States is a member along with 29 other nations. \n\nIt matters. American diplomacy matters. American resolve matters. \n\nPutin’s latest attack on Ukraine was premeditated and unprovoked. \n\nHe rejected repeated efforts at diplomacy. \n\nHe thought the West and NATO wouldn’t respond. And he thought he could divide us at home. Putin was wrong. We were ready.  Here is what we did.   \n\nWe prepared extensively and carefully."
#     }
# }