from InstructorEmbedding import INSTRUCTOR
from flask import Flask, jsonify, request
from flask_healthz import healthz
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

app.register_blueprint(healthz, url_prefix="/healthz")


def liveness():
    pass


def readiness():
    pass


app.config.update(
    HEALTHZ={
        "live": app.name + ".liveness",
        "ready": app.name + ".readiness"
    }
)

model = INSTRUCTOR('hkunlp/instructor-large')


@app.route('/', methods=['POST'])
def flask():
    authenticated = False

    if 'key' in request.json:
        key = request.json['key']
        if key == 'B48KSZDAXDQT1NX2': authenticated = True

    if authenticated == False:
        response = {'error': 'no valid API key'}
        http_code = 401

    elif ('sentence' in request.json) and ('instruction' in request.json):
        sentence = str(request.json['sentence'])
        instruction = str(request.json['instruction'])
        embeddings = model.encode([[instruction, sentence]]).tolist()[0]
        response = {'embeddings': embeddings}
        http_code = 200

    else:
        response = {'error': 'no valid input'}
        http_code = 400

    return jsonify(response), http_code


if __name__ == "__main__":
    app.run(host='0.0.0.0')
