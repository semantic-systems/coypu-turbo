from InstructorEmbedding import INSTRUCTOR
from flask import Flask, jsonify, request
from flask_healthz import healthz
from typing import List

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
        sentence = request.json['sentence']
        instruction = request.json['instruction']

        if isinstance(instruction, str) and isinstance(sentence, str):
            prompts: List[List] = [[instruction, sentence]]
        elif isinstance(instruction, str) and isinstance(sentence, list):
            prompts: List[List] = [[instruction, s] for s in sentence]
        elif isinstance(instruction, list) and isinstance(sentence, str):
            prompts: List[List] = [[i, sentence] for i in instruction]
        elif isinstance(instruction, list) and isinstance(sentence, list):
            prompts: List[List] = [[instruction[i], sentence[i]] for i in range(len(sentence))]
        else:
            raise TypeError(f"Type Error: Type for both instruction and sentence need to be List or String. If both are List, their length must be same.")
        embeddings = model.encode(prompts, batch_size=256).tolist()  ## screw you stupid type hint
        response = {'embeddings': embeddings}
        http_code = 200

    else:
        response = {'error': 'no valid input'}
        http_code = 400

    return jsonify(response), http_code


if __name__ == "__main__":
    app.run(host='0.0.0.0')
