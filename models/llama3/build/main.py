import os
import requests
from flask import Flask, jsonify, request
from flask_healthz import healthz


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


def llama3(messages, temperature, max_new_tokens, top_p, max_seq_len, max_gen_len):
    try:
        url = 'https://llm-chat.skynet.coypu.org/generate_text'
        username = os.environ.get('USERNAME', None)
        password = os.environ.get('PASSWORD', None)
        response = requests.post(url,
                                json={"messages": messages,
                                       "temperature": temperature,
                                       "max_new_tokens": max_new_tokens,
                                       "top_p": top_p,
                                       "max_seq_len": max_seq_len,
                                       "max_gen_len": max_gen_len},
                                auth=(username, password)
                                ).json()
        return response.get('generated_text')
    except Exception as e:
        return e

@app.route('/', methods=['POST'])
def flask():
    if 'messages' in request.json:
        messages = request.json['messages']
        temperature = request.json.get('temperature', 0.7)
        top_p = request.json.get('top_p', 0.9)
        max_new_tokens = request.json.get('max_new_tokens', 256)
        max_seq_len = request.json.get('max_seq_len', 1024)
        max_gen_len = request.json.get('max_gen_len', 512)

        response = llama3(messages, temperature, max_new_tokens, top_p, max_seq_len, max_gen_len)
        print(response)
        response = {'content': response,
                    'meta': {"turbo_version": "llama 3",
                             "temperature": temperature,
                             "max_new_tokens": max_new_tokens,
                             "top_p": top_p,
                             "max_seq_len": max_seq_len,
                             "max_gen_len": max_gen_len,
                             }}
        http_code = 200

    else:
        response = {'error': 'no valid input'}
        http_code = 400

    return jsonify(response), http_code


if __name__ == "__main__":
    app.run(host='0.0.0.0')
