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


def llama3(messages, temperature, max_new_tokens, top_p):
    try:
        url = 'https://turbo.skynet.coypu.org/'
        response = requests.post(url,
                                json={"messages": messages,
                                       "temperature": temperature,
                                       "max_new_tokens": max_new_tokens,
                                       "top_p": top_p},
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


        response = llama3(messages, temperature, max_new_tokens, top_p)
        response = {'content': response,
                    'meta': {"turbo_version": "llama 3",
                             "temperature": temperature,
                             "max_new_tokens": max_new_tokens,
                             "top_p": top_p
                             }}
        http_code = 200

    else:
        response = {'error': 'no valid input'}
        http_code = 400

    return jsonify(response), http_code


if __name__ == "__main__":
    app.run(host='0.0.0.0')
