import json
import requests
from flask import Flask, jsonify, request
from flask_healthz import healthz
from fastchat.conversation import get_default_conv_template, SeparatorStyle
from fastchat.serve.inference import compute_skip_echo_len


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


def main(message: str, temperature: float = 0.1, max_new_tokens: int = 1024):
    model_name = 'vicuna-13b-v1.5'
    worker_addr = 'http://vicuna_worker:5001'

    conv = get_default_conv_template(model_name).copy()
    conv.append_message(conv.roles[0], message)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    headers = {"User-Agent": "fastchat Client"}
    pload = {
        "model": model_name,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "stop": conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
    }
    response = requests.post(worker_addr + "/worker_generate_stream", headers=headers,
            json=pload, stream=True)

    # print(f"{conv.roles[0]}: {message}")
    print(f"response {response}")
    output = ""
    for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            skip_echo_len = compute_skip_echo_len(model_name, conv, prompt)
            output += data["text"][skip_echo_len:].strip()
            # print(f"{conv.roles[1]}: {output}", end="\r")
    # print("")
    return output


@app.route('/', methods=['POST'])
def flask():
    authenticated = False

    if 'key' in request.json:
        key = request.json['key']
        if key == 'M7ZQL9ELMSDXXE86': authenticated = True

    if authenticated == False:
        response = {'error': 'no valid API key'}
        http_code = 401

    elif 'prompt' in request.json:
        prompt = request.json['prompt']
        temperature = request.json.get('temperature', 0.1)
        max_new_tokens = request.json.get('max_new_tokens', 1024)

        response = main(prompt, temperature, max_new_tokens)
        response = {'content': response,
                    'meta': {"turbo_version": "vicuna-13b-v1.5",
                             "temperature": temperature,
                             "max_new_tokens": max_new_tokens}}
        http_code = 200

    else:
        response = {'error': 'no valid input'}
        http_code = 400

    return jsonify(response), http_code


if __name__ == "__main__":
    app.run(host='0.0.0.0')
