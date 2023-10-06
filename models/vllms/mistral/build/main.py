from flask import abort, Flask, jsonify, request
from flask_healthz import healthz
import os, sys
from vllm import LLM, SamplingParams

app = Flask(__name__)

app.register_blueprint(healthz, url_prefix="/healthz")


def liveness():
    pass


def readiness():
    pass


app.config.update(
    HEALTHZ = {
        "live": app.name + ".liveness",
        "ready": app.name + ".readiness"
    }
)

llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.1")


@app.route('/', methods=['POST'])
def flask():
    authenticated = False

    if 'key' in request.json:
        key = request.json['key']
        if (key == '2LL6Y5MGRCFQT68Y'): authenticated = True

    if (authenticated == False):
        response = {'error': 'no valid API key'}
        http_code = 401

    elif ('prompt' in request.json):
        temperature = request.json.get('temperature', DEFAULT_TEMPERATURE)
        top_p = request.json.get('top_p', DEFAULT_TOP_P)
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p)
        generated_sequence = llm.generate([str(request.json['prompt'])], sampling_params)
        torch.cuda.empty_cache()
        response = {'content': generated_sequence[0],
                    'meta': {"turbo_version": "llama-vanilla",
                             "temperature": temperature,
                             "top_p": top_p}}
        http_code = 200

    else:
        response = {'error': 'no valid input'}
        http_code = 400

    return jsonify(response), http_code


if __name__ == "__main__":
    # app.run(host='0.0.0.0')
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.1")
    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
