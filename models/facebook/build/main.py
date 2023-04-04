from flask import abort, Flask, jsonify, request
from flask_healthz import healthz
import os, sys
import torch
from example import setup_model_parallel, load


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


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "5284"
os.environ["USE_TF"] = "0"
os.environ["NCCL_SOCKET_IFNAME"] = "lo"

CKPT_PATH = "/data/llama/7B/"
TOKENIZER_PATH = "/data/llama/tokenizer.model"

ckpt_dir = CKPT_PATH
tokenizer_path = TOKENIZER_PATH
temperature = 0.8
top_p = 0.95
max_seq_len = 512
max_gen_seq_len = 256
max_batch_size = 32


local_rank, world_size = setup_model_parallel()

if local_rank > 0:
    sys.stdout = open(os.devnull, "w")

generator = load(
    ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
)

@app.route('/', methods=['POST'])
def flask():
    authenticated = False

    if 'key' in request.json:
        key = request.json['key']
        if (key == '2LL6Y5MGRCFQT68Y'): authenticated = True

    if (authenticated == False):
        response = {'error': 'no valid API key'}
        http_code = 401

    elif ('message' in request.json):
        temperature = request.json.get('temperature', 0.8)
        top_p = request.json.get('top_p', 0.95)
        max_gen_seq_len = request.json.get('max_gen_seq_len', 256)
        generated_sequence = generator.generate([str(request.json['message'])], max_gen_len=max_gen_seq_len, temperature=temperature, top_p=top_p)
        torch.cuda.empty_cache()
        response = {'prediction': generated_sequence, 'turbo_version': "fb_wo_fine_tuning"}
        http_code = 200

    else:
        response = {'error': 'no valid input'}
        http_code = 400

    return jsonify(response), http_code


if __name__ == "__main__":
    app.run(host='0.0.0.0')
