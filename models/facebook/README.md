# Facebook implementation of Llama

### Resources
repo: [link](https://github.com/facebookresearch/llama)

models: /coypu/static-data/models/llama on Skynet

### Requirements
python = 3.9 (does not work for 3.10)

### Setup
```
pip install -r requirements.txt
pip install -e .
```

### Inference
#### Specification
- max. input sequence length: 512 (max for a 24GB GPU on Skynet)
- max. generated sequence length: 256
