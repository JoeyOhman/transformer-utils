## Convert Megatron (and NeMo) Checkpoint to HuggingFace

The conversion script `convert_gpt/convert_megatron_gpt2_checkpoint.py` is a modified version of:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/megatron_gpt2/convert_megatron_gpt2_checkpoint.py

This looks promising, but so far no guarantees that everything works correctly.

### Setup
0. `pip install -r requirements.txt`


### Convert
1. Convert the model: `python convert_gpt/convert_megatron_gpt2_checkpoint.py <megatron_checkpoint_file>`


### Inference

2. In `convert_gpt/hf_play.py`
- set paths (model, tokenizer) in global variables 
- change `_input_texts` if wanted 

3. Run `python convert_gpt/hf_play.py`
