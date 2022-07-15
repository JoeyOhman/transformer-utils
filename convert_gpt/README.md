## Convert Megatron (and NeMo) Checkpoint to HuggingFace

The conversion script `convert_megatron_gpt2_checkpoint.py` is a modified version of:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/megatron_gpt2/convert_megatron_gpt2_checkpoint.py

This looks promising, but so far no guarantees that everything works correctly.

### Setup
1. `git clone git@github.com:NVIDIA/Megatron-LM.git`
2. `cd Megatron-LM`
3. Move this repository's `convert_gpt/convert_megatron_gpt2_checkpoint.py` into the root of the cloned Megatron-LM


### Convert
1. Convert the model: `python convert_megatron_gpt2_checkpoint.py <megatron_checkpoint_file>`


### Inference
1. Change directory to this repository's `convert_gpt/`
2. Have the converted model path ready
3. Have the SentencePiece tokenizer path ready
4. Set paths in global variables in `convert_gpt/convert_megatron_gpt2_checkpoint.py`
5. Change `text` in `main()` to something nice
6. Profit
