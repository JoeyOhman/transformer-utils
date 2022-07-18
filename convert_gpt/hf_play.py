from transformers import GPT2LMHeadModel, AutoTokenizer, pipeline
# import tokenizers
import sentencepiece as spm
import torch
from typing import Tuple, List


PATH_MODEL_DIR = "/home/joey/code/gpt_convert/Megatron-LM/HF_model_converted"
TOKENIZER_PATH = PATH_MODEL_DIR + "/model.model"


def load_model_and_tokenizer() -> Tuple[GPT2LMHeadModel, spm.SentencePieceProcessor]:
    # EDIT THESE PATHS
    model = GPT2LMHeadModel.from_pretrained(PATH_MODEL_DIR)
    model.eval()
    # _tokenizer = AutoTokenizer.from_pretrained(tok_path)
    # pipe = pipeline('text-generation', model=base_model, tokenizer=_tokenizer, device=0)
    # print(pipe("hej"))

    sp_tokenizer = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    return model, sp_tokenizer


def encode_text(sp_tokenizer: spm.SentencePieceProcessor, text: str) -> List[int]:
    token_ids = sp_tokenizer.encode(text)
    return token_ids


def decode_token_ids(sp_tokenizer: spm.SentencePieceProcessor, token_ids: List[int]) -> str:
    text = sp_tokenizer.decode(token_ids)
    return text


def generate(model: GPT2LMHeadModel, input_token_ids: List[int]) -> List[int]:

    input_token_ids = torch.tensor(input_token_ids, dtype=torch.long)
    input_token_ids.unsqueeze_(0)

    # generated_output = base_model.generate(input_token_ids, repetition_penalty=2.)
    generated_output = model.generate(input_token_ids)
    generated_token_ids = generated_output[0].numpy().tolist()

    return generated_token_ids


def main(input_texts):
    model, sp_tokenizer = load_model_and_tokenizer()

    for input_text in input_texts:
        input_token_ids = encode_text(sp_tokenizer, input_text)
        generated_token_ids = generate(model, input_token_ids)  # generated_token_ids includes input_token_ids
        generated_text = decode_token_ids(sp_tokenizer, generated_token_ids)

        print(generated_text)


if __name__ == '__main__':
    _input_texts = [
        "Hej jag heter Ariel",
    ]
    main(_input_texts)
