from transformers import GPT2LMHeadModel, AutoTokenizer, pipeline
# import tokenizers
import sentencepiece as spm
import torch


PATH_MODEL_DIR = "/home/joey/code/gpt_convert/Megatron-LM/HF_model_converted"
TOKENIZER_PATH = PATH_MODEL_DIR + "/model.model"


def load_model_and_tokenizer():
    # EDIT THESE PATHS
    model = GPT2LMHeadModel.from_pretrained(PATH_MODEL_DIR)
    model.eval()
    # _tokenizer = AutoTokenizer.from_pretrained(tok_path)
    # pipe = pipeline('text-generation', model=base_model, tokenizer=_tokenizer, device=0)
    # print(pipe("hej"))

    sp_tokenizer = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    return model, sp_tokenizer


def encode_text(sp_tokenizer, text):
    tokenized_text = sp_tokenizer.encode(text)

    tokenized_text = torch.tensor(tokenized_text, dtype=torch.long)
    tokenized_text.unsqueeze_(0)

    return tokenized_text


def generate(sp_tokenizer, model, text):
    tokenized_text = encode_text(sp_tokenizer, text)

    # gen = base_model.generate(tokenized_text, repetition_penalty=2.)
    gen = model.generate(tokenized_text)

    list_gen = gen[0].numpy().tolist()

    decoded = sp_tokenizer.decode(list_gen)
    return decoded


def main():
    model, sp_tokenizer = load_model_and_tokenizer()

    text = "Hej jag heter Ariel"

    generated_text = generate(sp_tokenizer, model, text)
    print(generated_text)


if __name__ == '__main__':
    main()
