# This script assumes a cased BertWordPiece tokenizer

import argparse
import os
from pathlib import Path

from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, BertTokenizerFast, AutoTokenizer


# NOT USED
def vocab_to_tokenizer_old(vocab_path, out_path):
    # vocab -> tokenizers
    print(vocab_path)
    tokenizer = BertWordPieceTokenizer(vocab_path, lowercase=False)
    print("id of [PAD]:", tokenizer.token_to_id("[PAD]"))
    tokenizer.add_special_tokens(["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"])
    Path("./tmpdir").mkdir(exist_ok=True)
    tokenizer.save("./tmpdir/tmp.json", pretty=False)

    # tokenizers -> transformers
    tokenizer_hf = BertTokenizer.from_pretrained("./tmpdir", do_lower_case=False)
    os.remove("./tmpdir/tmp.json")
    os.rmdir("./tmpdir")
    tokenizer_hf.save_pretrained(out_path if out_path else "./hf_tokenizer/")


def vocab_to_tokenizer(vocab_path, out_path):
    tokenizer_hf = BertTokenizer.from_pretrained(vocab_path, do_lower_case=False)
    special_tokens_dict = {
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "unk_token": "[UNK]",
        "mask_token": "[MASK]",
    }

    tokenizer_hf.add_special_tokens(special_tokens_dict)
    tokenizer_hf.save_pretrained(out_path if out_path else "./hf_tokenizer/")


def how_to_load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("./hf_tokenizer/")
    tokenized = tokenizer("Hej där [MASK]")
    decoded = tokenizer.decode(tokenized['input_ids'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("vocab_path", type=str,
                        help="Path to the vocab text file with one mapping per line.")
    parser.add_argument("--save_path", type=str, default="",
                        help="Path to the save directory for the tokenizer. Defaults to hf_tokenizer/ in current dir.")
    args = parser.parse_args()

    vocab_to_tokenizer(args.vocab_path, args.save_path)


