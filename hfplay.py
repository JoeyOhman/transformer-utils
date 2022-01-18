import argparse

from transformers import AutoModelForMaskedLM, AutoConfig, AutoTokenizer, pipeline


def get_model_tokenizer(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # tokenized = tokenizer("Hej där [MASK]")
    # decoded = tokenizer.decode(tokenized['input_ids'])
    # print(decoded)

    # config = AutoConfig.from_pretrained(model_name_or_path)
    # print(config)

    # model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, config=config)
    model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)

    return tokenizer, model


def main(model_name_or_path):
    tokenizer, model = get_model_tokenizer(model_name_or_path)
    model.eval()

    mlm = pipeline("fill-mask", model=model, tokenizer=tokenizer)

    text = "Huvudstaden i Frankrike är [MASK]."
    res = mlm(text, top_k=10)
    top_k_tuples = [(r['score'], r['token_str']) for r in res]
    top_k = sorted(top_k_tuples, key=lambda x: x[0], reverse=True)

    print(f"{'Token':<15} Probability")
    for (p, tok) in top_k:
        print(f"{tok:<15} {round(p, 3)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name_or_path", type=str, help="Path to model+tokenizer directory.")
    args = parser.parse_args()
    main(args.model_name_or_path)
