import re
import string

import nltk

nltk.download("punkt_tab", quiet=True)

PUNCT_PATTERN = f"[{string.punctuation}]"


def text_to_indices(text, vocab):
    indexed_text = []

    text = text.lower()
    text = re.sub(PUNCT_PATTERN, " ", text)

    for token in nltk.word_tokenize(text):
        if token in vocab:
            indexed_text.append(vocab[token])
        else:
            indexed_text.append(vocab["<UNK>"])

    return indexed_text