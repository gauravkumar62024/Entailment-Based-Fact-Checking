import re
import spacy

# Load once (global)
NLP = spacy.load("xx_sent_ud_sm")
def split_into_sentences(text: str):
    """
    Multilingual sentence split
    """
    doc = NLP(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]

import re

# Load once (global)


def clean_web_text(text: str) -> str:
    """
    Light cleaning: remove markdown, links, junk headers
    """
    text = re.sub(r"\[.*?\]\(.*?\)", " ", text)  # markdown links
    text = re.sub(r"http\S+", " ", text)         # raw URLs
    text = re.sub(r"\s+", " ", text)              # whitespace
    return text.strip()