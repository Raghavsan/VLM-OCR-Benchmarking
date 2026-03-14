import re
from jiwer import cer, wer

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def compute_cer(prediction, reference):
    return min(cer(normalize_text(reference), normalize_text(prediction)), 1.0)

def compute_wer(prediction, reference):
    return min(wer(normalize_text(reference), normalize_text(prediction)), 1.0)

def compute_accuracy(prediction, reference):
    c = compute_cer(prediction, reference)
    w = compute_wer(prediction, reference)
    return max(0.0, (1.0 - ((c + w) / 2.0)) * 100)
