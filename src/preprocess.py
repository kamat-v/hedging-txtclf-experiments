# src/data/preprocess.py
# Sentence extraction and cleaning functions for earnings call transcripts.
# Migrated from 01_data_exploration.ipynb once cleaning logic was finalized.

from nltk.tokenize import sent_tokenize

BOILERPLATE_PHRASES = [
    "ladies and gentlemen",
    "thank you for standing by",
    "listen-only mode",
    "question-and-answer session",
    "conference is being recorded",
    "turn the call over",
    "investor relations website",
    "forward-looking statement",
    "non-gaap",
    "reconciliation",
    "slide presentation",
    "earnings release",
    "welcome to the",
    "good morning",
    "good afternoon", 
    "turn to slide",
    "form 10-k",
    "risk factors",
    "please note",
    "make some formal comments",
    "take your questions",
    "with me today",
    "i would like to take a moment",
    "that concludes",
    "turn the call back",
    "thank you for joining",
    "please disconnect",
    "question-and-answer portion",
]

def is_boilerplate(s):
    s_lower = s.lower()
    return any(phrase in s_lower for phrase in BOILERPLATE_PHRASES)

def clean_sentences(sentences):
    cleaned = []
    for s in sentences:
        # Remove speaker prefix if present
        if ":" in s:
            prefix = s.split(":")[0].strip()
            if len(prefix) < 40 and " " in prefix or prefix == "Operator":
                s = s.split(":", 1)[1].strip()
        # Skip operator boilerplate and bracketed instructions
        if s.startswith("["):
            continue
        if len(s) < 20:
            continue
        if is_boilerplate(s):
            continue
        cleaned.append(s)
    return cleaned

def extract_sentences (text):
    return sent_tokenize(text)