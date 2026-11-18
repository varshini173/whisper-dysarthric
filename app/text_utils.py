import re

DIGITS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six",
          "7": "seven", "8": "eight", "9": "nine", "10": "ten"}


def normalize_text(t: str) -> str:
    """Lower-case, strip punctuation, spell out small numbers (same rule as in notebook 04)."""
    t = str(t or "").lower().replace("’", "'")
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return " ".join(DIGITS.get(w, w) for w in t.split()).strip()
