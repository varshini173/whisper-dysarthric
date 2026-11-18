"""Prompt matching: snap the recognised text to the closest phrase of the known prompt list
(TORGO prompts + UA-Speech words, saved by notebook 04 as lexicon.txt next to the model)."""
import difflib
from pathlib import Path

from .text_utils import normalize_text


class Lexicon:
    def __init__(self, phrases):
        self.phrases = sorted({normalize_text(p) for p in phrases if normalize_text(p)})
        self._set = set(self.phrases)

    @classmethod
    def from_model_dir(cls, model_dir):
        p = Path(model_dir) / "lexicon.txt"
        if not p.exists():
            return cls([])
        return cls(p.read_text(encoding="utf-8", errors="ignore").splitlines())

    def __len__(self):
        return len(self.phrases)

    def snap(self, candidates, cutoff=0.6):
        """candidates: n-best recognitions (best first). Returns (text, how) with how in exact/fuzzy/none."""
        cands = [normalize_text(c) for c in candidates if normalize_text(c)]
        if not cands or not self.phrases:
            return (cands[0] if cands else ""), "none"
        for c in cands:                                   # an n-best entry that IS a known prompt wins
            if c in self._set:
                return c, "exact"
        best, best_score = None, 0.0
        for rank, c in enumerate(cands):
            for m in difflib.get_close_matches(c, self.phrases, n=3, cutoff=cutoff):
                s = difflib.SequenceMatcher(None, c, m).ratio() - 0.02 * rank   # prefer higher-ranked guesses
                if s > best_score:
                    best, best_score = m, s
        if best is None:
            return cands[0], "none"
        return best, "fuzzy"
