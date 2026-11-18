"""Fine-tuned Whisper recogniser (TORGO + UA-Speech, notebook 04)."""
import gc
import os

import numpy as np

from . import config


class Recognizer:
    def __init__(self, model_path=None, device=None):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.torch = torch
        self.model_path = model_path or config.find_asr_model()
        self.finetuned = os.path.isdir(self.model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Fine-tuning does not change the feature extractor or tokenizer. Colab (transformers 5.x) may not have
        # written preprocessor_config.json, so fall back to the original whisper-small files when it is missing.
        has_proc = os.path.exists(os.path.join(self.model_path, "preprocessor_config.json"))
        self.processor = WhisperProcessor.from_pretrained(self.model_path if (has_proc or not self.finetuned)
                                                          else config.BASE_PROCESSOR)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype=self.dtype).to(self.device).eval()
        for cfg in (self.model.generation_config, self.model.config):
            try:
                cfg.forced_decoder_ids = None
            except Exception:
                pass

    @property
    def mode(self):
        return ("fine-tuned on dysarthric speech (TORGO + UA-Speech)" if self.finetuned
                else f"generic {self.model_path} (NOT trained on dysarthric speech)")

    def transcribe(self, audio16k, n_best=config.N_BEST):
        """audio16k: float32 mono 16 kHz (<= 30 s). Returns the n-best list, best first."""
        torch = self.torch
        feats = self.processor.feature_extractor([audio16k], sampling_rate=16000,
                                                 return_tensors="pt").input_features.to(self.device, dtype=self.dtype)
        kw = dict(language="en", task="transcribe", max_new_tokens=60)
        n = max(1, int(n_best))
        try:
            with torch.no_grad():
                ids = (self.model.generate(feats, num_beams=n, num_return_sequences=n, **kw) if n > 1
                       else self.model.generate(feats, **kw))
        except torch.cuda.OutOfMemoryError:
            gc.collect(); torch.cuda.empty_cache()
            with torch.no_grad():
                ids = self.model.generate(feats, **kw)
        texts = self.processor.tokenizer.batch_decode(ids, skip_special_tokens=True)
        seen, out = set(), []
        for t in texts:                                    # keep order, drop duplicates
            t = t.strip()
            if t and t.lower() not in seen:
                seen.add(t.lower()); out.append(t)
        return out
