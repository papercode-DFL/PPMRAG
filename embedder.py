import gc
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class Embedder:
    """Qwen3 embedding model: last-token pooling, L2-normalised, no instruction prefix."""

    def __init__(self, path, batch_size, max_length):
        self.tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(path, dtype=torch.float16, local_files_only=True,
                                               trust_remote_code=True).to("cuda").eval()
        self.batch_size, self.max_length = batch_size, max_length

    @torch.inference_mode()
    def encode(self, texts):
        out = []
        for s in range(0, len(texts), self.batch_size):
            batch = self.tok(texts[s:s + self.batch_size], padding=True, truncation=True, max_length=self.max_length,
                             return_tensors="pt").to("cuda")
            hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"]
            if mask[:, -1].all():
                vec = hidden[:, -1]
            else:
                vec = hidden[torch.arange(len(hidden), device=hidden.device), mask.sum(dim=1) - 1]
            out.append(F.normalize(vec, p=2, dim=1).cpu().numpy().astype(np.float32))
        return np.concatenate(out)

    def timed_encode(self, texts):
        torch.cuda.synchronize()
        t = time.perf_counter()
        vecs = self.encode(texts)
        torch.cuda.synchronize()
        return vecs, time.perf_counter() - t

    def close(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
