"""Summary embeddings for every node of the index, keyed "server|<id>" and "edge:<edge>|<id>".

    python embed_index.py --config config.yaml
"""
import argparse
import os

import numpy as np

from common import load_config, read_jsonl_gz
from embedder import Embedder

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    index_dir, out = cfg["data"]["index"], cfg["data"]["node_embeddings"]

    ids, texts = [], []
    for r in read_jsonl_gz(os.path.join(index_dir, "server", "nodes.jsonl.gz")):
        ids.append(f"server|{r['id']}")
        texts.append(r["summary"])
    for e in sorted(os.listdir(os.path.join(index_dir, "edges"))):
        for r in read_jsonl_gz(os.path.join(index_dir, "edges", e, "nodes.jsonl.gz")):
            ids.append(f"edge:{e}|{r['id']}")
            texts.append(r["summary"])

    uniq = list(dict.fromkeys(texts))
    model = Embedder(cfg["models"]["embedding"], **cfg["embedding"])
    vecs = dict(zip(uniq, model.encode(uniq)))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, node_ids=np.array(ids), node_emb=np.stack([vecs[t] for t in texts]))
    print(f"{len(ids)} nodes, {len(uniq)} distinct summaries -> {out}")


if __name__ == "__main__":
    main()
