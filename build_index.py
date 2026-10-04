import argparse
import hashlib
import json
import os
import shutil
import time

from common import (extract_json, image_files, load_config, load_media, norm_space, read_children, read_jsonl_gz,
                    write_jsonl_gz)
from index import dedup
from llm import LLM

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_KEEP = ("id", "node_type", "edge_id", "cluster_id", "edge_root_id", "size", "height", "summary", "meta")


def phrases(value, cap):
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    items = [" ".join(str(v) for v in x.values()) if isinstance(x, dict) else x for x in value]
    return dedup(items)[:cap]


def leaf_prompt(P, cfg, page, modality, has_image):
    part = P["leaf_" + modality]
    body = page["caption"] if modality == "image" else page["raw_data"]
    return P["leaf"].format(
        title=norm_space(page["title"])[:cfg["title_chars"]], modality=modality, body_label=part["body_label"],
        body=norm_space(body)[:cfg["body_chars"]], image_note=part["image_note"] if has_image else "",
        summary_extra=part["summary_extra"], keyword_hint=part["keyword_hint"], n=cfg["n_doc_query"],
        attr_kind=part["attr_kind"])


def parent_prompt(P, summaries, max_chars):
    lines, used = [], 0
    for i, s in enumerate(summaries, 1):
        line = f"{i}. {norm_space(s)}"
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line)
    return P["parent"].format(n_children=len(summaries), children="\n".join(lines))


def heights(children, ids):
    h = {}

    def visit(n):
        if n not in h:
            h[n] = 1 + max((visit(c) for c in children[n]), default=-1)
        return h[n]

    for n in ids:
        visit(n)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    P, ic = cfg["prompts"]["index"], cfg["index"]
    src, out = cfg["data"]["index_source"], cfg["data"]["index"]
    media = load_media(cfg["data"]["media"])
    images = image_files(cfg["data"]["images"])

    edge_ids = sorted(os.listdir(os.path.join(src, "edges")))
    edge_rows = {e: read_jsonl_gz(os.path.join(src, "edges", e, "nodes.jsonl.gz")) for e in edge_ids}
    server_rows = read_jsonl_gz(os.path.join(src, "server", "nodes.jsonl.gz"))
    leaves = [r for e in edge_ids for r in edge_rows[e] if r["node_type"] == "leaf"]
    print(f"{len(leaves)} leaves, {sum(map(len, edge_rows.values()))} edge nodes, {len(server_rows)} server nodes",
          flush=True)

    os.makedirs(out, exist_ok=True)
    log_path = os.path.join(out, "build_io.jsonl")
    done = {}
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done[r["key"]] = r["raw"]
    log = open(log_path, "a", encoding="utf-8")
    llm = LLM(cfg["models"]["index"], cfg["vllm"])

    def generate(keys, prompts, imgs, max_tokens, chunk):
        todo = [i for i, k in enumerate(keys) if k not in done]
        for s in range(0, len(todo), chunk):
            part = todo[s:s + chunk]
            outs = llm.generate(P["system"], [prompts[i] for i in part], [imgs[i] for i in part], max_tokens)
            for i, (text, _) in zip(part, outs):
                done[keys[i]] = text
                log.write(json.dumps({"key": keys[i], "raw": text}, ensure_ascii=False) + "\n")
            log.flush()
        return len(todo)

    # leaves
    t0 = time.time()
    keys = ["leaf|" + r["id"] for r in leaves]
    imgs = [images.get(str(r["page_id"])) if r["page_modality"] == "image" else None for r in leaves]
    prompts = [leaf_prompt(P, ic, media[str(r["page_id"])], r["page_modality"], bool(im)) for r, im in zip(leaves, imgs)]
    n = generate(keys, prompts, imgs, cfg["max_tokens"]["leaf"], ic["chunk"])
    parsed = 0
    for r, k in zip(leaves, keys):
        obj = extract_json(done[k])
        summary = norm_space(obj.get("summary"))
        entities = phrases(obj.get("entities"), 64)
        keywords = phrases(obj.get("keywords"), ic["n_keywords"])
        doc_query = phrases(obj.get("doc_query"), ic["n_doc_query"])
        parsed += bool(summary and (entities or doc_query))
        # a reply that does not parse keeps the source index's summary and meta
        old = r.get("meta") or {}
        r["summary"] = summary or norm_space(r.get("summary"))
        r["meta"] = {"schema": "v9", "modality": r["page_modality"], "entities": entities or old.get("entities", []),
                     "keywords": keywords, "doc_query": doc_query or old.get("doc_query", [])}
    leaf_seconds = time.time() - t0
    print(f"leaves: {n} generated, {parsed}/{len(leaves)} parsed, {leaf_seconds:.0f}s", flush=True)

    def summarize(rows, children, scope):
        by_id = {r["id"]: r for r in rows}
        h = heights(children, by_id)
        calls = 0
        for level in range(1, max(h.values()) + 1):
            nodes = [r for r in rows if h[r["id"]] == level]
            kids = [[by_id[c]["summary"] for c in children[r["id"]]] for r in nodes]
            keys = [f"parent|{scope}|{r['id']}|" + hashlib.sha1("\x00".join(ks).encode()).hexdigest()[:16]
                    for r, ks in zip(nodes, kids)]
            prompts = [parent_prompt(P, ks, ic["parent_chars"]) for ks in kids]
            generate(keys, prompts, [None] * len(keys), cfg["max_tokens"]["parent"], len(keys))
            for r, k, ks in zip(nodes, keys, kids):
                r["summary"] = norm_space(extract_json(done[k]).get("summary")) or next((norm_space(s) for s in ks if s), "")
            calls += len(nodes)
            print(f"{scope} level {level}: {len(nodes)} nodes", flush=True)
        return calls

    # internal summaries: edge trees, then the server tree over the cluster roots
    t1 = time.time()
    calls = 0
    for e in edge_ids:
        for r in edge_rows[e]:
            if r["node_type"] != "leaf":
                r["meta"] = {"schema": "v9-internal"}
        calls += summarize(edge_rows[e], read_children(os.path.join(src, "edges", e)), e)
    cluster_summary = {r["id"]: r["summary"] for e in edge_ids for r in edge_rows[e]
                       if r["node_type"] == "edge_cluster_root"}
    for r in server_rows:
        r["meta"] = {"schema": "v9-internal"}
        if r["node_type"] == "server_leaf":
            r["summary"] = cluster_summary[r["edge_root_id"]]
    calls += summarize(server_rows, read_children(os.path.join(src, "server")), "server")
    parent_seconds = time.time() - t1
    log.close()

    for e in edge_ids:
        d = os.path.join(out, "edges", e)
        os.makedirs(d, exist_ok=True)
        write_jsonl_gz(edge_rows[e], os.path.join(d, "nodes.jsonl.gz"))
        for f in ("tree_structure.jsonl.gz", "metadata.json.gz"):
            shutil.copy(os.path.join(src, "edges", e, f), os.path.join(d, f))
    d = os.path.join(out, "server")
    os.makedirs(d, exist_ok=True)
    write_jsonl_gz([{k: r[k] for k in SERVER_KEEP} for r in server_rows], os.path.join(d, "nodes.jsonl.gz"))
    for f in ("tree_structure.jsonl.gz", "metadata.json.gz"):
        shutil.copy(os.path.join(src, "server", f), os.path.join(d, f))
    manifest = {"built": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": src, "model": cfg["models"]["index"],
                "leaves": len(leaves), "leaves_parsed": parsed, "leaf_seconds": leaf_seconds,
                "parent_summaries": calls, "parent_seconds": parent_seconds, "server_fields": list(SERVER_KEEP)}
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
