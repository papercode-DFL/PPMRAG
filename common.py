import gzip
import hashlib
import json
import os
import re

import yaml


def load_config(path, overrides=()):
    """Read the YAML config, apply `a.b=value` overrides and resolve data paths against data.root."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, value = item.split("=", 1)
        *parents, last = key.split(".")
        node = cfg
        for k in parents:
            node = node[k]
        node[last] = yaml.safe_load(value)
    with open(os.path.join(os.path.dirname(os.path.abspath(path)), cfg["prompts"]), encoding="utf-8") as f:
        cfg["prompts"] = yaml.safe_load(f)
    data = cfg["data"]
    for k in data:
        if k != "root":
            data[k] = os.path.join(data["root"], data[k])
    return cfg


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path, indent=1):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl_gz(rows, path):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_children(tree_dir):
    rows = read_jsonl_gz(os.path.join(tree_dir, "tree_structure.jsonl.gz"))
    return {r["id"]: [c for c in (r["left_id"], r["right_id"]) if c] for r in rows}


def load_media(path):
    return {str(m["id"]): m for m in load_json(path)}


def image_files(image_dir):
    """page id -> image path"""
    return {os.path.splitext(f)[0]: os.path.join(image_dir, f) for f in os.listdir(image_dir)}


def digest(*parts):
    return hashlib.sha1("\x00".join(parts).encode("utf-8")).hexdigest()


def extract_json(text):
    """The JSON object in a model reply ({} if there is none)."""
    try:
        obj = json.loads(text)
    except ValueError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            obj = json.loads(m.group(0)) if m else {}
        except ValueError:
            obj = {}
    return obj if isinstance(obj, dict) else {}


def to_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def short_text(value, limit):
    return to_text(value).replace("\n", " ").strip()[:limit]


def norm_space(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def strip_quotes(value):
    text = str(value or "").strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        text = text[1:-1].strip()
    return text


def qcate(value):
    text = str(value or "").strip().lower().replace("_", "").replace("-", "")
    return "yesno" if text in ("yesno", "yn", "y/n") else text


def is_text_question(rec):
    return qcate(rec["qcate"]) == "text" or str(rec.get("doc_modality") or "").lower() == "text"


def gold_pages(rec):
    keys = ("img_posFacts", "txt_posFacts", "correct_answer_doc_ids")
    return sorted({str(x) for k in keys for x in rec.get(k) or []})
