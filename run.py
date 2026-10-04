
import argparse
import csv
import json
import os
import subprocess
import time
from collections import Counter, defaultdict
from contextlib import contextmanager

import metrics
import prompts as PR
from common import (digest, extract_json, gold_pages, image_files, is_text_question, load_config, load_json,
                    load_media, save_json, strip_quotes)
from embedder import Embedder
from index import META_FIELDS, SEARCH, EdgeRanker, Index, ServerScorer, node_embeddings

HERE = os.path.dirname(os.path.abspath(__file__))
OFFSET = 50  # round-2 rewrite of subquery i is stored as OFFSET + i
RETRIEVAL_STAGES = ("r1_query_embedding", "r1_server_tree_search", "r1_edge_leaf_ranking",
                    "r2_query_embedding", "r2_server_tree_search", "r2_edge_leaf_ranking")
OTHER_STAGES = ("subquery_generation", "r1_edge_answering", "r2_subquery_rewrite", "r2_edge_answering",
                "final_answer", "cpu_glue")


def load_records(path, subset=None):
    rows = load_json(path)["results"]
    keep = set(load_json(subset)["record_indices"]) if subset else range(len(rows))
    return [{"record_index": i, "qid": r["qid"], "question": r["question"], "qcate": r["qcate"],
             "modality": "text" if is_text_question(r) else "image", "gold_answer": r["gold_answer"],
             "reference_answers": r["reference_answers"], "keywords_answer": r.get("keywords_answer") or "",
             "gold_pages": gold_pages(r)}
            for i, r in enumerate(rows) if i in keep]


class Pipeline:
    def __init__(self, cfg, run_dir):
        self.cfg, self.P, self.run_dir = cfg, cfg["prompts"], run_dir
        self.times = defaultdict(Counter)  # record_index -> stage -> seconds
        self.setup, self.llm_stages = {}, {}
        self.glue = 0.0

        self.t_start = t0 = time.time()
        r, e = cfg["retrieval"], cfg["edge"]
        self.media = load_media(cfg["data"]["media"])
        self.images = image_files(cfg["data"]["images"])
        self.index = Index(cfg["data"]["index"], r["n_keywords"], r["n_doc_query"])
        emb = node_embeddings(cfg["data"]["node_embeddings"])
        fields = [f for f in META_FIELDS if f in r["fields"]]
        server_fields = [f for f in fields if r["server_fields"] is None or f in r["server_fields"]]
        server = self.index.server
        self.scorer = ServerScorer(server, {n: emb[f"server|{n}"] for n in server.nodes}, server_fields,
                                   r["alpha"], r["k1"], r["qmix"])
        edge_alpha = r["alpha"] if e["alpha"] is None else e["alpha"]
        self.rankers = {name: EdgeRanker(t, {n: emb[f"edge:{name}|{n}"] for n in t.of_type("leaf")}, self.media,
                                         fields, e["rank"], r["k1"], edge_alpha)
                        for name, t in self.index.edges.items()}
        self.search = SEARCH[r["method"]]
        self.setup["index_and_bm25_load"] = time.time() - t0

    @contextmanager
    def cpu(self):
        t = time.perf_counter()
        yield
        self.glue += time.perf_counter() - t

    def charge(self, stage, seconds, weights):
        total = sum(weights.values())
        for r, w in weights.items():
            self.times[r][stage] += seconds * w / total

    def generate(self, tag, model, system, jobs, max_tokens, stage, slots):
        """jobs {key: (prompt, image)} -> {key: reply}; slots [(record_index, key)] share out the time."""
        if not jobs:
            return {}
        path = {x: os.path.join(self.run_dir, "gen", f"{tag}.{x}") for x in ("jobs.jsonl", "outs.jsonl", "meta.json", "log")}
        with open(path["jobs.jsonl"], "w", encoding="utf-8") as f:
            for k, (prompt, image) in jobs.items():
                f.write(json.dumps({"k": k, "prompt": prompt, "image": image}, ensure_ascii=False) + "\n")
        print(f"[{tag}] {len(jobs)} prompts", flush=True)
        cmd = [self.cfg["python"]["vllm"], os.path.join(HERE, "llm.py"), "--config", os.path.join(self.run_dir, "config.json"),
               "--model", model, "--system", system, "--jobs", path["jobs.jsonl"], "--out", path["outs.jsonl"],
               "--meta", path["meta.json"], "--max-tokens", str(max_tokens)]
        t0 = time.time()
        with open(path["log"], "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        wall = time.time() - t0
        with open(path["outs.jsonl"], encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        meta = load_json(path["meta.json"])
        meta["setup_seconds"] = wall - meta["generate_seconds"]
        self.setup[f"{stage}_model_setup"] = meta["setup_seconds"]
        self.llm_stages[stage] = meta
        print(f"[{tag}] generate {meta['generate_seconds']:.1f}s, setup {meta['setup_seconds']:.1f}s", flush=True)

        tokens = {row["k"]: row["tokens"] for row in rows}
        shared = Counter(k for _, k in slots)
        weights = Counter()
        for r, k in slots:
            weights[r] += tokens[k] / shared[k]
        self.charge(stage, meta["generate_seconds"], weights)
        return {row["k"]: row["v"] for row in rows}

    def embed(self, texts, rnd, weights):
        t0 = time.time()
        model = Embedder(self.cfg["models"]["embedding"], **self.cfg["embedding"])
        model.encode(["warm up the embedding kernels"] * self.cfg["embedding"]["batch_size"])
        self.setup[f"embedder_load_{rnd}"] = time.time() - t0
        uniq = list(dict.fromkeys(texts))
        vecs, seconds = model.timed_encode(uniq)
        model.close()
        self.charge(f"{rnd}_query_embedding", seconds, weights)
        return dict(zip(uniq, vecs))

    def retrieve(self, rec, subqueries, sub_vec, q_vec, rnd):
        r, top_k, n_leaves = rec["record_index"], self.cfg["retrieval"]["top_k"], self.cfg["edge"]["candidates"]
        out = []
        for s in subqueries:
            vec = sub_vec[(r, s["index"])]
            t = time.perf_counter()
            ctx = self.scorer.prepare(s["text"], vec, rec["question"], q_vec[r])
            ranked, n_scored = self.search(self.scorer, ctx, top_k)
            self.times[r][f"{rnd}_server_tree_search"] += time.perf_counter() - t
            nodes = []
            for rank, (nid, score) in enumerate(ranked, 1):
                edge, root = self.index.server.nodes[nid]["edge_id"], self.index.server.nodes[nid]["edge_root_id"]
                tree = self.index.edges[edge]
                t = time.perf_counter()
                leaves = self.rankers[edge].rank(tree.leaves_under(root), s["text"], vec)
                self.times[r][f"{rnd}_edge_leaf_ranking"] += time.perf_counter() - t
                nodes.append({"server_rank": rank, "server_node_id": nid, "edge_id": edge, "edge_root_id": root,
                              "score": score,
                              "leaves": [{"leaf_rank": i, "leaf_node_id": l, "page_id": str(tree.nodes[l]["page_id"]),
                                          "modality": tree.nodes[l]["page_modality"], "score": sc}
                                         for i, (l, sc) in enumerate(leaves[:n_leaves], 1)]})
            out.append(dict(s, nodes=nodes, nodes_scored=n_scored))
        return out

    def edge_answers(self, records, retrieved, replies, tag, stage):
        """Edge LLM answers for the top leaves of every routed cluster; `replies` is shared across rounds."""
        e = self.cfg["edge"]
        with self.cpu():
            jobs, slots = {}, []
            for rec in records:
                for s in retrieved[rec["record_index"]]:
                    for node in s["nodes"]:
                        for leaf in node["leaves"][:e["leaf_k"]]:
                            image = self.images.get(leaf["page_id"]) if leaf["modality"] == "image" else None
                            prompt = PR.edge_prompt(self.P["edge"], e, rec["question"] if e["show_question"] else None,
                                                    s["text"], self.media[leaf["page_id"]], leaf["modality"], image)
                            k = digest(prompt, image or "")
                            jobs.setdefault(k, (prompt, image))
                            slots.append((rec["record_index"], k, s, node, leaf))
            todo = {k: j for k, j in jobs.items() if k not in replies}
        replies.update(self.generate(tag, "edge", self.P["edge"]["system"], todo, self.cfg["max_tokens"]["edge"], stage,
                                     [(r, k) for r, k, *_ in slots if k in todo]))
        with self.cpu():
            answers = defaultdict(list)
            for r, k, s, node, leaf in slots:
                answers[r].append({"subquery_index": s["index"], "subquery": s["text"],
                                   "server_rank": node["server_rank"], "leaf_rank": leaf["leaf_rank"],
                                   "page_id": leaf["page_id"], "page_modality": leaf["modality"],
                                   "edge_id": node["edge_id"], **PR.parse_edge(replies[k])})
            return [{"record_index": rec["record_index"], "question": rec["question"],
                     "subqueries": [{"index": s["index"], "text": s["text"]} for s in retrieved[rec["record_index"]]],
                     "edge_answers": answers[rec["record_index"]]} for rec in records]

    def run(self, records):
        cfg, P = self.cfg, self.P
        ev_cfg = cfg["evidence"]
        by_id = {r["record_index"]: r for r in records}

        if cfg["pipeline"]["subquery"]:
            jobs = {str(r["record_index"]): (P["subquery"]["template"].format(question=r["question"]), None)
                    for r in records}
            raw = self.generate("1_subquery", "server", P["subquery"]["system"], jobs, cfg["max_tokens"]["subquery"],
                                "subquery_generation", [(r["record_index"], str(r["record_index"])) for r in records])
            with self.cpu():
                for r in records:
                    r["raw_subqueries"] = raw[str(r["record_index"])]
                    subs = PR.parse_subqueries(r["raw_subqueries"], r["question"], cfg["pipeline"]["max_subqueries"])
                    r["subqueries"] = [dict(s, index=j) for j, s in enumerate(subs)]
        else:
            for r in records:
                r["subqueries"] = [{"text": r["question"], "entity": "", "index": 0}]
        save_json(records, os.path.join(self.run_dir, "subqueries.json"))

        # round 1
        texts = [r["question"] for r in records] + [s["text"] for r in records for s in r["subqueries"]]
        vecs = self.embed(texts, "r1", {r["record_index"]: 1 + len(r["subqueries"]) for r in records})
        q_vec = {r["record_index"]: vecs[r["question"]] for r in records}
        sub_vec = {(r["record_index"], s["index"]): vecs[s["text"]] for r in records for s in r["subqueries"]}
        ret1 = {r["record_index"]: self.retrieve(r, r["subqueries"], sub_vec, q_vec, "r1") for r in records}
        edge_replies = {}
        rows1 = self.edge_answers(records, ret1, edge_replies, "2_edge_r1", "r1_edge_answering")

        # round 2: rewrite the subqueries without a usable answer, using the answers that did come back
        ret2, rows2 = {}, {}
        if cfg["pipeline"]["round2"]:
            with self.cpu():
                jobs, missing = {}, {}
                for row in rows1:
                    ev = PR.select_evidence(row["edge_answers"], **ev_cfg)
                    have = {a["subquery_index"] for a in ev}
                    todo = [s for s in row["subqueries"] if s["index"] not in have]
                    if todo:
                        prompt = PR.rewrite_prompt(P["rewrite"], row["question"], row["subqueries"], ev, todo)
                        jobs[str(row["record_index"])] = (prompt, None)
                        missing[row["record_index"]] = todo
            if missing:
                rewrites = self.generate("3_rewrite", "server", P["rewrite"]["system"], jobs, cfg["max_tokens"]["rewrite"],
                                         "r2_subquery_rewrite", [(r, str(r)) for r in missing])
                with self.cpu():
                    recs2 = [dict(by_id[r], subqueries=PR.parse_rewrite(rewrites[str(r)], subs, OFFSET))
                             for r, subs in missing.items()]
                vecs = self.embed([s["text"] for r in recs2 for s in r["subqueries"]], "r2",
                                  {r["record_index"]: len(r["subqueries"]) for r in recs2})
                sub_vec.update({(r["record_index"], s["index"]): vecs[s["text"]] for r in recs2 for s in r["subqueries"]})
                ret2 = {r["record_index"]: self.retrieve(r, r["subqueries"], sub_vec, q_vec, "r2") for r in recs2}
                rows2 = {row["record_index"]: row
                         for row in self.edge_answers(recs2, ret2, edge_replies, "4_edge_r2", "r2_edge_answering")}

        # final answer
        with self.cpu():
            merged = []
            for row in rows1:
                new = rows2.get(row["record_index"])
                if new:
                    replaced = {s["index"] - OFFSET for s in new["subqueries"]}
                    row = dict(row, subqueries=sorted([s for s in row["subqueries"] if s["index"] not in replaced]
                                                      + new["subqueries"], key=lambda s: s["index"] % OFFSET),
                               edge_answers=[a for a in row["edge_answers"] if a["subquery_index"] not in replaced]
                               + new["edge_answers"])
                merged.append(row)
            evidence = [PR.select_evidence(row["edge_answers"], **ev_cfg) for row in merged]
            prompts = [PR.final_prompt(P["final"], cfg["final"], by_id[row["record_index"]], row["subqueries"], ev)
                       for row, ev in zip(merged, evidence)]
            keys = [digest(p) for p in prompts]
            jobs = {k: (p, None) for k, p in zip(keys, prompts)}
        finals = self.generate("5_final", "server", P["final"]["system"], jobs, cfg["max_tokens"]["final"],
                               "final_answer", [(row["record_index"], k) for row, k in zip(merged, keys)])

        with self.cpu():
            results = []
            for row, ev, k in zip(merged, evidence, keys):
                rec = by_id[row["record_index"]]
                parsed = extract_json(finals[k])
                pred = strip_quotes(parsed.get("answer", finals[k]))
                refs = [x for x in map(strip_quotes, rec["reference_answers"] or [rec["gold_answer"]]) if x]
                acc = metrics.qa_acc(pred, refs, rec["qcate"], rec["keywords_answer"])
                fl = metrics.qa_fl(pred, refs)
                strict = metrics.strict_acc(acc, pred, rec["gold_answer"], rec["qcate"])
                gold = set(rec["gold_pages"])
                results.append({
                    "record_index": rec["record_index"], "qid": rec["qid"], "question": rec["question"],
                    "qcate": rec["qcate"], "modality": rec["modality"], "gold_answer": rec["gold_answer"],
                    "predicted_answer": pred, "reason": parsed.get("reason", ""),
                    "qa_acc": acc, "qa_fl": fl, "qa": acc * fl, "qa_acc_strict": strict, "qa_strict": strict * fl,
                    "n_evidence": len(ev), "n_gold_evidence": sum(a["page_id"] in gold for a in ev),
                    "gold_pages_in_evidence": len({a["page_id"] for a in ev} & gold),
                    "evidence": [{k: a[k] for k in ("subquery_index", "server_rank", "leaf_rank", "page_id", "answer",
                                                     "evidence", "count")} for a in ev],
                })
        self.charge("cpu_glue", self.glue, {r["record_index"]: 1 for r in records})
        wall = time.time() - self.t_start

        n_rewritten = sum(len(row["subqueries"]) for row in rows2.values())
        n_recovered = sum(len({a["subquery_index"] for a in PR.select_evidence(row["edge_answers"], **ev_cfg)})
                          for row in rows2.values())
        for res in results:
            r = res["record_index"]
            replaced = {s["index"] - OFFSET for s in ret2.get(r, [])}
            final_subs = [s for s in ret1[r] if s["index"] not in replaced] + ret2.get(r, [])
            res["round2"] = int(r in ret2)
            res["recall_r1"] = self.recall(by_id[r]["gold_pages"], ret1[r])
            res["recall_final"] = self.recall(by_id[r]["gold_pages"], final_subs)
            t = self.times[r]
            res["time"] = {"retrieval_seconds": sum(t[k] for k in RETRIEVAL_STAGES),
                           "other_seconds": sum(t[k] for k in OTHER_STAGES),
                           "breakdown": {k: t[k] for k in RETRIEVAL_STAGES + OTHER_STAGES}}
            res["time"]["total_seconds"] = res["time"]["retrieval_seconds"] + res["time"]["other_seconds"]

        meta = self.summarize(results, wall)
        meta["round2"] = {"questions": len(rows2), "rewritten": n_rewritten, "recovered": n_recovered}
        meta["config"] = {k: v for k, v in cfg.items() if k != "prompts"}
        save_json({"metadata": meta, "results": results}, os.path.join(self.run_dir, "results.json"))
        save_json({"round1": rows1, "round2": list(rows2.values())}, os.path.join(self.run_dir, "edge_answers.json"), None)
        write_csv(results, os.path.join(self.run_dir, "results.csv"))
        tm = meta["time"]["per_question_mean_seconds"]
        by_mod = " ".join(f"{k}={v['qa']:.4f}" for k, v in sorted(meta["by_modality"].items()))
        print(f"QA={meta['qa']:.4f} acc={meta['qa_acc']:.4f} fl={meta['qa_fl']:.4f} {by_mod} | server page recall "
              f"r1={meta['recall']['r1']['server_page_recall']:.4f} final={meta['recall']['final']['server_page_recall']:.4f}"
              f" | per question: retrieval {tm['retrieval']:.4f}s, other {tm['other']:.4f}s | wall {wall:.0f}s",
              flush=True)

    def recall(self, gold, subqueries):
        leaf_k = self.cfg["edge"]["leaf_k"]
        routed, candidates, answered, hits = set(), set(), set(), 0
        for s in subqueries:
            ids = {n["server_node_id"] for n in s["nodes"]}
            routed |= ids
            hits += any(self.index.page_to_server[p] & ids for p in gold)
            for n in s["nodes"]:
                for leaf in n["leaves"]:
                    candidates.add(leaf["page_id"])
                    if leaf["leaf_rank"] <= leaf_k:
                        answered.add(leaf["page_id"])
        covered = {p for p in gold if self.index.page_to_server[p] & routed}
        n = len(gold)
        return {"n_gold_pages": n, "n_subqueries": len(subqueries), "subquery_hits": hits,
                "nodes_scored": sum(s["nodes_scored"] for s in subqueries),
                "server_gold_pages": len(covered), "edge_candidate_gold_pages": len(candidates & set(gold)),
                "answered_gold_pages": len(answered & set(gold)),
                "server_page_recall": len(covered) / n, "edge_candidate_page_recall": len(candidates & set(gold)) / n,
                "answered_page_recall": len(answered & set(gold)) / n,
                "server_all_gold": int(len(covered) == n), "answered_all_gold": int(set(gold) <= answered)}

    def summarize(self, results, wall):
        n = len(results)

        def mean(key, rows=results):
            return sum(x[key] for x in rows) / len(rows)

        def group(rows):
            return {"n": len(rows), "qa_acc": mean("qa_acc", rows), "qa_fl": mean("qa_fl", rows), "qa": mean("qa", rows)}

        by_mod, by_cat = defaultdict(list), defaultdict(list)
        for x in results:
            by_mod[x["modality"]].append(x)
            by_cat[x["qcate"]].append(x)
        md = {"questions": n, "qa_acc": mean("qa_acc"), "qa_fl": mean("qa_fl"), "qa": mean("qa"),
              "qa_acc_strict": mean("qa_acc_strict"), "qa_strict": mean("qa_strict"),
              "text_qa": group(by_mod["text"])["qa"] if by_mod["text"] else None,
              "image_qa": group(by_mod["image"])["qa"] if by_mod["image"] else None,
              "by_modality": {k: group(v) for k, v in by_mod.items()},
              "by_qcate": {k: group(v) for k, v in by_cat.items()}}

        n_ev = sum(x["n_evidence"] for x in results)
        md["evidence_per_question"] = n_ev / n
        md["evidence_gold_precision"] = sum(x["n_gold_evidence"] for x in results) / max(n_ev, 1)
        md["q_with_gold_evidence"] = sum(x["n_gold_evidence"] > 0 for x in results) / n
        md["q_without_evidence"] = sum(x["n_evidence"] == 0 for x in results) / n

        md["recall"] = {}
        for tag in ("r1", "final"):
            rc = [x[f"recall_{tag}"] for x in results]
            pages = sum(c["n_gold_pages"] for c in rc)
            subs = sum(c["n_subqueries"] for c in rc)
            md["recall"][tag] = {
                "server_page_recall": sum(c["server_gold_pages"] for c in rc) / pages,
                "edge_candidate_page_recall": sum(c["edge_candidate_gold_pages"] for c in rc) / pages,
                "answered_page_recall": sum(c["answered_gold_pages"] for c in rc) / pages,
                "server_subquery_recall": sum(c["subquery_hits"] for c in rc) / subs,
                "server_question_all_recall": sum(c["server_all_gold"] for c in rc) / n,
                "answered_question_all_recall": sum(c["answered_all_gold"] for c in rc) / n,
                "nodes_scored_per_subquery": sum(c["nodes_scored"] for c in rc) / subs,
                "subqueries_per_question": subs / n,
            }
        md["recall"]["final"]["evidence_page_recall"] = sum(x["gold_pages_in_evidence"] for x in results) / sum(
            x["recall_final"]["n_gold_pages"] for x in results)

        totals = {k: sum(self.times[x["record_index"]][k] for x in results) for k in RETRIEVAL_STAGES + OTHER_STAGES}
        md["time"] = {
            "per_question_mean_seconds": {"retrieval": sum(totals[k] for k in RETRIEVAL_STAGES) / n,
                                          "other": sum(totals[k] for k in OTHER_STAGES) / n,
                                          "total": sum(totals.values()) / n,
                                          "breakdown": {k: v / n for k, v in totals.items()}},
            "stage_total_seconds": totals, "one_time_setup_seconds": self.setup, "llm_stages": self.llm_stages,
            "run_wall_clock_seconds": wall,
        }
        return md


def write_csv(results, path):
    rows = []
    for x in results:
        row = {k: x[k] for k in ("record_index", "qid", "qcate", "modality", "question", "gold_answer",
                                 "predicted_answer", "qa_acc", "qa_fl", "qa", "qa_acc_strict", "qa_strict",
                                 "n_evidence", "n_gold_evidence", "round2")}
        row["n_gold_pages"] = x["recall_final"]["n_gold_pages"]
        for tag in ("r1", "final"):
            for k in ("server_page_recall", "edge_candidate_page_recall", "answered_page_recall"):
                row[f"{tag}_{k}"] = x[f"recall_{tag}"][k]
        row["final_n_subqueries"] = x["recall_final"]["n_subqueries"]
        row["final_nodes_scored"] = x["recall_final"]["nodes_scored"]
        row["time_retrieval_s"] = x["time"]["retrieval_seconds"]
        row["time_other_s"] = x["time"]["other_seconds"]
        row["time_total_s"] = x["time"]["total_seconds"]
        row.update({f"time_{k}_s": v for k, v in x["time"]["breakdown"].items()})
        rows.append(row)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--subset", help="json with record_indices, e.g. dev100.json")
    ap.add_argument("--name", help="run directory under data.output (default: subset name or 'all')")
    ap.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE", help="override config entries")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    name = args.name or (os.path.splitext(os.path.basename(args.subset))[0] if args.subset else "all")
    run_dir = os.path.join(cfg["data"]["output"], name)
    os.makedirs(os.path.join(run_dir, "gen"), exist_ok=True)
    save_json(cfg, os.path.join(run_dir, "config.json"))
    Pipeline(cfg, run_dir).run(load_records(cfg["data"]["records"], args.subset))


if __name__ == "__main__":
    main()
