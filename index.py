import gzip
import heapq
import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np

from common import read_children, read_jsonl_gz

META_FIELDS = ("entities", "keywords", "doc_query")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "did",
    "do", "does", "for", "from", "had", "has", "have", "how", "in", "into",
    "is", "it", "its", "of", "on", "or", "the", "their", "there", "these",
    "this", "those", "to", "was", "were", "what", "when", "where", "which",
    "who", "whose", "with", "would",
}
TOKEN_RE = re.compile(r"[a-z0-9]+")
PRUNE_EPS = 1e-4


def tokenize(text):
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    text = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def dedup(items):
    seen, out = set(), []
    for item in items:
        value = re.sub(r"\s+", " ", str(item or "")).strip()
        if value and value.lower() not in seen:
            seen.add(value.lower())
            out.append(value)
    return out


class Tree:
    def __init__(self, path):
        self.nodes = {r["id"]: r for r in read_jsonl_gz(os.path.join(path, "nodes.jsonl.gz"))}
        self.children = read_children(path)
        self.parents = {c: p for p, kids in self.children.items() for c in kids}
        self.meta = {}

    def resolve(self, leaf_meta):
        def visit(nid):
            kids = self.children[nid]
            for c in kids:
                visit(c)
            if kids:
                self.meta[nid] = {f: dedup([x for c in kids for x in self.meta[c][f]]) for f in META_FIELDS}
            else:
                self.meta[nid] = leaf_meta(nid)

        for nid in self.nodes:
            if nid not in self.parents:
                visit(nid)

    def surface(self, nid, fields):
        m = self.meta[nid]
        return " | ".join(" ; ".join(m[f]) for f in fields if m[f])

    def of_type(self, node_type):
        return [n for n, r in self.nodes.items() if r["node_type"] == node_type]

    def leaves_under(self, nid):
        out, stack = [], [nid]
        while stack:
            n = stack.pop()
            if self.children[n]:
                stack.extend(self.children[n])
            else:
                out.append(n)
        return out


class Index:
    def __init__(self, path, n_keywords, n_doc_query):
        def own_meta(tree, nid):
            m = tree.nodes[nid]["meta"]
            return {"entities": m["entities"], "keywords": m["keywords"][:n_keywords],
                    "doc_query": m["doc_query"][:n_doc_query]}

        self.edges = {e: Tree(os.path.join(path, "edges", e)) for e in sorted(os.listdir(os.path.join(path, "edges")))}
        for t in self.edges.values():
            t.resolve(lambda nid, t=t: own_meta(t, nid))
        cluster_meta = {n: t.meta[n] for t in self.edges.values() for n in t.of_type("edge_cluster_root")}

        self.server = Tree(os.path.join(path, "server"))
        with gzip.open(os.path.join(path, "server", "metadata.json.gz"), "rt") as f:
            self.server.root = json.load(f)["root_id"]
        self.server.resolve(lambda nid: cluster_meta[self.server.nodes[nid]["edge_root_id"]])

        # page -> server leaves whose cluster holds it, for recall
        to_server = {self.server.nodes[n]["edge_root_id"]: n for n in self.server.of_type("server_leaf")}
        self.page_to_server = defaultdict(set)
        for t in self.edges.values():
            root_of = {t.nodes[n]["cluster_id"]: n for n in t.of_type("edge_cluster_root")}
            for n in t.of_type("leaf"):
                leaf = t.nodes[n]
                self.page_to_server[str(leaf["page_id"])].add(to_server[root_of[leaf["cluster_id"]]])


class BM25:
    """BM25 with b = 0, i.e. no document length normalisation."""

    def __init__(self, docs, k1):
        self.k1 = k1
        self.tf = {n: Counter(tokenize(t)) for n, t in docs.items()}
        df = Counter(t for c in self.tf.values() for t in c)
        n = len(self.tf)
        self.idf = {t: math.log(1.0 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def score(self, terms, nid):
        tf, k1 = self.tf[nid], self.k1
        return sum((self.idf[t] * tf[t] * (k1 + 1.0) / (tf[t] + k1) for t in terms if t in tf), 0.0)

    def term_max(self, nodes):
        best = {}
        for nid in nodes:
            for t, tf in self.tf[nid].items():
                v = tf * (self.k1 + 1.0) / (tf + self.k1)
                if v > best.get(t, 0.0):
                    best[t] = v
        return best


class ServerScorer:
    def __init__(self, tree, emb, fields, alpha, k1, qmix):
        self.tree, self.emb, self.qmix = tree, emb, qmix
        self.alpha = alpha if fields else 1.0
        self.bm25 = BM25({n: tree.surface(n, fields) for n in tree.nodes}, k1)
        self.term_max = self.bm25.term_max(tree.of_type("server_leaf"))

    def _query(self, text, vec):
        terms = tokenize(text)
        z = sum(self.bm25.idf.get(t, 0.0) * self.term_max.get(t, 0.0) for t in terms)
        return list(dict.fromkeys(terms)), max(z, 1e-9), vec

    def prepare(self, subquery, sub_vec, question, q_vec):
        ctx = [self._query(subquery, sub_vec)]
        if self.qmix > 0:
            ctx.append(self._query(question, q_vec))
        return ctx

    def _score(self, nid, query):
        terms, z, vec = query
        bm = self.bm25.score(terms, nid) / z if self.alpha < 1.0 else 0.0
        cos = (float(np.dot(vec, self.emb[nid])) + 1.0) / 2.0 if self.alpha > 0.0 else 0.0
        return (1.0 - self.alpha) * bm + self.alpha * cos

    def score(self, nid, ctx):
        s = self._score(nid, ctx[0])
        if len(ctx) > 1:
            s = (1.0 - self.qmix) * s + self.qmix * self._score(nid, ctx[1])
        return s


class EdgeRanker:
    """Ranks the leaves of one edge tree; mode "raw" adds the page title and text, which only the edge holds."""

    def __init__(self, tree, emb, media, fields, mode, k1, alpha):
        docs = {}
        for nid in tree.of_type("leaf"):
            text = tree.surface(nid, fields)
            if mode == "raw":
                page = media[str(tree.nodes[nid]["page_id"])]
                text = " | ".join(x for x in (page["title"], page["caption"], page["raw_data"], text) if x)
            docs[nid] = text
        self.bm25 = BM25(docs, k1)
        self.emb, self.alpha = emb, alpha

    def rank(self, leaves, text, vec):
        terms = list(dict.fromkeys(tokenize(text)))
        bm = {l: self.bm25.score(terms, l) for l in leaves}
        z = max(max(bm.values()), 1e-9)
        scored = []
        for l in leaves:
            cos = (float(np.dot(vec, self.emb[l])) + 1.0) / 2.0
            scored.append((l, (1 - self.alpha) * bm[l] / z + self.alpha * cos))
        return sorted(scored, key=lambda x: -x[1])


def best_first(scorer, ctx, top_k):
    """Descend the server tree by node score; stop once nothing left can beat the k-th best leaf."""
    tree = scorer.tree
    frontier = [(-scorer.score(tree.root, ctx), tree.root)]
    n_scored = 1
    best = []
    while frontier:
        neg, nid = heapq.heappop(frontier)
        if len(best) >= top_k and -neg <= best[0][0] - PRUNE_EPS:
            break
        kids = tree.children[nid]
        if not kids:
            s = scorer.score(nid, ctx)
            if len(best) < top_k:
                heapq.heappush(best, (s, nid))
            elif s > best[0][0]:
                heapq.heapreplace(best, (s, nid))
            continue
        for c in kids:
            s = scorer.score(c, ctx)
            n_scored += 1
            if len(best) < top_k or s > best[0][0] - PRUNE_EPS:
                heapq.heappush(frontier, (-s, c))
    return sorted(((n, s) for s, n in best), key=lambda x: -x[1]), n_scored


def levelwise(scorer, ctx, top_k, max_rounds=32):
    """Beam search: score the children of the beam, keep the top_k, repeat."""
    tree = scorer.tree
    beam, selected, n_scored = [tree.root], [], 0
    for _ in range(max_rounds):
        cands = [c for n in beam for c in tree.children[n]]
        if not cands:
            break
        selected = sorted(((c, scorer.score(c, ctx)) for c in cands), key=lambda x: -x[1])[:top_k]
        n_scored += len(cands)
        beam = [n for n, _ in selected if tree.children[n]]
        if not beam:
            break
    return [(n, s) for n, s in selected if tree.nodes[n]["node_type"] == "server_leaf"], n_scored


def flat(scorer, ctx, top_k):
    leaves = scorer.tree.of_type("server_leaf")
    return sorted(((n, scorer.score(n, ctx)) for n in leaves), key=lambda x: -x[1])[:top_k], len(leaves)


SEARCH = {"best_first": best_first, "levelwise": levelwise, "flat": flat}


def node_embeddings(path):
    d = np.load(path)
    return dict(zip(d["node_ids"].tolist(), d["node_emb"]))
