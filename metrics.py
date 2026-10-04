"""WebQA QA metrics.

QA-Acc: keyword recall against the gold answer, or F1 inside a closed word
domain for yes/no, colour, shape and number questions. QA-FL: token F1 against
the reference answers. QA = QA-Acc * QA-FL.
"""
import re
import string
from collections import Counter

from common import qcate, strip_quotes

try:
    import spacy
    NLP = spacy.load("en_core_web_sm", disable=["ner", "textcat", "parser"])
except (ImportError, OSError):
    NLP = None
try:
    from word2number import w2n
except ImportError:
    w2n = None

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
YES_NO = {"yes", "no"}
COLORS = {
    "orangebrown", "spot", "yellow", "blue", "rainbow", "ivory", "brown", "gray",
    "teal", "bluewhite", "orangepurple", "black", "white", "gold", "redorange",
    "pink", "blonde", "tan", "turquoise", "grey", "beige", "golden", "orange",
    "bronze", "maroon", "purple", "bluere", "red", "rust", "violet", "transparent",
    "yes", "silver", "chrome", "green", "aqua",
}
SHAPES = {
    "globular", "octogon", "ring", "hoop", "octagon", "concave", "flat", "wavy",
    "shamrock", "cross", "cylinder", "cylindrical", "pentagon", "point", "pyramidal",
    "crescent", "rectangular", "hook", "tube", "cone", "bell", "spiral", "ball",
    "convex", "square", "arch", "h", "cuboid", "step", "rectangle", "dot", "oval",
    "circle", "star", "crosse", "crest", "octagonal", "cube", "triangle", "semicircle",
    "domeshape", "obelisk", "corkscrew", "curve", "circular", "xs", "slope", "pyramid",
    "round", "bow", "straight", "triangular", "heart", "fork", "teardrop", "fold",
    "curl", "spherical", "diamond", "keyhole", "conical", "dome", "sphere", "bellshaped",
    "rounded", "hexagon", "flower", "globe", "torus",
}
DOMAINS = {"yesno": YES_NO, "color": COLORS, "shape": SHAPES, "number": "number"}
PUNCT = set(string.punctuation) - {"."}


def to_num(word):
    if word == "point":
        return word
    if w2n:
        try:
            return w2n.word_to_num(word)
        except Exception:
            pass
    return NUMBER_WORDS.get(word, word)


def lemma(word):
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def lemmatize(text):
    if NLP is not None:
        return " ".join(tok.lemma_ for tok in NLP(text))
    return " ".join(lemma(w) for w in text.split())


def normalize(text):
    text = strip_quotes(text).lower()
    if len(text.strip()) == 1:
        return " ".join(str(to_num(w)) for w in text.split())
    out = re.sub(r"\.(?!\d)", "", "".join(ch for ch in text if ch not in PUNCT))
    if len(text.strip().split()) > 1:
        out = re.sub(r"\b(a|an|the)\b", " ", out)
    return lemmatize(" ".join(str(to_num(w)) for w in out.split()))


def tokens(text):
    return normalize(text).split()


def numbers(toks):
    out = []
    for t in toks:
        try:
            out.append(str(int(t)))
        except ValueError:
            pass
    return out


def overlap(pred, answer, domain=None):
    """(F1, recall) of the prediction's tokens against the answer's, optionally inside a word domain."""
    p, a = tokens(pred), tokens(answer)
    if domain == "number":
        p, a = numbers(p), numbers(a)
    elif domain:
        p, a = list(domain.intersection(p)), list(domain.intersection(a))
    same = sum((Counter(a) & Counter(p)).values())
    if same == 0:
        return 0.0, 0.0
    precision, recall = same / len(p), same / len(a)
    return 2 * precision * recall / (precision + recall + 1e-5), recall


def answer_text(refs, keywords=None):
    if keywords:
        return " ".join(strip_quotes(k) for k in keywords) if isinstance(keywords, list) else strip_quotes(keywords)
    return strip_quotes(refs[0]) if refs else ""


def qa_acc(pred, refs, category, keywords=None):
    answer = answer_text(refs, keywords)
    if not answer:
        return 0.0
    category = qcate(category)
    if category in DOMAINS:
        return overlap(pred, answer, DOMAINS[category])[0]
    return overlap(pred, answer)[1]


def qa_fl(pred, refs):
    p = Counter(tokens(pred))
    best = 0.0
    for ref in refs:
        r = Counter(tokens(ref))
        same = sum((r & p).values())
        if same:
            precision, recall = same / sum(p.values()), same / sum(r.values())
            best = max(best, 2 * precision * recall / (precision + recall + 1e-5))
    return best


NEG_RE = re.compile(r"(?i)\b(?:not|no|neither|none|never|isn'?t|aren'?t|doesn'?t|don'?t|didn'?t|without)\b")
HEDGE_RE = re.compile(r"(?i)\b(?:unclear|cannot\s+be\s+determined|can'?t\s+be\s+determined|cannot\s+definitively|"
                      r"not\s+possible\s+to\s+determine|does\s+not\s+(?:indicate|specify|state)\s+whether|"
                      r"insufficient|not\s+enough\s+(?:information|evidence)|unable\s+to\s+determine|"
                      r"no\s+(?:information|evidence)\s+(?:to|about|regarding)|inconclusive|conflicting)\b")
UNANSWERABLE_RE = re.compile(r"(?i)^\W*(?:cannot\s+be\s+answered|unanswerable|n/?a)\W*$")


def polarity(text):
    m = re.match(r"(?i)^\W*(yes|no)\b", text.strip())
    return m.group(1).lower() if m else None


def strict_acc(acc, pred, gold, category):
    category = qcate(category)
    answer = strip_quotes(gold)
    toks = tokens(answer)
    if category == "number":
        in_domain = numbers(toks)
    elif category in DOMAINS:
        in_domain = DOMAINS[category].intersection(toks)
    else:
        return acc
    if in_domain:
        return acc
    if category != "yesno":
        return overlap(pred, answer)[1]
    if UNANSWERABLE_RE.match(answer.strip()):
        return acc
    gold_pol = polarity(answer) or ("no" if NEG_RE.search(answer) else "yes")
    pred = pred.strip()
    if polarity(pred):
        pred_pol = None if HEDGE_RE.search(pred[:40]) else polarity(pred)
    else:
        pred_pol = None if HEDGE_RE.search(pred) else ("no" if NEG_RE.search(pred) else "yes")
    return 0.0 if pred_pol is None else float(pred_pol == gold_pol)
