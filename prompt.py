"""Prompt rendering (templates live in prompts.yaml) and parsing of model replies."""
import json
import re
from collections import defaultdict

from common import extract_json, norm_space, qcate, short_text, to_text

NOT_FOUND = "NOT_FOUND"

# an edge reply that says the page does not have the answer
DECLINE_RE = re.compile(
    r"(?i)\b(?:does|do|did)\s+not\s+(?:contain|mention|state|specify|provide|include|"
    r"describe|show|indicate|depict|say|appear|reference)|"
    r"\b(?:is|are|was|were)\s+not\s+(?:mentioned|stated|specified|provided|described|"
    r"shown|depicted|indicated|available|found|included|given)|"
    r"\bno\s+(?:information|evidence|mention|data|details?)\s+(?:about|regarding|on|"
    r"found|available|provided|in\s+the)|"
    r"\b(?:not|no)\s+(?:enough|sufficient)\s+(?:information|evidence|data|detail)")

REFERENT = r"(?:the\s+)?(?:provided\s+|given\s+)?(?:information|evidence|text|document|context|data|excerpt|source|page|content|image)"
HARD_NO_ANSWER_RE = re.compile(
    rf"(?i)("
    rf"{REFERENT}\s+(?:does\s+not|doesn['’]?t|is\s+not|isn['’]?t)\s+(?:state|contain|specify|mention|provide|include|describe|available)|"
    rf"information\s+not\s+found|"
    rf"(?:no|not)\s+(?:enough|sufficient)\s+information|"
    rf"no\s+(?:information|evidence|mention|data)\s+(?:found|available|provided)|"
    rf"cannot\s+(?:determine|find|answer)\s+(?:this|the|from)|"
    rf"unable\s+to\s+(?:determine|find|answer)|"
    rf"no\s+relevant\s+(?:information|evidence)|"
    rf"(?:cannot|can['’]?t)\s+be\s+determined\b"
    rf")"
)
SOFT_NO_ANSWER_RE = re.compile(
    r"(?i)("
    r"\bnot\s+mentioned\b|"
    r"\bno\s+specific\b[^.]{0,60}\b(?:is|are)\s+(?:explicitly\s+|clearly\s+|directly\s+)?"
    r"(?:mentioned|depicted|described|shown|specified|stated|provided|found)\b"
    r")"
)
REFUSAL_RE = re.compile(
    r"(?i)\bno\s+(?:specific|such|direct|explicit)\b|"
    r"\bthere\s+is\s+no\s+(?:evidence|information|mention|indication)\b|"
    r"\bnot\s+(?:mentioned|stated|specified|provided|described|discussed|found|included|given|available)\b|"
    r"\bdoes\s+not\s+(?:mention|state|specify|provide|contain|include|describe|discuss|say|indicate)\b|"
    r"\bno\s+(?:information|evidence|mention|data|details?)\b|"
    r"\bnot\s+enough\b|"
    r"\bcannot\s+be\s+(?:determined|identified|answered)\b")
YESNO_QUESTION_RE = re.compile(
    r"(?i)^\s*(?:is|are|was|were|do|does|did|can|could|has|have|had|will|would|should|shall|may|might)\b")
BARE_NON_ANSWER_RE = re.compile(r"(?i)^\s*(?:none|null|n/?a|unknown|not\s+applicable)\s*\.?\s*$")
BARE_BOOL_RE = re.compile(r"(?i)^\s*(yes|no|true|false)\s*\.?\s*$")
STRUCTURED_RE = re.compile(r"^\s*[\{\[]")
JSON_PAIR_RE = re.compile(r'"([^"]+)"\s*:\s*"([^"]*)"')
# most explanation questions have a sentence as gold answer, other text questions a short span
EXPLAIN_QUESTION_RE = re.compile(
    r"(?i)^\s*(?:why|how\s+(?:does|do|did|is|are|was|were|can|could|would))\b|"
    r"\bexplain\b|\bdescribe\b|\bin\s+what\s+way\b")


def parse_subqueries(raw, question, limit):
    out, seen = [], set()
    for sq in extract_json(raw).get("subquestions") or []:
        if not isinstance(sq, dict):
            continue
        text = norm_space(sq.get("subquestion"))
        if text and text.lower() not in seen:
            seen.add(text.lower())
            out.append({"text": text, "entity": str(sq.get("entity") or "").strip()})
    return out[:limit] or [{"text": question, "entity": ""}]


def edge_prompt(P, cfg, question, subquery, page, modality, image):
    if modality == "image":
        label, body = P["image_label"], short_text(page["caption"], cfg["caption_chars"])
    else:
        label, body = P["text_label"], short_text(page["raw_data"], cfg["text_chars"])
    return P["template"].format(
        main_question=P["main_question"].format(question=question) if question else "",
        main_question_hint=P["main_question_hint"] if question else "",
        subquery=subquery, title=short_text(page["title"], cfg["title_chars"]), body_label=label, body=body,
        image_note=P["image_note"] if image else "", image_source=P["image_source"] if image else "")


def is_decline(answer):
    return not answer or answer.upper().startswith(NOT_FOUND) or bool(DECLINE_RE.search(answer[:140]))


def parse_edge(raw):
    obj = extract_json(raw)
    answer = to_text(obj.get("answer", raw)).strip()
    rel = str(obj.get("relevant", "")).strip().lower()
    relevant = rel.startswith("y") if rel else None
    return {"answer": answer, "evidence": to_text(obj.get("evidence", "")).strip(), "relevant": relevant,
            "declined": relevant is False or is_decline(answer)}


def normalize_answer(answer, subquery):
    """A JSON answer becomes "key: value; ..."; True/False to a yes/no subquestion becomes Yes/No."""
    text = answer.strip()
    if STRUCTURED_RE.match(text):
        parts = []

        def walk(value, key=""):
            if isinstance(value, dict):
                for k, v in value.items():
                    walk(v, str(k).replace("_", " "))
            elif isinstance(value, list):
                for v in value:
                    walk(v, key)
            elif value is not None and str(value).strip():
                parts.append(f"{key}: {value}" if key else str(value))

        try:
            walk(json.loads(text))
        except ValueError:
            parts = [f"{k.replace('_', ' ')}: {v}" for k, v in JSON_PAIR_RE.findall(text) if v.strip()]
        return "; ".join(parts)
    m = BARE_BOOL_RE.match(text)
    if m and YESNO_QUESTION_RE.match(subquery):
        return {"true": "Yes", "false": "No"}.get(m.group(1).lower(), m.group(1).capitalize())
    return text


def is_unusable(answer, subquery):
    text = answer.strip()
    if not text or text.upper().startswith(NOT_FOUND) or HARD_NO_ANSWER_RE.search(text):
        return True
    if len(text) <= 140 and SOFT_NO_ANSWER_RE.search(text):
        return True
    if BARE_NON_ANSWER_RE.match(text) or STRUCTURED_RE.match(text):
        return True
    if BARE_BOOL_RE.match(text) and not YESNO_QUESTION_RE.match(subquery):
        return True
    return bool(REFUSAL_RE.search(text[:60]))


def select_evidence(answers, max_server_rank, per_sub_cap, max_items):
    """Usable edge answers, grouped by subquery; identical answers are merged and counted."""
    groups = defaultdict(list)
    for a in sorted(answers, key=lambda a: (a["server_rank"] + a["leaf_rank"], a["server_rank"])):
        if a["declined"] or a["server_rank"] > max_server_rank:
            continue
        ans = normalize_answer(a["answer"], a["subquery"])
        if is_unusable(ans, a["subquery"]):
            continue
        group = groups[a["subquery_index"]]
        key = re.sub(r"\W+", " ", ans.lower()).strip()
        same = next((g for g in group if g["key"] == key), None)
        if same:
            same["count"] += 1
        elif len(group) < per_sub_cap:
            group.append({**a, "answer": ans, "count": 1, "key": key})
    out = [a for s in sorted(groups) for a in groups[s]]
    out = sorted(out, key=lambda a: (a["server_rank"] + a["leaf_rank"], a["subquery_index"]))[:max_items]
    return sorted(out, key=lambda a: (a["subquery_index"], a["server_rank"] + a["leaf_rank"]))


def rewrite_prompt(P, question, subqueries, evidence, missing):
    answered = []
    for s in subqueries:
        answers = [e["answer"] for e in evidence if e["subquery_index"] == s["index"]]
        if answers:
            answered.append(P["answered"].format(subquery=s["text"], answers="; ".join(answers)))
    return P["template"].format(question=question, answered="\n".join(answered) or P["none"],
                                unanswered="\n".join(P["unanswered"].format(subquery=s["text"]) for s in missing))


def parse_rewrite(raw, missing, offset):
    new = [s for s in extract_json(raw).get("subquestions") or []
           if isinstance(s, dict) and str(s.get("subquestion", "")).strip()]
    out = []
    for j, s in enumerate(missing):
        if j < len(new):
            text, entity = str(new[j]["subquestion"]).strip(), str(new[j].get("entity", "")).strip()
        else:
            text, entity = s["text"], ""
        out.append({"index": offset + s["index"], "text": re.sub(r"\s+", " ", text), "entity": entity})
    return out


def evidence_block(P, cfg, subqueries, evidence):
    by_sub = defaultdict(list)
    for a in evidence:
        by_sub[a["subquery_index"]].append(a)
    lines = []
    for n, s in enumerate(subqueries, 1):
        lines.append(P["subquestion"].format(n=n, text=s["text"]))
        if not by_sub[s["index"]]:
            lines.append(P["none"])
        for a in by_sub[s["index"]]:
            count = P["count"].format(count=a["count"]) if a["count"] > 1 else ""
            lines.append(P["answer"].format(answer=short_text(a["answer"], cfg["answer_chars"]), count=count))
            if a["evidence"]:
                lines.append(P["support"].format(evidence=short_text(a["evidence"], cfg["evidence_chars"])))
    return "\n".join(lines)


def final_prompt(P, cfg, rec, subqueries, evidence):
    question = rec["question"]
    block = evidence_block(P["evidence"], cfg, subqueries, evidence)
    if rec["modality"] == "text":
        form = P["explain_form"] if EXPLAIN_QUESTION_RE.search(question) else P["short_form"]
        return P["text"].format(form=form, examples=P["text_examples"], question=question, note=P["note"],
                                evidence=block)
    guidance = P["guidance"].get(qcate(rec["qcate"]), "")
    return P["image"].format(examples=P["image_examples"], guidance=f"\n{guidance}\n" if guidance else "",
                             question=question, note=P["note"], evidence=block)
