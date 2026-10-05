"""Backend for the RAGTruth hallucination detector.

Data -> SQLite -> sentence features -> scikit-learn model -> predictions.
A sentence of an LLM response is "hallucinated" if it overlaps a gold span in RAGTruth.
"""
import hashlib
import json
import re
import sqlite3
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DB_PATH = ROOT / "ragtruth.db"
MODEL_PATH = ROOT / "model.joblib"
RAW_URL = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/{}"
FILES = ["response.jsonl", "source_info.jsonl"]

STOPWORDS = set("""a an the and or but if then so of to in on at by for with from as is are was were be been being
it its this that these those he she they them his her their we our you your i me my not no can could will would
should may might must do does did has have had there here which who whom whose what when where why how all any each
also than too very just about into over after before such only own same other more most some both few""".split())

FEATURES = ["n_words", "novel_word_ratio", "novel_words", "novel_numbers", "novel_number_ratio",
            "novel_capitals", "bigram_overlap", "trigram_overlap", "position", "is_last"]


# ---------- data + database ----------

def download_data():
    DATA_DIR.mkdir(exist_ok=True)
    for f in FILES:
        if not (DATA_DIR / f).exists():
            print("downloading", f)
            urllib.request.urlretrieve(RAW_URL.format(f), DATA_DIR / f)


def connect():
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA foreign_keys = ON")
    return con


SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY, task_type TEXT NOT NULL, context TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS responses (
    id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(source_id),
    model TEXT, split TEXT, quality TEXT, response TEXT NOT NULL,
    labels TEXT NOT NULL, hallucinated INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, model_name TEXT, level TEXT,
    accuracy REAL, precision REAL, recall REAL, f1 REAL, roc_auc REAL, threshold REAL, chosen INTEGER);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, context TEXT, response TEXT,
    score REAL, flagged_sentences INTEGER, verdict TEXT);
"""


def build_db():
    """Load both jsonl files into SQLite (idempotent)."""
    download_data()
    con = connect()
    con.executescript(SCHEMA)
    if con.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 0:
        with open(DATA_DIR / "source_info.jsonl") as f:
            src = [json.loads(l) for l in f]
        # the prompt holds the full retrieved context (passages / news / business JSON) for every task
        con.executemany("INSERT INTO sources VALUES (?,?,?)",
                        [(s["source_id"], s["task_type"], s["prompt"]) for s in src])
        with open(DATA_DIR / "response.jsonl") as f:
            rows = [json.loads(l) for l in f]
        con.executemany("INSERT INTO responses VALUES (?,?,?,?,?,?,?,?)",
                        [(r["id"], r["source_id"], r["model"], r["split"], r["quality"], r["response"],
                          json.dumps(r["labels"]), int(bool(r["labels"]))) for r in rows])
        con.commit()
    con.close()


def query(sql, params=()):
    with connect() as con:
        return pd.read_sql_query(sql, con, params=params)


def load_dataset():
    """Joined responses + context, dropping refusals / truncated answers."""
    df = query("""SELECT r.*, s.task_type, s.context FROM responses r
                  JOIN sources s USING(source_id) WHERE r.quality = 'good'""")
    df["labels"] = df["labels"].apply(json.loads)
    # dev split carved from train by source_id, so the 6 answers to one passage never leak across splits
    is_dev = df["source_id"].apply(lambda s: int(hashlib.md5(s.encode()).hexdigest(), 16) % 10 == 0)
    df.loc[(df["split"] == "train") & is_dev, "split"] = "dev"
    return df


# ---------- features ----------

# a dot stays inside a token only before a digit (94.9), so "tall.and" splits into two words
WORD = re.compile(r"[A-Za-z0-9]+(?:(?:['\-]|\.(?=\d))[A-Za-z0-9]+)*")


def norm(w):
    w = w.lower().strip(".'")
    return w[:-1] if len(w) > 3 and w.endswith("s") else w


def split_sentences(text):
    """Return [(start, end, sentence)] with character offsets into text."""
    out, start = [], 0
    for m in re.finditer(r"(?<=[.!?])\s+|\n+", text):
        if text[start:m.start()].strip():
            out.append((start, m.start(), text[start:m.start()]))
        start = m.end()
    if text[start:].strip():
        out.append((start, len(text), text[start:]))
    return out


def context_index(context):
    words = [norm(w) for w in WORD.findall(context)]
    return {"words": set(words), "bigrams": set(zip(words, words[1:])),
            "trigrams": set(zip(words, words[1:], words[2:]))}


def novel_words(sentence, idx):
    """Content words of the sentence that never appear in the context."""
    return [w for w in WORD.findall(sentence)
            if norm(w) not in idx["words"] and norm(w) not in STOPWORDS and len(norm(w)) > 1]


def sentence_features(sentence, idx, i, n):
    raw = WORD.findall(sentence)
    words = [norm(w) for w in raw]
    content = [w for w in words if w not in STOPWORDS and len(w) > 1]
    novel = [w for w in content if w not in idx["words"]]
    numbers = [w for w in words if any(c.isdigit() for c in w)]
    caps = [r for r in raw[1:] if r[0].isupper() and norm(r) not in idx["words"]]
    bi, tri = list(zip(words, words[1:])), list(zip(words, words[1:], words[2:]))
    return [len(words),
            len(novel) / max(len(content), 1),
            len(novel),
            sum(w not in idx["words"] for w in numbers),
            sum(w not in idx["words"] for w in numbers) / max(len(numbers), 1),
            len(caps),
            sum(b in idx["bigrams"] for b in bi) / max(len(bi), 1),
            sum(t in idx["trigrams"] for t in tri) / max(len(tri), 1),
            i / max(n - 1, 1),
            int(i == n - 1)]


def sentence_table(df):
    """One row per response sentence with features + label (1 if it overlaps a gold span)."""
    rows = []
    for r in df.itertuples():
        idx = context_index(r.context)
        sents = split_sentences(r.response)
        for i, (s, e, text) in enumerate(sents):
            label = int(any(l["start"] < e and l["end"] > s for l in r.labels))
            rows.append([r.id, r.split, r.task_type, label, *sentence_features(text, idx, i, len(sents))])
    return pd.DataFrame(rows, columns=["response_id", "split", "task_type", "label", *FEATURES])


# ---------- training ----------

CANDIDATES = {
    "Logistic Regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=0),
    "Gradient Boosting": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0),
}


def metrics(y, prob, thr):
    pred = (prob >= thr).astype(int)
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, prob)}


def best_threshold(y, prob):
    grid = np.arange(0.05, 0.96, 0.01)
    return float(grid[np.argmax([f1_score(y, prob >= t) for t in grid])])


def train(sent):
    """Train every candidate, tune threshold + pick the best on dev, report test metrics, save it + log runs to DB."""
    tr, dev, te = (sent[sent.split == s] for s in ("train", "dev", "test"))
    results, best = [], None
    for name, make in CANDIDATES.items():
        clf = make().fit(tr[FEATURES], tr.label)
        dev_prob = clf.predict_proba(dev[FEATURES])[:, 1]
        thr = best_threshold(dev.label, dev_prob)
        dev_f1 = f1_score(dev.label, dev_prob >= thr)
        prob = clf.predict_proba(te[FEATURES])[:, 1]
        m = metrics(te.label, prob, thr)
        # response level: a response is hallucinated if any of its sentences is
        resp = te.assign(prob=prob).groupby("response_id").agg(y=("label", "max"), p=("prob", "max"))
        results.append({"model": name, "threshold": thr, "dev_f1": dev_f1, **m,
                        **{"resp_" + k: v for k, v in metrics(resp.y, resp.p, thr).items()}})
        # model selection on dev only; the test split is just for reporting
        if best is None or dev_f1 > best[2]:
            best = (name, clf, dev_f1, thr)
    name, clf, _, thr = best
    joblib.dump({"name": name, "model": clf, "threshold": thr}, MODEL_PATH)
    now = datetime.now().isoformat(timespec="seconds")
    with connect() as con:
        for r in results:
            for level, p in (("sentence", ""), ("response", "resp_")):
                con.execute("INSERT INTO runs (created_at, model_name, level, accuracy, precision, recall, f1, roc_auc, threshold, chosen) "
                            "VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (now, r["model"], level, r[p + "accuracy"], r[p + "precision"], r[p + "recall"],
                             r[p + "f1"], r[p + "roc_auc"], r["threshold"], int(r["model"] == name)))
    return pd.DataFrame(results), name


# ---------- inference ----------

_model = None


def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)  # safe: model.joblib is produced locally by train(), never downloaded
    return _model


def detect(context, response, log=True):
    """Score every sentence of the response against the context."""
    bundle = load_model()
    idx = context_index(context)
    sents = split_sentences(response)
    if not sents:
        return {"score": 0.0, "verdict": "No text", "sentences": [], "model": bundle["name"], "threshold": bundle["threshold"]}
    X = pd.DataFrame([sentence_features(t, idx, i, len(sents)) for i, (_, _, t) in enumerate(sents)], columns=FEATURES)
    probs = bundle["model"].predict_proba(X)[:, 1]
    out = [{"start": s, "end": e, "text": t, "prob": float(p), "flagged": bool(p >= bundle["threshold"]),
            "unsupported_words": novel_words(t, idx)} for (s, e, t), p in zip(sents, probs)]
    score = float(probs.max())
    verdict = "Likely hallucinated" if score >= bundle["threshold"] else "Looks grounded"
    if log:
        with connect() as con:
            con.execute("INSERT INTO predictions (created_at, context, response, score, flagged_sentences, verdict) VALUES (?,?,?,?,?,?)",
                        (datetime.now().isoformat(timespec="seconds"), context, response, score,
                         sum(o["flagged"] for o in out), verdict))
    return {"score": score, "verdict": verdict, "sentences": out, "model": bundle["name"], "threshold": bundle["threshold"]}



# ---------- Gemini (generates the answer that the detector then checks) ----------

GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"
PROMPTS = {
    "strict": ("Briefly answer the following question:\n{q}\n"
               "Bear in mind that your response should be strictly based on the following passages:\n{c}"),
    # invites the model to go beyond the context, so the demo shows unsupported claims getting flagged
    "loose": ("Answer the following question in 3-5 sentences:\n{q}\n"
              "Here is some background:\n{c}\n"
              "Also add interesting extra facts from your own knowledge (history, people, numbers) that are not in the background."),
}


def build_context(question, passages):
    """Context the detector checks against: question + passages, like RAGTruth's QA prompts."""
    return f"Question: {question}\nPassages:\n{passages}"


def gemini_answer(question, passages, api_key, model=GEMINI_MODELS[0], style="strict"):
    """Ask Gemini to answer from the passages. Raises RuntimeError with a readable message on failure."""
    body = {"contents": [{"parts": [{"text": PROMPTS[style].format(q=question, c=passages)}]}],
            "generationConfig": {"temperature": 0.7}}
    req = urllib.request.Request(GEMINI_URL.format(model), data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "x-goog-api-key": api_key})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        try:
            msg = json.load(e)["error"]["message"]
        except Exception:
            msg = e.reason
        raise RuntimeError(f"Gemini error {e.code}: {msg}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach Gemini: {e.reason}") from None
    try:
        return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"]).strip()
    except (KeyError, IndexError):
        raise RuntimeError(f"Gemini returned no answer: {json.dumps(data)[:300]}") from None

if __name__ == "__main__":
    # self-check on pure functions (no data needed)
    assert [t for _, _, t in split_sentences("Hi there. Second one! Third")] == ["Hi there.", "Second one!", "Third"]
    idx = context_index("The shop is at 826 B Street, Hayward.")
    assert novel_words("The shop is at 999 Oak Street.", idx) == ["999", "Oak"]
    f = dict(zip(FEATURES, sentence_features("The shop is at 999 Oak Street.", idx, 0, 1)))
    assert f["novel_numbers"] == 1 and f["novel_capitals"] == 1
    assert WORD.findall("tall.and 94.9 o'clock") == ["tall", "and", "94.9", "o'clock"]
    print("backend self-check ok")
