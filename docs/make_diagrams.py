"""Run: python docs/make_diagrams.py. System-design diagrams for the report (architecture, use case, module/class, activity, ER)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse

OUT = str(__import__("pathlib").Path(__file__).parent / "figures") + "/"
NAVY, RED, TEAL, BLUE, GRAY, LIGHT, INK = "#14213D", "#D7263D", "#1B998B", "#2E86AB", "#8A96A8", "#F2F4F7", "#14213D"
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 10})


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, fc=LIGHT, ec=NAVY, color=INK, size=10, bold=False, r=0.08, align="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=1.2))
    tx = x + w / 2 if align == "center" else x + 0.12
    ax.text(tx, y + h / 2, text, ha=align, va="center", color=color, fontsize=size, fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, p, q, label="", color=NAVY, style="-|>", ls="-", at=None):
    ax.annotate("", q, p, arrowprops=dict(arrowstyle=style, color=color, lw=1.3, ls=ls, shrinkA=0, shrinkB=0))
    if label:  # default: just right of the arrow's midpoint
        x, y = at or ((p[0] + q[0]) / 2 + 0.1, (p[1] + q[1]) / 2)
        ax.text(x, y, label, ha="left", va="center", fontsize=8.5, color=GRAY,
                bbox=dict(fc="white", ec="none", pad=1))


def save(fig, name):
    fig.savefig(OUT + name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---- 1. architecture ----
fig, ax = canvas(12.4, 6.6)
NL = "\n"
layers = [("Presentation layer", 5.0, [("Jupyter notebook" + NL + "(Run-All walkthrough)", 0),
                                       ("Streamlit app  (app.py)" + NL + "Detector · Dataset · Model · History", 1)]),
          ("Application / ML layer  (backend.py)", 2.9, [("Data loader" + NL + "download, build_db", 0), ("Feature engine" + NL + "sentences, 10 features", 1),
                                                         ("Trainer" + NL + "LR · RF · Grad. Boosting", 2), ("Inference" + NL + "detect(), risk score", 3),
                                                         ("Gemini client" + NL + "retry + fallback", 4)]),
          ("Data layer", 0.8, [("model.joblib" + NL + "(best model + threshold)", 0),
                               ("SQLite ragtruth.db" + NL + "sources · responses · runs · predictions", 1), ("RAGTruth jsonl" + NL + "(data/)", 2)])]
for title, y, items in layers:
    ax.add_patch(FancyBboxPatch((0.2, y - 0.25), 9.4, 1.75, boxstyle="round,pad=0,rounding_size=0.1", fc="white", ec=GRAY, lw=1, ls="--"))
    ax.text(0.4, y + 1.3, title, fontsize=11, fontweight="bold", color=NAVY)
    n = len(items); w = (9.0 - 0.2 * (n - 1)) / n
    for t, i in items:
        box(ax, 0.4 + i * (w + 0.2), y, w, 1.0, t, fc=LIGHT if "Gemini" not in t else "#FBE9EB", size=8.5)
ax.text(11.1, 6.3, "External services", fontsize=10, fontweight="bold", color=GRAY, ha="center")
box(ax, 10.1, 2.9, 2.0, 1.0, "Google" + NL + "Gemini API", fc="#FBE9EB", ec=RED, size=9.5)
box(ax, 10.1, 0.8, 2.0, 1.0, "RAGTruth" + NL + "(GitHub, MIT)", fc="white", ec=GRAY, size=9.5)
arrow(ax, (4.8, 5.0), (4.8, 3.9), "function calls")
arrow(ax, (4.8, 2.9), (4.8, 1.8), "SQL (sqlite3) / joblib")
arrow(ax, (9.4, 3.4), (10.1, 3.4), color=RED)
ax.text(9.75, 3.1, "HTTPS", fontsize=8.5, color=RED, ha="center")
arrow(ax, (10.1, 1.3), (9.4, 1.3), color=GRAY, ls="--")
ax.text(9.75, 1.0, "urllib", fontsize=8.5, color=GRAY, ha="center")
save(fig, "diagram_architecture.png")

# ---- 2. use case ----
fig, ax = canvas(11, 7)
ax.add_patch(FancyBboxPatch((2.6, 0.05), 5.8, 6.65, boxstyle="round,pad=0,rounding_size=0.15", fc="white", ec=NAVY, lw=1.3))
ax.text(5.5, 6.45, "RAG Hallucination Detector", ha="center", fontsize=12, fontweight="bold", color=NAVY)


def actor(x, y, name):
    ax.add_patch(plt.Circle((x, y + 0.75), 0.17, fc="white", ec=NAVY, lw=1.5))
    ax.plot([x, x], [y + 0.58, y + 0.15], color=NAVY, lw=1.5); ax.plot([x - 0.3, x + 0.3], [y + 0.45, y + 0.45], color=NAVY, lw=1.5)
    ax.plot([x, x - 0.25], [y + 0.15, y - 0.2], color=NAVY, lw=1.5); ax.plot([x, x + 0.25], [y + 0.15, y - 0.2], color=NAVY, lw=1.5)
    ax.text(x, y - 0.45, name, ha="center", fontsize=10, fontweight="bold", color=NAVY)


user_cases = ["Detect hallucinations in own text", "Load a demo scenario", "Ask Gemini and check its answer", "Browse RAGTruth test examples",
              "View dataset statistics", "View model performance", "View detection history"]
dev_cases = ["Run notebook (build DB, train models)", "Run tests"]
pos = {}
for i, t in enumerate(user_cases):
    y = 5.8 - i * 0.68
    ax.add_patch(Ellipse((5.5, y), 4.3, 0.52, fc=LIGHT, ec=NAVY, lw=1.1)); ax.text(5.5, y, t, ha="center", va="center", fontsize=9.5)
    pos[t] = y
for i, t in enumerate(dev_cases):  # developer use cases below the user ones
    y = 1.0 - i * 0.6
    ax.add_patch(Ellipse((5.5, y), 4.3, 0.5, fc="#E8F5F3", ec=TEAL, lw=1.1)); ax.text(5.5, y, t, ha="center", va="center", fontsize=9.5)
actor(1.2, 4.2, "User\n(student / teacher)")
actor(1.2, 0.9, "Developer")
actor(9.8, 3.8, "Gemini API")
for t, y in pos.items():
    arrow(ax, (1.55, 4.65), (3.35, y), style="-")
arrow(ax, (1.55, 1.35), (3.35, 1.0), style="-", color=TEAL)
arrow(ax, (1.55, 1.35), (3.35, 0.4), style="-", color=TEAL)
arrow(ax, (9.45, 4.25), (7.65, pos["Ask Gemini and check its answer"]), style="-", color=RED)
save(fig, "diagram_use_case.png")

# ---- 3. module / class diagram ----
fig, ax = canvas(11, 7.2)


def cls(x, y, w, name, attrs, ops, fc=LIGHT, ec=NAVY):
    lines_a, lines_o = len(attrs), len(ops)
    h = 0.42 + 0.22 * (lines_a + lines_o) + 0.2
    ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle="round,pad=0,rounding_size=0.05", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y - 0.22, name, ha="center", va="center", fontweight="bold", color=NAVY, fontsize=10)
    ax.plot([x, x + w], [y - 0.42, y - 0.42], color=ec, lw=1)
    yy = y - 0.58
    for a in attrs:
        ax.text(x + 0.1, yy, a, fontsize=8.3, va="center", family="Consolas"); yy -= 0.22
    if attrs:
        ax.plot([x, x + w], [yy + 0.1, yy + 0.1], color=ec, lw=0.8); yy -= 0.05
    for o in ops:
        ax.text(x + 0.1, yy, o, fontsize=8.3, va="center", family="Consolas"); yy -= 0.22
    return (x, y - h, w, h)


app = cls(0.2, 7.0, 3.4, "«module» app.py", ["SCENARIOS: dict", "st.session_state"],
          ["test_examples()", "highlight(result)", "gold_html(resp, labels)", "show_result(ctx, resp)", "run_detection(ctx, resp)", "gemini_key()"])
be = cls(4.0, 7.0, 3.4, "«module» backend.py", ["DB_PATH, MODEL_PATH", "FEATURES: list[10]", "CANDIDATES: dict", "GEMINI_MODELS: list"],
         ["build_db() / query(sql)", "load_dataset()", "split_sentences(text)", "sentence_features(s, idx, i, n)", "train(sent) -> (results, best)",
          "detect(ctx, resp) -> dict", "gemini_generate(prompt, key)", "gemini_passages() / gemini_answer()"])
tst = cls(0.2, 3.6, 3.4, "«module» test_app.py", [], ["fake_gemini(prompt, key, model)", "test_* (7 tests)"], fc="#E8F5F3", ec=TEAL)
nb = cls(0.2, 1.6, 3.4, "«notebook» RAGTruth_...ipynb", [], ["Run All: build → EDA → train → launch"], fc="white")
sk = cls(7.8, 7.0, 3.0, "«sklearn» classifiers", [], ["LogisticRegression", "RandomForestClassifier", "HistGradientBoosting-", "   Classifier (chosen)", "predict_proba(X)"], fc="#FBE9EB", ec=RED)
db = cls(7.8, 4.6, 3.0, "«sqlite3» ragtruth.db", [], ["sources", "responses", "runs", "predictions"], fc="white")
gm = cls(7.8, 2.4, 3.0, "«REST» Gemini API", [], ["models/{m}:generateContent"], fc="#FBE9EB", ec=RED)
arrow(ax, (3.6, 5.8), (4.0, 5.8), "uses", at=(3.62, 6.0))
arrow(ax, (3.6, 3.0), (4.0, 4.2), "tests (fake Gemini)", ls="--", at=(1.2, 3.8))
arrow(ax, (3.6, 1.2), (4.0, 3.9), "imports", ls="--", at=(3.9, 2.0))
arrow(ax, (7.4, 6.4), (7.8, 6.4), "trains", at=(7.42, 6.6))
arrow(ax, (7.4, 4.3), (7.8, 4.0), "SQL", at=(7.45, 4.45))
arrow(ax, (7.4, 3.9), (7.8, 2.1), "HTTPS", at=(7.65, 3.0))
save(fig, "diagram_class.png")

# ---- 4. activity diagram (detection) ----
fig, ax = canvas(8, 10.5)
cx = 3.4


def step(y, t, fc=LIGHT, ec=NAVY):
    box(ax, cx - 1.9, y, 3.8, 0.6, t, fc=fc, ec=ec, r=0.25, size=9.5)


ax.add_patch(plt.Circle((cx, 10.1), 0.15, color=NAVY))
step(9.1, "Enter context + response\n(own text, demo, Gemini or RAGTruth)")
ax.add_patch(plt.Polygon([[cx, 8.6], [cx + 1.2, 8.1], [cx, 7.6], [cx - 1.2, 8.1]], fc="white", ec=NAVY, lw=1.2))
ax.text(cx, 8.1, "both filled?", ha="center", va="center", fontsize=9.5)
box(ax, 5.8, 7.8, 2.0, 0.6, "Show warning\n“Fill in both …”", fc="#FBE9EB", ec=RED, r=0.25, size=9)
steps = ["split_sentences(response)", "context_index(context)\nwords, bigrams, trigrams", "sentence_features()\n10 features per sentence",
         "model.predict_proba(X)", "flag sentences with p ≥ threshold (0.66)\nrisk score = max p", "INSERT into predictions (SQLite)",
         "Render: red sentences, underlined\nunsupported words, metrics, table"]
ys = [6.7 - i * 0.9 for i in range(len(steps))]
for y, t in zip(ys, steps):
    step(y, t, fc="#FBE9EB" if "flag" in t else LIGHT)
arrow(ax, (cx, 9.95), (cx, 9.7)); arrow(ax, (cx, 9.1), (cx, 8.6))
arrow(ax, (cx, 7.6), (cx, ys[0] + 0.6), "yes"); arrow(ax, (cx + 1.2, 8.1), (5.8, 8.1), "no")
for a, b2 in zip(ys, ys[1:]):
    arrow(ax, (cx, a), (cx, b2 + 0.6))
ax.add_patch(plt.Circle((cx, ys[-1] - 0.45), 0.17, fc="white", ec=NAVY, lw=1.5)); ax.add_patch(plt.Circle((cx, ys[-1] - 0.45), 0.1, color=NAVY))
arrow(ax, (cx, ys[-1]), (cx, ys[-1] - 0.28))
save(fig, "diagram_activity.png")

# ---- 5. ER diagram ----
fig, ax = canvas(11.8, 2.8)


def entity(x, y, name, cols):
    h = 0.45 + 0.27 * len(cols)
    ax.add_patch(FancyBboxPatch((x, y - h), 2.45, h, boxstyle="round,pad=0,rounding_size=0.05", fc="white", ec=NAVY, lw=1.3))
    ax.add_patch(FancyBboxPatch((x, y - 0.45), 2.45, 0.45, boxstyle="round,pad=0,rounding_size=0.05", fc=NAVY, ec=NAVY))
    ax.text(x + 1.22, y - 0.22, name, ha="center", va="center", color="white", fontweight="bold")
    for i, c in enumerate(cols):
        ax.text(x + 0.1, y - 0.62 - i * 0.27, c, fontsize=8.5, va="center", family="Consolas",
                color=RED if "PK" in c or "FK" in c else INK)
    return h


entity(0.2, 2.7, "sources", ["source_id  TEXT  PK", "task_type  TEXT", "context    TEXT"])
entity(3.5, 2.7, "responses", ["id         TEXT  PK", "source_id  TEXT  FK", "model      TEXT", "split      TEXT", "quality    TEXT",
                               "response   TEXT", "labels     TEXT (JSON)", "hallucinated INT"])
entity(6.5, 2.7, "runs", ["id  INT  PK (auto)", "created_at, model_name", "level (sentence/response)", "accuracy, precision", "recall, f1, roc_auc", "threshold, chosen"])
entity(9.2, 2.7, "predictions", ["id  INT  PK (auto)", "created_at", "context, response", "score", "flagged_sentences", "verdict"])
arrow(ax, (2.65, 2.0), (3.5, 2.0), style="-")
ax.text(2.75, 2.08, "1", fontsize=10, fontweight="bold"); ax.text(3.33, 2.08, "N", fontsize=10, fontweight="bold")
ax.text(3.07, 1.75, "has", fontsize=9, color=GRAY, ha="center")
save(fig, "diagram_er.png")
print("diagrams done")
