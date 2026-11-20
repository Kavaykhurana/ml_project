"""Regenerate the report/PPT figures from the trained project (run the notebook first).

    python docs/make_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import backend as b

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 12, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
                     "figure.dpi": 200, "savefig.bbox": "tight", "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True})


def save(fig, name):
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)


def barh(series, title, name, fmt="{:.0%}", color=BLUE):
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(series) + 1.2))
    ax.barh(series.index, series.values, color=color, height=0.6)
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(series.values):
        ax.text(v, i, " " + fmt.format(v), va="center", color=INK, fontsize=11)
    ax.set_xlim(0, series.max() * 1.18)
    ax.set_title(title)
    save(fig, name)


rate = b.query("SELECT model, AVG(hallucinated) rate FROM responses GROUP BY model ORDER BY rate").set_index("model").rate
barh(rate, "Share of responses with a hallucination, by LLM", "eda_by_llm.png")
task = b.query("""SELECT task_type, AVG(hallucinated) rate FROM responses JOIN sources USING(source_id)
                  GROUP BY task_type ORDER BY rate""").set_index("task_type").rate
barh(task, "Share of responses with a hallucination, by task", "eda_by_task.png")
spans = b.query("SELECT labels FROM responses WHERE hallucinated=1").labels.apply(json.loads).explode().dropna()
barh(spans.apply(lambda l: l["label_type"]).value_counts().sort_values(), "Annotated hallucination span types",
     "eda_span_types.png", fmt="{:,.0f}")

runs = b.query("SELECT * FROM runs WHERE created_at = (SELECT MAX(created_at) FROM runs)")
for level in ("sentence", "response"):
    r = runs[runs.level == level].set_index("model_name")[["precision", "recall", "f1", "roc_auc"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.26
    for k, (model, color) in enumerate(zip(r.index, (BLUE, ORANGE, AQUA))):
        xs = [i + (k - 1) * w for i in range(4)]
        ax.bar(xs, r.loc[model], width=w - 0.02, color=color, label=model)
        if model == "Gradient Boosting":
            for x, v in zip(xs, r.loc[model]):
                ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=10, color=INK)
    ax.set_xticks(range(4), ["Precision", "Recall", "F1", "ROC-AUC"])
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.02))
    ax.set_title(f"{level.capitalize()}-level results on the held-out test set", pad=28)
    save(fig, f"models_{level}.png")

df = b.load_dataset()
sent = b.sentence_table(df)
means = sent[sent.split == "train"].groupby("label")[["novel_word_ratio", "novel_number_ratio", "bigram_overlap", "trigram_overlap"]].mean().T
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar([i - 0.2 for i in range(4)], means[0], width=0.38, color=BLUE, label="Grounded sentence")
ax.bar([i + 0.2 for i in range(4)], means[1], width=0.38, color=ORANGE, label="Hallucinated sentence")
ax.set_xticks(range(4), ["Novel word ratio", "Novel number ratio", "Bigram overlap", "Trigram overlap"])
ax.grid(axis="x", visible=False)
ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.02))
ax.set_ylim(0, 0.65)
ax.set_title("Feature averages: grounded vs hallucinated (train)", pad=28)
save(fig, "feature_means.png")

bundle = b.load_model()
te = sent[sent.split == "test"].sample(5000, random_state=0)
imp = pd.Series(permutation_importance(bundle["model"], te[b.FEATURES], te.label, scoring="roc_auc",
                                       n_repeats=3, random_state=0).importances_mean, b.FEATURES).sort_values()
barh(imp, f"Permutation importance ({bundle['name']}, drop in AUC)", "feature_importance.png", fmt="{:.3f}")

te = sent[sent.split == "test"].copy()
te["prob"] = bundle["model"].predict_proba(te[b.FEATURES])[:, 1]
f1 = {t: b.metrics(*g.groupby("response_id").agg(y=("label", "max"), p=("prob", "max")).T.values, bundle["threshold"])["f1"]
      for t, g in te.groupby("task_type")}
barh(pd.Series(f1).sort_values(), "Response-level F1 by task (test)", "f1_by_task.png", fmt="{:.2f}")
print("figures written to", OUT)
