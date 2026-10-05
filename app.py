"""Streamlit frontend. Run via the notebook, or: streamlit run app.py"""
import html
import json

import pandas as pd
import streamlit as st

import backend as b

st.set_page_config(page_title="RAG Hallucination Detector", page_icon="🔎", layout="wide")


@st.cache_data
def test_examples():
    return b.query("""SELECT r.id, r.model, r.response, r.labels, s.task_type, s.context
                      FROM responses r JOIN sources s USING(source_id)
                      WHERE r.split='test' AND r.quality='good' ORDER BY CAST(r.id AS INTEGER)""")


def highlight(result):
    """Response HTML: flagged sentences shaded red, unsupported words underlined."""
    parts = []
    for s in result["sentences"]:
        text = html.escape(s["text"])
        for w in sorted(set(s["unsupported_words"]), key=len, reverse=True):
            text = text.replace(html.escape(w), f"<u><b>{html.escape(w)}</b></u>")
        bg = f"rgba(220,38,38,{0.15 + 0.5 * s['prob']:.2f})" if s["flagged"] else "transparent"
        parts.append(f"<span title='p={s['prob']:.2f}' style='background:{bg};padding:2px;border-radius:4px'>{text}</span>")
    return "<div style='line-height:1.9;font-size:1.05rem'>" + " ".join(parts) + "</div>"


def gold_html(response, labels):
    out, pos = [], 0
    for l in sorted(labels, key=lambda l: l["start"]):
        if l["start"] < pos:
            continue
        out += [html.escape(response[pos:l["start"]]),
                f"<mark title='{html.escape(l['label_type'])}' style='background:#fde68a'>{html.escape(response[l['start']:l['end']])}</mark>"]
        pos = l["end"]
    return "<div style='line-height:1.9'>" + "".join(out) + html.escape(response[pos:]) + "</div>"


st.title("🔎 RAG Hallucination Detector")
st.caption("Trained on RAGTruth (ParticleMedia). Paste the retrieved context and the LLM answer. "
           "The model flags the sentences the context does not support.")

if not b.MODEL_PATH.exists():
    st.error("No trained model found. Run the notebook first.")
    st.stop()

tab_detect, tab_data, tab_model, tab_history = st.tabs(["Detector", "Dataset", "Model Performance", "History"])

EIFFEL = "The Eiffel Tower is in Paris. It was completed in 1889 and is 330 metres tall."
APPLE = ("Apple Inc. reported quarterly revenue of 94.9 billion dollars for the fourth quarter of 2024, "
         "up 6 percent year over year. CEO Tim Cook said iPhone sales reached a record in emerging markets. "
         "The company is headquartered in Cupertino, California.")
SCENARIOS = {
    "Invented fact": (EIFFEL, "The Eiffel Tower is in Paris and was completed in 1889. It was designed by Leonardo da Vinci."),
    "Wrong numbers": (EIFFEL, "The Eiffel Tower is in Paris. It was completed in 1925 and is 500 metres tall."),
    "Fully grounded": (EIFFEL, "The Eiffel Tower is in Paris. It was completed in 1889 and is 330 metres tall."),
    "News: made-up claim": (APPLE, "Apple reported revenue of 94.9 billion dollars in the fourth quarter of 2024. "
                                   "Tim Cook announced that the company will move its headquarters to Austin, Texas next year."),
}

for k in ("ctx", "resp"):
    st.session_state.setdefault(k, "")


def load_scenario():
    name = st.session_state.scenario
    if name:  # deselecting a scenario keeps whatever the user has typed
        st.session_state.ctx, st.session_state.resp = SCENARIOS[name]


def clear_inputs():
    st.session_state.ctx = st.session_state.resp = ""
    st.session_state.scenario = None


def show_result(context, response, labels=None):
    """Render the last detection, but only if it belongs to the inputs currently on screen."""
    res = st.session_state.get("result")
    if not res or res["context"] != context or res["response"] != response:
        return
    r = res["output"]
    with st.container(border=True):
        m1, m2, m3 = st.columns(3)
        m1.metric("Verdict", r["verdict"])
        m2.metric("Risk score", f"{r['score']:.2f}", help=f"Max sentence probability. Threshold {r['threshold']:.2f}")
        m3.metric("Flagged sentences", f"{sum(s['flagged'] for s in r['sentences'])} / {len(r['sentences'])}")
        st.subheader("Model prediction")
        st.markdown(highlight(r), unsafe_allow_html=True)
        st.caption("Red = flagged sentence (darker = more confident). Underlined = words not found in the context. "
                   "Hover a sentence to see its probability.")
        if labels is not None:
            st.subheader("Gold annotation (human labels)")
            st.markdown(gold_html(response, labels) if labels else "<i>No hallucination annotated.</i>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(r["sentences"])[["text", "prob", "flagged", "unsupported_words"]],
                     width="stretch", hide_index=True)


def run_detection(context, response):
    if not context.strip() or not response.strip():
        st.warning("Fill in both the context and the response.")
        return
    st.session_state.result = {"context": context, "response": response, "output": b.detect(context, response)}


with tab_detect:
    mode = st.segmented_control("Input", ["Write my own", "RAGTruth test example"], default="Write my own", key="mode")
    if mode == "RAGTruth test example":
        ex = test_examples()
        c1, c2 = st.columns(2)
        task = c1.selectbox("Task type", sorted(ex.task_type.unique()))
        only_h = c2.checkbox("Only examples with hallucinations", value=True)
        sub = ex[ex.task_type == task]
        if only_h:
            sub = sub[sub.labels != "[]"]
        i = st.slider("Example #", 0, len(sub) - 1, 0)
        row = sub.iloc[i]
        st.caption(f"Response id {row.id} · generated by {row.model}")
        with st.expander("Context (retrieved evidence)"):
            st.text(row.context)
        st.text_area("Response", row.response, height=150, disabled=True)
        if st.button("Detect hallucinations", type="primary", icon=":material/search:"):
            run_detection(row.context, row.response)
        show_result(row.context, row.response, json.loads(row.labels))
    else:
        st.markdown("Type any **context** (the facts the LLM was given) and any **response** (the LLM's answer), "
                    "then click **Detect**. Need a quick start? Load a demo scenario and edit it.")
        with st.container(horizontal=True, vertical_alignment="bottom"):
            st.segmented_control("Demo scenarios", list(SCENARIOS), key="scenario", on_change=load_scenario)
            st.button("Clear", icon=":material/close:", on_click=clear_inputs)
        with st.form("own_input"):
            st.text_area("Context (retrieved evidence)", key="ctx", height=160,
                         placeholder="e.g. The Eiffel Tower is in Paris. It was completed in 1889 and is 330 metres tall.")
            st.text_area("LLM response", key="resp", height=120,
                         placeholder="e.g. The Eiffel Tower is in Paris. It was designed by Leonardo da Vinci.")
            submitted = st.form_submit_button("Detect hallucinations", type="primary", icon=":material/search:")
        if submitted:
            run_detection(st.session_state.ctx, st.session_state.resp)
        show_result(st.session_state.ctx, st.session_state.resp)

with tab_data:
    stats = b.query("""SELECT COUNT(*) responses, SUM(hallucinated) hallucinated,
                       (SELECT COUNT(*) FROM sources) sources FROM responses""").iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Responses", f"{stats.responses:,}")
    c2.metric("Source passages", f"{stats.sources:,}")
    c3.metric("Hallucinated", f"{stats.hallucinated / stats.responses:.1%}")
    c1, c2 = st.columns(2)
    c1.markdown("**Hallucination rate by generator model**")
    c1.bar_chart(b.query("SELECT model, AVG(hallucinated) rate FROM responses GROUP BY model ORDER BY rate DESC").set_index("model"))
    c2.markdown("**Hallucination rate by task**")
    c2.bar_chart(b.query("""SELECT task_type, AVG(hallucinated) rate FROM responses JOIN sources USING(source_id)
                            GROUP BY task_type""").set_index("task_type"))
    spans = b.query("SELECT labels FROM responses WHERE hallucinated=1").labels.apply(json.loads).explode().dropna()
    st.markdown("**Annotated span types**")
    st.bar_chart(spans.apply(lambda l: l["label_type"]).value_counts())
    st.markdown("**Browse the database**")
    st.dataframe(b.query("""SELECT r.id, s.task_type, r.model, r.split, r.hallucinated, r.response
                            FROM responses r JOIN sources s USING(source_id) LIMIT 500"""),
                 width="stretch", hide_index=True)

with tab_model:
    runs = b.query("SELECT * FROM runs WHERE created_at = (SELECT MAX(created_at) FROM runs)")
    best = runs[runs.chosen == 1].model_name.iloc[0]
    st.markdown(f"Three classifiers compared on the **held-out test split**. Threshold tuned on dev. Chosen: **{best}**.")
    level = st.radio("Level", ["sentence", "response"], horizontal=True,
                     help="Response level: an answer counts as hallucinated if any sentence is flagged.")
    r = runs[runs.level == level].set_index("model_name")[["accuracy", "precision", "recall", "f1", "roc_auc", "threshold"]]
    st.dataframe(r.style.format("{:.3f}").highlight_max(axis=0, color="#bbf7d0"), width="stretch")
    st.bar_chart(r[["precision", "recall", "f1", "roc_auc"]].T, stack=False)
    st.markdown("**Features used** (computed per sentence against the context): " + ", ".join(f"`{f}`" for f in b.FEATURES))

with tab_history:
    hist = b.query("SELECT id, created_at, verdict, ROUND(score,3) score, flagged_sentences, response FROM predictions ORDER BY id DESC")
    st.markdown(f"**{len(hist)}** detections logged in SQLite (`predictions` table).")
    st.dataframe(hist, width="stretch", hide_index=True)
