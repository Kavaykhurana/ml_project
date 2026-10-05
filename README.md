# RAG Hallucination Detector (RAGTruth)

This project detects hallucinations in RAG (retrieval-augmented generation) answers. You give it the retrieved **context** and the LLM's **response**. It flags the sentences the context doesn't support, underlines the words that never appear in the context and returns a risk score for the whole answer.

It's trained on [RAGTruth](https://github.com/ParticleMedia/RAGTruth) (MIT licence), which has 17,790 responses from 6 LLMs across QA, Summary and Data-to-text tasks. Humans annotated the hallucinated character spans.

## Quick start (2 steps)

> **You need your own Gemini API key.** Each person who uses this project must use their own key; this project doesn't include one. Get a free key at **https://aistudio.google.com/apikey**. It's only needed for the *Ask Gemini* mode; everything else works without it.

**1. Open the notebook and click Run → Run All**

```bash
pip install jupyter          # only if you don't have Jupyter yet
jupyter notebook RAGTruth_Hallucination_Detector.ipynb
```

**2. Paste your Gemini API key when the notebook asks (step 2 of the notebook) and press Enter**

That's all. The rest runs by itself and the app opens in your browser at **http://localhost:8501**.

**Where your key goes:** the notebook saves it to `.streamlit/secrets.toml` in the project folder, as `GEMINI_API_KEY = "your-key"`. It's asked only once. To change the key, edit that file; to remove it, delete the file. The file stays on your computer: git ignores it, and the shared zip doesn't include it. If you skip the prompt, you can also paste a key straight into the app's *Ask Gemini* screen.

What Run All does, in order:

1. installs the requirements
2. asks for your Gemini API key (once)
3. downloads the dataset (~36 MB) and loads it into a SQLite database (`ragtruth.db`)
4. runs the EDA charts and builds the features
5. trains and compares 3 models (about 10 seconds)
6. launches the Streamlit app at **http://localhost:8501**, or the next free port, and opens it in your browser

To stop the app, interrupt the last cell (■ button). Everything runs locally except the *Ask Gemini* calls to Google.

## Files

| File | What it is |
|---|---|
| `RAGTruth_Hallucination_Detector.ipynb` | Main notebook: run-all walkthrough + app launcher |
| `backend.py` | Python backend: download, SQLite, features, training, inference, Gemini call |
| `app.py` | Streamlit frontend (4 tabs) |
| `.streamlit/config.toml` | Light theme, binds to localhost only |
| `requirements.txt` | Dependencies |
| `test_app.py` | End-to-end UI tests |

Generated at runtime (git-ignored): `data/` (raw jsonl), `ragtruth.db`, `model.joblib`.

## How it works

```
response.jsonl + source_info.jsonl
        │  backend.build_db()
        ▼
SQLite: sources ─< responses           (+ runs, predictions)
        │  load_dataset(): drop refusals/truncated, group dev split by source_id
        ▼
split each response into sentences → label 1 if it overlaps a gold span
        │  sentence_features(): compare sentence vs context
        ▼
LogReg / RandomForest / GradientBoosting → threshold + model picked on dev → best saved
        │
        ▼
detect(context, response) → per-sentence probability, flagged sentences, risk score = max prob
```

**Features** (per sentence, computed against the context):

| Feature | Intuition |
|---|---|
| `novel_word_ratio`, `novel_words` | content words that never appear in the context |
| `novel_numbers`, `novel_number_ratio` | invented numbers and dates |
| `novel_capitals` | invented names and entities |
| `bigram_overlap`, `trigram_overlap` | how much of the sentence is copied from the context |
| `n_words`, `position`, `is_last` | length and position in the answer |

**Splits.** RAGTruth's official train/test split is used. A dev set (10%) is carved from train, grouped by `source_id` (`md5(source_id) % 10 == 0`). All 6 LLM answers to the same passage therefore stay in one split, so nothing leaks.

## Results (held-out test set)

| Model | Sentence F1 | Sentence AUC | Response F1 | Response AUC |
|---|---|---|---|---|
| Logistic Regression | 0.301 | 0.771 | 0.644 | 0.781 |
| Random Forest | 0.332 | 0.786 | 0.647 | 0.796 |
| **Gradient Boosting** (chosen) | **0.338** | **0.797** | **0.655** | 0.800 |

The model is picked by F1 on the **dev** split, and its threshold is tuned there too. The test split is only used for reporting, so it never influences any choice. Gradient Boosting had the best dev F1 (0.392).

*Response level* means an answer counts as hallucinated if any of its sentences is flagged. Sentence-level F1 is low because only 8% of test sentences are hallucinated and the features are lexical. Response-level detection, the "should I trust this answer?" question, is the practical use.

## Database schema (SQLite, `ragtruth.db`)

| Table | Columns |
|---|---|
| `sources` | `source_id` PK, `task_type`, `context` |
| `responses` | `id` PK, `source_id` FK → sources, `model`, `split`, `quality`, `response`, `labels` (JSON spans), `hallucinated` |
| `runs` | one row per model × level per training run: accuracy, precision, recall, f1, roc_auc, threshold, chosen |
| `predictions` | every detection made in the app: timestamp, context, response, score, flagged count, verdict |

## The app

- **Detector** has 3 input modes:
  - **Write my own** (default): type any context and any response, then click Detect. Four demo scenarios (invented fact, wrong numbers, fully grounded, made-up news claim) fill the boxes so you can start from one and edit it. **Clear** empties both boxes.
  - **Ask Gemini**: type a question. Passages are optional: leave them empty and Gemini first writes the reference passages (simulated retrieval), then answers. The detector checks the answer against the passages. *Strictly from passages* usually comes out grounded. *Add outside facts* asks Gemini to add facts from its own knowledge, and those sentences get flagged. The default model is `gemini-3.5-flash-lite`, and `gemini-3.8-flash` is selectable. Busy or rate-limited models (429/503) are retried, then the app switches to the other model automatically.
  - **RAGTruth test example**: browse real test answers and compare the prediction with the human gold labels.
- **Dataset**: stats and charts queried live from SQLite, plus a table browser.
- **Model Performance**: all 3 models compared at sentence and response level.
- **History**: every detection logged in the `predictions` table.

## Sharing the project (zip)

Don't zip the folder by hand. `.streamlit/secrets.toml` holds the Gemini API key, and a hand-made zip would include it, along with the 36 MB dataset, the database and the trained model. Instead, share the clean archive made from git:

```bash
git archive -o ragtruth-share.zip HEAD
```

It contains only the code, the notebook and the README. There's no key and no data. Your friend follows the 2-step quick start: Run All, then paste **their own** Gemini key when asked.

## Demo script (about 5 minutes)

1. **Run All** in the notebook. Walk through the EDA charts, the feature table (hallucinated sentences have more novel words and less n-gram overlap) and the model comparison.
2. In the app, **Write my own** → load **Invented fact** → Detect. The second sentence is red and "Leonardo da Vinci" is underlined because it isn't in the context.
3. Load **Fully grounded** → Detect. It comes out "Looks grounded".
4. **Ask Gemini** → type any question, leave passages empty → *Strictly from passages* → it's grounded. Switch to *Add outside facts* → the extra facts get flagged.
5. Let the teacher type their own: a short paragraph of facts as context, and an answer that repeats some facts and adds one new name, number or claim.
6. **RAGTruth test example** → compare the model's red sentences with the yellow human labels.
7. **Model Performance** and **History** tabs: metrics and every detection you ran, read from SQLite.

## Testing

```bash
python backend.py     # self-check: sentence splitting, features, Gemini retry/fallback (no data needed)
python test_app.py    # 7 end-to-end UI tests (needs the notebook to have run once; Gemini is faked)
```

`test_app.py` drives the real Streamlit app headlessly. It covers your own input, all demo scenarios, Clear, text surviving a mode switch, Gemini with no passages, a bad key, RAGTruth examples, and HTML escaping. It writes to a temporary copy of the database, so your History stays clean.

## Limitations

Response-level F1 on the test set differs a lot by task type: Data2txt 0.78, QA 0.53, Summary 0.39. On Data2txt the model flags almost everything (recall 0.98, precision 0.65). The context there is JSON, so any reworded field looks like new words.


The features are lexical. A correct paraphrase can look "novel", and a contradiction that reuses context words (for example, swapped numbers) can look grounded. Next steps would be a transformer token classifier (ModernBERT or DeBERTa on `[context] [SEP] [response]`) for word-level spans, or an NLI entailment score added as a feature.
