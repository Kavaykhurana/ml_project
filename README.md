# RAG Hallucination Detector (RAGTruth)

This project detects hallucinations in RAG (retrieval-augmented generation) answers. You give it the retrieved **context** and the LLM's **response**. It flags the sentences the context doesn't support, underlines the words that never appear in the context and returns a risk score for the whole answer.

It's trained on [RAGTruth](https://github.com/ParticleMedia/RAGTruth) (MIT licence), which has 17,790 responses from 6 LLMs across QA, Summary and Data-to-text tasks. Humans annotated the hallucinated character spans.

## Quick start

```bash
pip install jupyter          # if you don't have it
jupyter notebook RAGTruth_Hallucination_Detector.ipynb
```

Then click **Run → Run All**. The notebook will:

1. install the requirements
2. download the dataset (~36 MB) and load it into a SQLite database (`ragtruth.db`)
3. run the EDA charts
4. build the features
5. train and compare 3 models (about 10 seconds)
6. launch the Streamlit app at **http://localhost:8501**, which opens in your browser

To stop the app, interrupt the last cell (■ button). Everything runs locally. The only optional online part is the **Ask Gemini** mode.

### Optional: Gemini API key (only for the "Ask Gemini" mode)

- **What:** a Google Gemini API key. It's free from [Google AI Studio](https://aistudio.google.com/apikey).
- **Why:** in Ask Gemini mode, Gemini *generates* the answer and the trained model *checks* it for hallucinations.
- **Where:** put it in `.streamlit/secrets.toml` as `GEMINI_API_KEY = "..."`. That file is git-ignored, so the key never reaches GitHub. You can also set the `GEMINI_API_KEY` environment variable instead. With neither set, the app shows a password field.

Everything else works without a key.

You can also run the app on its own once the notebook has trained the model: `streamlit run app.py`

## Files

| File | What it is |
|---|---|
| `RAGTruth_Hallucination_Detector.ipynb` | Main notebook: run-all walkthrough + app launcher |
| `backend.py` | Python backend: download, SQLite, features, training, inference, Gemini call |
| `app.py` | Streamlit frontend (4 tabs) |
| `.streamlit/config.toml` | Light theme, binds to localhost only |
| `requirements.txt` | Dependencies |

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
| Logistic Regression | 0.300 | 0.770 | 0.646 | 0.782 |
| Random Forest | 0.337 | 0.784 | 0.652 | 0.795 |
| **Gradient Boosting** (chosen) | 0.331 | **0.796** | 0.651 | **0.802** |

The model is picked by F1 on the **dev** split, and its threshold is tuned there too. The test split is only used for reporting, so it never influences any choice. Gradient Boosting had the best dev F1 (0.387).

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
  - **Ask Gemini**: type a question and passages. Gemini writes the answer and the detector checks it against the passages. *Strictly from passages* usually comes out grounded. *Add outside facts* asks Gemini to add facts from its own knowledge, and those sentences get flagged. The default model is `gemini-3.5-flash-lite`, and `gemini-3.8-flash` is selectable.
  - **RAGTruth test example**: browse real test answers and compare the prediction with the human gold labels.
- **Dataset**: stats and charts queried live from SQLite, plus a table browser.
- **Model Performance**: all 3 models compared at sentence and response level.
- **History**: every detection logged in the `predictions` table.

## Demo script (about 5 minutes)

1. **Run All** in the notebook. Walk through the EDA charts, the feature table (hallucinated sentences have more novel words and less n-gram overlap) and the model comparison.
2. In the app, **Write my own** → load **Invented fact** → Detect. The second sentence is red and "Leonardo da Vinci" is underlined because it isn't in the context.
3. Load **Fully grounded** → Detect. It comes out "Looks grounded".
4. **Ask Gemini** → Load example → *Strictly from passages* → it's grounded. Switch to *Add outside facts* → the extra facts get flagged.
5. Let the teacher type their own: a short paragraph of facts as context, and an answer that repeats some facts and adds one new name, number or claim.
6. **RAGTruth test example** → compare the model's red sentences with the yellow human labels.
7. **Model Performance** and **History** tabs: metrics and every detection you ran, read from SQLite.

## Testing

`python backend.py` runs an assert-based self-check of sentence splitting and feature extraction.

## Limitations

Response-level F1 on the test set differs a lot by task type: Data2txt 0.78, QA 0.51, Summary 0.41. On Data2txt the model flags almost everything (recall 0.99, precision 0.64). The context there is JSON, so any reworded field looks like new words.


The features are lexical. A correct paraphrase can look "novel", and a contradiction that reuses context words (for example, swapped numbers) can look grounded. Next steps would be a transformer token classifier (ModernBERT or DeBERTa on `[context] [SEP] [response]`) for word-level spans, or an NLI entailment score added as a feature.
