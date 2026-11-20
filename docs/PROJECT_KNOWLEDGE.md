# RAG Hallucination Detector: Complete Project Knowledge Guide

**Team:** Kavay Khurana (1024170154) · Gurliv Singh (1024170161)
**Project duration:** 10 October 2026 – 20 November 2026
**Repository:** https://github.com/Kavaykhurana/ml_project

This document explains *everything* in the project: the problem, the data, the ML model, how the LLM (Gemini) is used, every file and function, the results, the demos to show, and the questions a teacher is likely to ask. Read it top to bottom once before the evaluation.

---

## 1. The project in one paragraph

Large Language Models (LLMs) are often used in **RAG (Retrieval-Augmented Generation)** systems: a search step retrieves some documents (the **context**), and the LLM writes an answer (the **response**) from them. Even with the right documents, LLMs sometimes add facts that the documents never said. These are **hallucinations**. Our project is a **hallucination detector**. You give it the context and the response. It splits the response into sentences, computes 10 features per sentence that compare it with the context, and a trained **Gradient Boosting classifier** gives each sentence a probability of being hallucinated. Sentences above the threshold are shaded red, words not found in the context are underlined, and the whole answer gets a **risk score**. It's trained on the **RAGTruth** dataset (17,790 human-annotated LLM responses) and comes with a Streamlit web app, a SQLite database, and a Google **Gemini** integration that generates live answers for the model to check.

---

## 2. Q1: Which ML model is used?

**Final model: Gradient Boosting**, using scikit-learn's `HistGradientBoostingClassifier`.

We trained and compared **three** supervised binary classifiers on the same features:

| # | Model | scikit-learn class | Role |
|---|---|---|---|
| 1 | Logistic Regression | `LogisticRegression` (inside a `StandardScaler` pipeline) | Linear baseline |
| 2 | Random Forest | `RandomForestClassifier` | Bagging ensemble of trees |
| 3 | **Gradient Boosting** ✅ | `HistGradientBoostingClassifier` | Boosting ensemble of trees (**chosen**) |

Gradient Boosting was chosen automatically because it had the **highest F1 on the dev (validation) split** (0.384 vs 0.379 RF vs 0.351 LR). The test split was never used for choosing, only for reporting.

**Type of ML problem:** supervised learning, binary classification (sentence is hallucinated = 1, grounded = 0), with imbalanced classes (only ~11% of sentences are hallucinated).

---

## 3. Q2: Model details

### 3.1 Exact configuration (from `backend.py`, `CANDIDATES`)

```python
"Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
"Random Forest":       RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=0)
"Gradient Boosting":   HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0)
```

Full hyper-parameters of the saved Gradient Boosting model (defaults included):

| Parameter | Value | Meaning |
|---|---|---|
| `loss` | `log_loss` | Binary cross-entropy, so the output is a probability |
| `learning_rate` | 0.05 | How much each new tree corrects the previous ones (small = slower but safer) |
| `max_iter` | 300 | Upper limit on the number of boosting rounds (trees) |
| `n_iter_` (actual) | **148** | Early stopping kicked in after 148 trees |
| `early_stopping` | `auto` → **on** | Turned on automatically because training data > 10,000 rows; 10% of train is held aside internally and training stops when the loss doesn't improve for 10 rounds |
| `max_leaf_nodes` | 31 | Each tree has at most 31 leaves |
| `min_samples_leaf` | 20 | Each leaf needs ≥ 20 sentences (prevents overfitting) |
| `max_bins` | 255 | Each feature is bucketed into ≤ 255 bins (the "Hist" part, makes it fast) |
| `class_weight` | `balanced` | Hallucinated sentences get ~8× more weight since they are rare |
| `random_state` | 0 | Reproducible results |
| **Decision threshold** | **0.66** | Tuned on dev for best F1 (not the default 0.5) |

### 3.2 How Gradient Boosting works (explain it in your own words)

1. Start with a constant prediction (the log-odds of the base rate).
2. Compute the **gradient** of the log-loss, i.e. how wrong each training sentence currently is.
3. Fit a small **decision tree** that predicts these errors.
4. Add that tree to the model, scaled by the learning rate (0.05).
5. Repeat. Each new tree fixes the mistakes of all previous trees.
6. The final score is a sum of all the trees' outputs, passed through a sigmoid to get a probability.

**"Hist"** means that before training, each feature's values are grouped into at most 255 bins. Finding the best split then only checks 255 candidate points instead of every unique value, so training on 110,000 sentences takes seconds.

**Why boosting beat the others:** our features interact non-linearly. For example, a high novel-word ratio matters more when bigram overlap is also low. Trees capture such interactions; Logistic Regression can only draw one straight boundary. Boosting (sequential error-fixing) usually edges out Random Forest (independent trees averaged) on tabular data.

### 3.3 The 10 features (computed per sentence *against the context*)

| Feature | What it measures | Why it signals hallucination |
|---|---|---|
| `n_words` | Number of words in the sentence | Longer sentences have more room to add things |
| `novel_word_ratio` | Fraction of content words (non-stopwords) not found in the context | Invented content uses words the source never used |
| `novel_words` | Count of those novel content words | Same, as an absolute count |
| `novel_numbers` | Numbers/dates in the sentence not in the context | Invented figures (e.g. "500 metres") |
| `novel_number_ratio` | Novel numbers ÷ all numbers in the sentence | Same, normalised |
| `novel_capitals` | Capitalised words (not the first word) not in the context | Invented names/entities ("Leonardo da Vinci") |
| `bigram_overlap` | Fraction of the sentence's word pairs that also appear in the context | Grounded sentences copy phrases from the source |
| `trigram_overlap` | Same for 3-word sequences | Stronger "copied from source" signal |
| `position` | Sentence index ÷ (number of sentences − 1) | LLMs often add unsupported extras near the end |
| `is_last` | 1 if it's the last sentence | Same idea |

**Text normalisation (`norm`)**: lower-case, strip `.` and `'`, remove a trailing plural "s" for words longer than 3 letters (so "towers" matches "tower").

**Evidence the features work** (training-set averages):

| Feature | Grounded | Hallucinated |
|---|---|---|
| novel_word_ratio | 0.295 | **0.511** |
| novel_numbers | 0.092 | **0.305** |
| novel_capitals | 0.079 | **0.259** |
| bigram_overlap | **0.345** | 0.147 |
| trigram_overlap | **0.211** | 0.060 |

**Most important features** (permutation importance = how much test AUC drops when the feature is shuffled): `novel_word_ratio` (0.053), `position` (0.045), `bigram_overlap` (0.027).

### 3.4 How labels are created

RAGTruth gives, for each response, a list of **character spans** that humans marked as hallucinated, e.g. `{"start": 120, "end": 175, "label_type": "Evident Baseless Info"}`. We split each response into sentences (keeping the character offsets), and a sentence gets **label 1 if it overlaps any span** (`span.start < sentence.end and span.end > sentence.start`), otherwise 0.

### 3.5 Train / dev / test split (no data leakage)

- RAGTruth already has an official **train** (15,090) / **test** (2,700) split; we keep it.
- We carve a **dev** set (~10%) out of train, **grouped by `source_id`**: `md5(source_id) % 10 == 0`. Every source passage was answered by 6 LLMs, so grouping keeps all 6 answers to one passage in the same split. Otherwise the model could "memorise" a passage in train and be tested on the same passage in dev.
- 173 bad responses (144 incorrect refusals and 29 truncated) are dropped (`quality = 'good'` only).

| Split | Responses | Sentences | % hallucinated sentences |
|---|---|---|---|
| train | 13,531 | 110,894 | 10.8% |
| dev | 1,411 | 11,148 | 11.6% |
| test | 2,675 | 20,299 | 8.2% |

### 3.6 Training procedure (`backend.train`)

For each of the 3 candidates:
1. Fit on **train** sentences.
2. Predict probabilities on **dev**, and search thresholds 0.05 → 0.95 (step 0.01) for the one that maximises F1 (`best_threshold`).
3. Predict on **test** and compute accuracy, precision, recall, F1, ROC-AUC at **sentence** level and at **response** level (a response is hallucinated if *any* of its sentences is).
4. Keep the model with the best **dev F1**, save it as `model.joblib` together with its name and threshold.
5. Write every result row into the SQLite `runs` table.

### 3.7 Results (held-out test set)

| Model | Sentence F1 | Sentence AUC | Response Precision | Response Recall | **Response F1** | Response AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.299 | 0.770 | 0.527 | 0.836 | 0.646 | 0.780 |
| Random Forest | 0.335 | 0.786 | 0.567 | 0.765 | 0.651 | 0.796 |
| **Gradient Boosting** | **0.338** | **0.798** | **0.571** | 0.792 | **0.664** | **0.800** |

Response-level F1 by task (Gradient Boosting): **Data2txt 0.78 · QA 0.55 · Summary 0.38**.

**How to explain the numbers:**
- **ROC-AUC 0.80** means: pick a random hallucinated response and a random clean one; 80% of the time the model gives the hallucinated one a higher risk score.
- **Response recall 0.79**: the model catches 79% of hallucinated answers.
- **Sentence F1 is low (0.34)** because only 8% of test sentences are hallucinated and our features are lexical (word matching). Response level ("should I trust this answer?") is the practical use.
- **Why we use F1, not accuracy:** a model that says "never hallucinated" would get ~92% sentence accuracy and be useless. F1 balances precision and recall on the rare class.

### 3.8 Limitations (be honest, teachers like this)

- **Lexical features only.** A correct paraphrase can look "novel", and a contradiction that reuses context words (e.g. swapping two numbers that both appear in the context) looks grounded. Demo: *"Kohli scored 45 and Rohit scored 82"* when the context says the opposite → "Looks grounded".
- **Data2txt** context is JSON, so almost any reworded field looks new: recall 0.97 but precision 0.65 there.
- **Future work:** fine-tune a transformer token classifier (ModernBERT / DeBERTa on `[context] [SEP] [response]`) for word-level spans, or add an NLI (natural language inference) entailment score as an extra feature.

---

## 4. Q3: How the LLM is managed

There are **two different roles for LLMs** in this project. Make this distinction clear.

### 4.1 LLMs as the *source of training data* (offline)

RAGTruth's 17,790 responses were written by **6 LLMs**: GPT-4-0613, GPT-3.5-turbo-0613, Llama-2-7B-chat, Llama-2-13B-chat, Llama-2-70B-chat and Mistral-7B-instruct. Humans then annotated their hallucinations. We never call these models; we only use their saved outputs. Our EDA shows the small open models hallucinate far more (Mistral-7B 66%, Llama-2-7B 62%) than GPT-4 / GPT-3.5 (14%).

### 4.2 Google Gemini as a *live answer generator* (online, "Ask Gemini" mode)

**Key idea: Gemini writes the answer and our trained ML model judges it.** The LLM is never trained or fine-tuned by us; it's the system under test.

Flow (`app.py` → `backend.py`):

```
User question (+ optional passages)
   │
   ├─ no passages? → gemini_passages(): Gemini acts as a "search engine" and writes 4-6
   │                 factual sentences about the topic (simulated retrieval)
   ▼
gemini_answer(question, passages, style)
   │   style "strict" → "answer strictly based on the following passages"
   │   style "loose"  → "also add extra facts from your own knowledge" (to provoke hallucinations)
   ▼
build_context(question, passages) = "Question: ...\nPassages:\n..."   (same shape as RAGTruth QA prompts)
   ▼
detect(context, answer) → sentences flagged, risk score
```

**How the API call is managed (`_call_gemini`, `gemini_generate`):**

| Concern | How it's handled |
|---|---|
| **API access** | Plain REST call with Python's built-in `urllib` to `https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`. No SDK dependency. |
| **API key security** | Key is read from `.streamlit/secrets.toml` (git-ignored, never committed) or the `GEMINI_API_KEY` environment variable, or typed into a password field in the app (not saved). Sent in the `x-goog-api-key` header, not in the URL. Each user uses their own free key. |
| **Model choice** | `GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]`, selectable in the UI. |
| **Generation settings** | `temperature = 0.7`; prompts end with "Write plain text only" so markdown doesn't confuse sentence splitting. |
| **Retries** | HTTP 429 (rate limit), 500, 503, 504 (busy) and timeouts → retried twice with back-off (2 s, 4 s). |
| **Fallback** | If the chosen model keeps failing, it automatically switches to the other model, and the UI shows which model actually answered. |
| **Model not available (404)** | Skip straight to the next model. |
| **Bad key (400)** | Fail immediately with a readable message (retrying wouldn't help). |
| **No internet** | `URLError` → "Could not reach Gemini (check your internet)". |
| **Safety block / empty answer** | `promptFeedback.blockReason` or no text → clear error message. |
| **"Thinking" parts** | Parts marked `thought` are dropped, so only the final answer is checked. |
| **Timeout** | 60 s per request. |

**Testing without a key:** in `test_app.py` and the `backend.py` self-check, `_call_gemini` is replaced by a fake function, so tests run offline and never spend API quota.

---

## 5. System architecture

```
┌─────────────────────┐   download (urllib)   ┌──────────────────────────┐
│ RAGTruth on GitHub  │ ────────────────────► │ data/*.jsonl  (36 MB)    │
└─────────────────────┘                       └────────────┬─────────────┘
                                                           │ build_db()
                                                           ▼
                                   ┌───────────────────────────────────────────┐
                                   │ SQLite ragtruth.db                         │
                                   │ sources ─< responses · runs · predictions │
                                   └───────────┬───────────────────────────────┘
                     load_dataset() + sentence_table()
                                               ▼
                         10 features per sentence ──► train() 3 models ──► model.joblib
                                                                              │
 ┌─────────────── Streamlit app (app.py) ───────────────┐                     │
 │ Detector │ Dataset │ Model Performance │ History     │ ◄── detect() ───────┘
 └──────┬───────────────────────────────────────────────┘
        │ Ask Gemini mode
        ▼
 Google Gemini REST API (answer generator)
```

**Three layers:**
1. **Data layer:** SQLite database with 4 tables.
2. **ML / backend layer:** `backend.py` (features, training, inference, Gemini client).
3. **Presentation layer:** `app.py` (Streamlit web UI) + the Jupyter notebook (walkthrough).

### 5.1 Database schema

| Table | Columns | Purpose |
|---|---|---|
| `sources` | `source_id` **PK**, `task_type`, `context` | 2,965 retrieved contexts (QA 989, Summary 943, Data2txt 1,033) |
| `responses` | `id` **PK**, `source_id` **FK → sources**, `model`, `split`, `quality`, `response`, `labels` (JSON spans), `hallucinated` | 17,790 LLM answers with human labels |
| `runs` | `id`, `created_at`, `model_name`, `level`, `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `threshold`, `chosen` | One row per model × level per training run |
| `predictions` | `id`, `created_at`, `context`, `response`, `score`, `flagged_sentences`, `verdict` | Every detection made in the app (History tab) |

Relationship: one source has many (6) responses (**1 : N**). Foreign keys are enforced with `PRAGMA foreign_keys = ON`.

---

## 6. Code walkthrough (file by file)

### `backend.py` (the brain)

| Function | What it does |
|---|---|
| `download_data()` | Downloads the 2 jsonl files; writes to a `.part` file and renames only when complete, so an interrupted download is retried |
| `connect()` | Context manager: opens SQLite, enables foreign keys, commits on success, always closes |
| `build_db()` | Creates the 4 tables and bulk-loads the data with `executemany` (idempotent: skips if already loaded) |
| `query(sql, params)` | Runs SQL and returns a pandas DataFrame (parameterised, no SQL injection) |
| `load_dataset()` | Joins responses + sources, keeps `quality='good'`, carves the grouped dev split |
| `WORD` regex | Tokeniser: keeps `94.9`, `o'clock`, `commission-based` as one token, splits `tall.and` |
| `norm(w)` | Lower-case + simple plural stemming |
| `split_sentences(text)` | Splits on `. ! ?` (optionally followed by a closing quote/bracket) or newlines, returns character offsets; doesn't split after abbreviations (`St.`, `Dr.`, `e.g.`, `No. 5`) or initials (`J. K. Rowling`) |
| `context_index(context)` | Pre-computes the sets of context words, bigrams and trigrams (fast O(1) lookups) |
| `novel_words()` | The words shown underlined in the app |
| `sentence_features()` | Computes the 10 features |
| `sentence_table(df)` | One row per sentence with features + label |
| `metrics()`, `best_threshold()` | Evaluation helpers |
| `train(sent)` | Trains / tunes / selects / saves / logs (see §3.6) |
| `load_model()` | Loads `model.joblib`, cached and auto-reloaded when the file changes (mtime check) |
| `detect(context, response)` | Inference: per-sentence probability, flagged flag, unsupported words, risk score = max probability, verdict, logs to `predictions` |
| `gemini_*` functions | LLM client (see §4.2) |
| `if __name__ == "__main__"` | Self-check with `assert`s (sentence splitting, features, retry/fallback with a fake network) |

### `app.py` (Streamlit UI)

- **Detector tab:** 3 input modes:
  - *Write my own*: 4 demo scenarios (Invented fact, Wrong numbers, Fully grounded, News: made-up claim) + Clear
  - *Ask Gemini*
  - *RAGTruth test example*: model prediction vs human gold labels
- **Dataset tab:** live SQL stats and bar charts + table browser.
- **Model Performance tab:** latest `runs`, best values highlighted, bar chart, sentence/response switch.
- **History tab:** every logged detection.
- **Safety:** all user text is HTML-escaped before rendering (`html.escape`), so typing `<script>` can't run code; underlining matches whole tokens only.
- **State:** `st.session_state` keeps the typed text when switching modes; results are only shown if they belong to the inputs currently on screen.

### `RAGTruth_Hallucination_Detector.ipynb`

Run-All walkthrough: install → API key (saved once to `secrets.toml`) → download + SQLite → EDA charts → split → features → train + compare → feature importance → try the detector → launch the app (finds a free port from 8501, opens the browser).

### `test_app.py`

7 end-to-end tests with Streamlit's `AppTest` (headless): write-your-own, all 4 scenarios + Clear + empty-input warning, text surviving mode switches, Gemini without passages (strict = grounded, loose = flagged), bad key + empty question, RAGTruth examples + gold labels, HTML escaping. Uses a temporary DB copy and a fake Gemini.

### `docs/make_figures.py`

Regenerates all charts used in the report and the presentation from the database and the trained model.

---

## 7. Bug-fix pass (done before final submission)

All 7 UI tests and the backend self-check pass. During the final review we found and fixed:

| # | Bug | Effect | Fix |
|---|---|---|---|
| 1 | Stopwords were checked *after* plural-stemming, so "this" → "thi" and "does" → "doe" were not treated as stopwords | "This" and "does" were underlined as unsupported words and inflated `novel_words` | Stopword set now also contains the normalised forms |
| 2 | A sentence ending in a quote or bracket (`He said "Hi." Then…`) was not split | Two sentences merged, wrong labels and features | Sentence regex accepts `.`/`!`/`?` followed by a closing quote/bracket |
| 3 | "The answer is no. It is fine." treated "no." as the abbreviation "No." | Sentences merged | "No." is an abbreviation only when a digit follows (`No. 5`) |
| 4 | If every Gemini model returned 404, the error said "models are busy" | Misleading message | Message now says the model isn't available to this key |
| 5 | App crashed if `ragtruth.db` was deleted but `model.joblib` existed | Stack trace instead of a hint | App checks both files and asks to run the notebook |

After fixing, the model was retrained: response-level F1 went from 0.655 to **0.664** and response accuracy from 0.693 to 0.717.

---

## 8. Project timeline (10 Oct 2026 → 20 Nov 2026)

| Week | Dates | Work | Lead |
|---|---|---|---|
| 1 | 10 – 16 Oct | Topic finalised (hallucinations in RAG). Literature survey: RAGTruth paper, HaluEval, SelfCheckGPT. Dataset chosen: RAGTruth (MIT licence). GitHub repo + Python environment set up. | Both |
| 2 | 17 – 23 Oct | Data download script, SQLite schema design (4 tables, PK/FK), bulk loading, first SQL queries. EDA: hallucination rate by LLM / task / span type. | Kavay |
| 3 | 24 – 30 Oct | Sentence splitter with character offsets, span-overlap labelling, 10 lexical features, grouped dev split (no leakage). Feature sanity check (grounded vs hallucinated averages). | Gurliv |
| 4 | 31 Oct – 6 Nov | Trained and compared Logistic Regression, Random Forest, Gradient Boosting. Threshold tuning on dev, sentence- and response-level evaluation, feature importance. Gradient Boosting selected. | Kavay |
| 5 | 7 – 13 Nov | Streamlit app: Detector (demo scenarios, RAGTruth examples with gold labels), Dataset, Model Performance, History tabs. Gemini integration (simulated retrieval, strict/loose answers, retry + fallback, key handling). | Gurliv |
| 6 | 14 – 20 Nov | Testing: backend self-check + 7 end-to-end UI tests. Final bug-fix pass and retraining. Run-All notebook polished. Report, presentation, final evaluation on **20 Nov**. | Both |

*(Adjust the "Lead" column to match who actually did what. Both of you should be able to explain every part.)*

**Milestones:** 23 Oct database ready · 30 Oct features ready · 6 Nov model selected · 13 Nov app + Gemini working · 18 Nov all tests passing · 20 Nov final submission.

---

## 9. Real-life use cases

1. **Customer-support chatbots:** before an answer about a refund policy goes to a customer, flag sentences not backed by the policy document.
2. **Healthcare / pharma assistants:** check that a dosage or side-effect answer only contains facts from the approved drug leaflet.
3. **Legal and compliance:** verify that a contract summary doesn't invent clauses, dates or amounts.
4. **News summarisation:** catch invented quotes, names and numbers in auto-generated summaries (our "News: made-up claim" demo).
5. **Education / college help desks:** a chatbot answering from the academic handbook shouldn't invent deadlines or fee amounts.
6. **Monitoring LLM quality:** log the risk score of every production answer (like our History tab) to compare LLMs, prompts or retrievers over time.

---

## 10. Demos to show the teacher (in this order, ~5 minutes)

| # | Where | What to do | What happens | What it proves |
|---|---|---|---|---|
| 1 | Notebook | Scroll through EDA, feature table, model comparison | Charts and tables | The data science pipeline |
| 2 | App → Write my own → **Invented fact** → Detect | Context: Eiffel Tower in Paris, 1889, 330 m. Answer adds "designed by Leonardo da Vinci" | 2nd sentence red (p = 0.87), "designed Leonardo da Vinci" underlined | Detects invented entities |
| 3 | **Wrong numbers** | Answer says 1925 and 500 metres | Sentence flagged, "1925" and "500" underlined | Detects invented numbers |
| 4 | **Fully grounded** | Answer copies the facts | "Looks grounded" (0.29) | No false alarm on correct text |
| 5 | **News: made-up claim** | Apple revenue + invented "move HQ to Austin, Texas" | 2nd sentence flagged (0.89) | Works on news-style text |
| 6 | Type your own (Indian example) | Context: *"Thapar Institute of Engineering and Technology is located in Patiala, Punjab. It was founded in 1956."* Answer: *"Thapar Institute is located in Patiala, Punjab and was founded in 1956. It was founded by Mahatma Gandhi and has 50,000 students."* | 2nd sentence flagged (0.82); "Mahatma Gandhi", "50", "000", "students" underlined | Works on any new text, not just the dataset |
| 7 | **Ask Gemini** | Question: "Tell me about the Taj Mahal", passages empty, *Strictly from passages* → then *Add outside facts* | Strict: grounded; Loose: extra facts flagged | Live LLM + our model as a judge |
| 8 | **RAGTruth test example** (QA) | Detect | Model's red sentences vs human yellow labels | Real benchmark data, model agrees with humans |
| 9 | Empty boxes → Detect | | "Fill in both the context and the response." | Input validation / error handling |
| 10 | Model Performance & History tabs | | Metrics from `runs`, detections from `predictions` | Database integration |
| 11 | (Honesty demo) Context: *"Virat Kohli scored 82 runs and Rohit Sharma scored 45 runs."* Answer swaps the numbers | | "Looks grounded" | Shows the known limitation: lexical features can't see swapped facts → future work: NLI / transformers |

Let the teacher type their own example too: a paragraph of facts as context, and an answer that repeats some and adds one new name or number.

---

## 11. How to run

```bash
pip install jupyter
jupyter notebook RAGTruth_Hallucination_Detector.ipynb   # then Run → Run All
```

Or manually: `pip install -r requirements.txt`, run the notebook cells once (creates `ragtruth.db` and `model.joblib`), then `streamlit run app.py`.
Tests: `python backend.py` (self-check) and `python test_app.py` (7 UI tests).
Figures: `python docs/make_figures.py`.

**Tech stack:** Python 3.12 · pandas · NumPy · scikit-learn · SQLite (`sqlite3`) · Streamlit · Matplotlib · joblib · Jupyter · Google Gemini REST API · Git/GitHub.

---

## 12. Likely viva questions, with answers

1. **What is a hallucination?** Text in an LLM answer that isn't supported by (or contradicts) the source context.
2. **What is RAG?** Retrieval-Augmented Generation: retrieve relevant documents, then let the LLM answer using them. It reduces but doesn't eliminate hallucination.
3. **Why RAGTruth?** It's a large (17,790 responses), human-annotated, *span-level* hallucination dataset for RAG, covering 3 tasks and 6 LLMs, with an MIT licence.
4. **Supervised or unsupervised?** Supervised binary classification. The labels come from the human-annotated spans.
5. **Why sentence level?** The labels are spans; sentences are a natural unit for flagging and give an explainable output. Response level is derived by taking the max.
6. **Why `class_weight="balanced"`?** Only ~11% of sentences are positive; without weighting the model would just predict "grounded".
7. **Why tune the threshold?** With imbalanced data, 0.5 isn't optimal; we pick the threshold (0.66) that maximises F1 on dev.
8. **Why a separate dev set if you have test?** Choosing the model or threshold on test would leak test information and give optimistic numbers. Test is touched only for the final report.
9. **Why group the dev split by `source_id`?** All 6 answers to a passage share the context; splitting them across sets would leak.
10. **Precision vs recall?** Precision = of the flagged, how many were truly hallucinated. Recall = of the truly hallucinated, how many we flagged. For safety use cases, recall matters more.
11. **What is ROC-AUC?** The probability that a random positive is ranked above a random negative. It's threshold-independent; 0.5 is random and 1.0 is perfect. Ours is 0.80.
12. **Difference between Random Forest and Gradient Boosting?** RF trains trees independently on bootstrap samples and averages them (reduces variance). GB trains trees sequentially, each correcting the previous errors (reduces bias).
13. **What does "Hist" in HistGradientBoosting mean?** Features are binned into ≤ 255 histogram bins, which makes split-finding very fast on large data (like LightGBM).
14. **What is the risk score?** The maximum sentence probability in the response.
15. **Why SQLite?** Zero setup, single file, real SQL with PK/FK, good enough for 17k rows, and it's built into Python.
16. **How do you prevent SQL injection?** Parameterised queries (`?` placeholders) everywhere.
17. **How is the API key kept safe?** It lives in a git-ignored `secrets.toml` or an environment variable, is sent in a header, and is never committed or logged.
18. **What if Gemini is down?** Retries with back-off, then automatic fallback to the other model, then a readable error.
19. **Does your model use Gemini to detect?** No. Gemini only generates the answer. Detection is 100% our trained scikit-learn model, running locally.
20. **Biggest limitation?** Lexical features: they can't detect contradictions that reuse context words. A transformer/NLI model would fix that.
21. **How did you test?** Backend self-check asserts + 7 Streamlit end-to-end tests with a fake Gemini and a temporary DB copy.
22. **What is early stopping?** Training stops when the loss on an internal validation slice stops improving for 10 rounds; our model stopped at 148 of the allowed 300 trees, which prevents overfitting.

---

## 13. References

1. Niu, C. et al. (2024). *RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models.* ACL 2024. https://github.com/ParticleMedia/RAGTruth
2. Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020.
3. Ji, Z. et al. (2023). *Survey of Hallucination in Natural Language Generation.* ACM Computing Surveys.
4. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12. https://scikit-learn.org
5. Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine.* Annals of Statistics.
6. Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 2017 (the histogram idea).
7. Breiman, L. (2001). *Random Forests.* Machine Learning 45.
8. Streamlit documentation: https://docs.streamlit.io
9. Google Gemini API documentation: https://ai.google.dev/gemini-api/docs
10. SQLite documentation: https://www.sqlite.org/docs.html
11. pandas documentation: https://pandas.pydata.org/docs
