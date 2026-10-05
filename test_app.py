"""End-to-end tests for the Streamlit app (run the notebook first so the DB and model exist).

    python test_app.py

Gemini is faked, so no API key or internet is needed, and a temporary copy of the database is
used so the History tab is not filled with test rows.
"""
import shutil
import tempfile
import urllib.error
from pathlib import Path

from streamlit.testing.v1 import AppTest

import backend as b

tmp = Path(tempfile.mkdtemp()) / "test.db"
shutil.copy(b.DB_PATH, tmp)
b.DB_PATH = tmp  # app.py imports this same module, so it writes to the copy

calls = []


def fake_gemini(prompt, api_key, model):
    calls.append(prompt)
    if api_key == "bad":
        raise urllib.error.HTTPError("u", 400, "Bad Request", {}, None)
    if "search engine" in prompt:  # simulated retrieval
        return "The Eiffel Tower is in Paris. It was completed in 1889 and is 330 metres tall."
    if "own knowledge" in prompt:  # "Add outside facts"
        return "The Eiffel Tower is in Paris. It was designed by Gustave Eiffel and painted bronze in 1968."
    return "The Eiffel Tower is in Paris. It was completed in 1889."


b._call_gemini = fake_gemini


def app():
    at = AppTest.from_file("app.py", default_timeout=120)
    at.secrets["GEMINI_API_KEY"] = "test-key"
    return at.run()


def metrics(at):
    return {m.label: m.value for m in at.metric}


def button(at, text):
    return next(x for x in at.button if text in x.label)


def test_write_my_own():
    at = app()
    at.text_area(key="ctx").input("My dog Bruno is 4 years old and loves to swim in the lake.")
    at.text_area(key="resp").input("Bruno is 4 years old. He won a gold medal at the 2020 Olympics.")
    button(at, "Detect").click().run()
    assert metrics(at)["Verdict"] == "Likely hallucinated" and metrics(at)["Flagged sentences"] == "1 / 2"


def test_scenarios_and_clear():
    at = app()
    expected = {"Invented fact": "Likely hallucinated", "Wrong numbers": "Likely hallucinated",
                "Fully grounded": "Looks grounded", "News: made-up claim": "Likely hallucinated"}
    for name, verdict in expected.items():
        at.segmented_control(key="scenario").set_value(name).run()
        button(at, "Detect").click().run()
        assert metrics(at)["Verdict"] == verdict, name
    button(at, "Clear").click().run()
    assert at.text_area(key="ctx").value == "" and "Verdict" not in metrics(at)
    button(at, "Detect").click().run()
    assert at.warning[0].value == "Fill in both the context and the response."


def test_text_survives_mode_switch():
    at = app()
    at.text_area(key="ctx").input("my context")
    at.text_area(key="resp").input("my answer")
    button(at, "Detect").click().run()
    for mode in ("Ask Gemini", "RAGTruth test example", "Write my own"):
        at.segmented_control(key="mode").set_value(mode).run()
    assert at.text_area(key="ctx").value == "my context" and "Verdict" in metrics(at)


def test_gemini_without_passages():
    at = app()
    at.segmented_control(key="mode").set_value("Ask Gemini").run()
    assert not any(t.label == "Gemini API key" for t in at.text_input)  # key comes from secrets
    at.text_input(key="g_q").input("Tell me about the Eiffel Tower")
    button(at, "Generate").click().run()
    assert metrics(at)["Verdict"] == "Looks grounded"
    assert at.session_state["g_answer"]["generated"]  # Gemini wrote the passages first
    at.segmented_control[1].set_value("Add outside facts")
    button(at, "Generate").click().run()
    assert metrics(at)["Verdict"] == "Likely hallucinated"


def test_gemini_bad_key_and_empty_question():
    at = app()
    at.segmented_control(key="mode").set_value("Ask Gemini").run()
    button(at, "Generate").click().run()
    assert at.warning[0].value == "Type a question first."
    at.secrets["GEMINI_API_KEY"] = "bad"
    at.text_input(key="g_q").input("anything")
    button(at, "Generate").click().run()
    assert "400" in at.error[0].value


def test_ragtruth_example_and_other_tabs():
    at = app()
    at.segmented_control(key="mode").set_value("RAGTruth test example").run()
    button(at, "Detect").click().run()
    assert "Verdict" in metrics(at) and not at.exception
    assert any("Gold annotation" in h.value for h in at.subheader)


def test_highlight_is_escaped_and_whole_words():
    at = app()
    at.text_area(key="ctx").input("We like modern painting.")
    at.text_area(key="resp").input("We start the art <script>x</script> show & more.")
    button(at, "Detect").click().run()
    html_out = next(m.value for m in at.markdown if "line-height:1.9" in m.value)
    assert "<script>" not in html_out and "&lt;" in html_out  # user text is escaped, never run as HTML
    assert "<u><b>art</b></u>" in html_out and "st<u>" not in html_out  # "art" not underlined inside "start"


if __name__ == "__main__":
    tests = [f for name, f in list(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(tests)} tests passed")
