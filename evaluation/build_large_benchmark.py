import json
import shutil
import ssl
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data"
OUT_JSON = ROOT / "evaluation" / "question.json"

SQUAD_V2_URL = (
    "https://rajpurkar."
    "github.io/SQuAD-explorer/"
    "dataset/dev-v2.0.json"
)

NUM_ARTICLES = 80         # Creates 80 separate files in data/
NUM_ANSWERABLE = 85       # 85 factual questions
NUM_UNANSWERABLE = 15     # 15 out-of-scope guardrail questions


def main():
    # Clean and recreate data/ directory
    if DOCS_DIR.exists():
        shutil.rmtree(DOCS_DIR)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading official Stanford SQuAD v2 dataset directly...")
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(
        SQUAD_V2_URL,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, context=ctx) as response:
        payload = json.loads(response.read().decode("utf-8"))

    unique_contexts = {}
    answerable_qa = []
    unanswerable_qa = []

    for article in payload["data"]:
        raw_title = article.get("title", "article")
        clean_title = "".join(c if c.isalnum() else "_" for c in raw_title)

        for paragraph in article["paragraphs"]:
            context = paragraph["context"].strip()

            # Save unique contexts as documents in data/
            if context not in unique_contexts and len(unique_contexts) < NUM_ARTICLES:
                doc_id = len(unique_contexts) + 1
                filename = f"doc_{doc_id:03d}_{clean_title}.md"
                (DOCS_DIR / filename).write_text(
                    f"# {raw_title}\n\n{context}\n", encoding="utf-8"
                )
                unique_contexts[context] = filename

            # Only collect questions for documents we actually saved in data/
            if context in unique_contexts:
                added_ans = 0
                added_unans = 0

                for qa in paragraph["qas"]:
                    is_impossible = qa.get("is_impossible", False)
                    answers = qa.get("answers", [])

                    if (
                        not is_impossible
                        and answers
                        and len(answerable_qa) < NUM_ANSWERABLE
                        and added_ans < 2
                    ):
                        answerable_qa.append({
                            "question": qa["question"].strip(),
                            "expected": answers[0]["text"].strip(),
                        })
                        added_ans += 1
                    elif (
                        is_impossible
                        and len(unanswerable_qa) < NUM_UNANSWERABLE
                        and added_unans < 1
                    ):
                        unanswerable_qa.append({
                            "question": qa["question"].strip(),
                            "expected": "Not in the document. This information is not provided in the context.",
                        })
                        added_unans += 1

            if (
                len(unique_contexts) >= NUM_ARTICLES
                and len(answerable_qa) >= NUM_ANSWERABLE
                and len(unanswerable_qa) >= NUM_UNANSWERABLE
            ):
                break

        if (
            len(unique_contexts) >= NUM_ARTICLES
            and len(answerable_qa) >= NUM_ANSWERABLE
            and len(unanswerable_qa) >= NUM_UNANSWERABLE
        ):
            break

    eval_set = answerable_qa + unanswerable_qa
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(eval_set, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Done! Created {len(unique_contexts)} files in {DOCS_DIR}")
    print(f"Saved {len(eval_set)} benchmark questions to {OUT_JSON}")


if __name__ == "__main__":
    main()
