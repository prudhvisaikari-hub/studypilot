# Full pipeline script for StudyPilot
# This script runs the complete end‑to‑end workflow using the built‑in back‑ends.
# It:
#   1. Loads the bundled sample lecture (plain‑text) via MockASR.
#   2. Generates an extractive summary (bullets + key terms).
#   3. Performs a sample RAG question.
#   4. Exports Markdown, PDF and Anki‑CSV flashcards.
#   5. Prints paths of the generated artefacts.

import os
from backend.asr import MockASR
from backend.summarizer import get_summarizer
from backend.rag import TfidfRAG, chunk_text
from backend.export import build_markdown, markdown_to_pdf, export_flashcards_csv


def run():
    # --------------------------------------------------
    # 1️⃣ Load sample lecture (plain‑text) via MockASR
    # --------------------------------------------------
    sample_path = os.path.join(os.path.dirname(__file__), "sample_data", "sample_lecture.txt")
    asr = MockASR()
    segments = asr.transcribe(sample_path)
    transcript = " ".join(s.text for s in segments)

    # --------------------------------------------------
    # 2️⃣ Summarize (extractive – zero‑download mode)
    # --------------------------------------------------
    summarizer = get_summarizer("extractive")
    summary = summarizer.summarize(transcript)

    # --------------------------------------------------
    # 3️⃣ Retrieval‑augmented Q&A (sample question)
    # --------------------------------------------------
    rag = TfidfRAG(chunk_text(transcript))
    qa_res = rag.answer("What is Newton's second law?", top_k=3)

    # --------------------------------------------------
    # 4️⃣ Export artefacts (Markdown, PDF, Anki CSV)
    # --------------------------------------------------
    md = build_markdown(
        title="StudyPilot Full‑Run Demo",
        bullets=summary["bullets"],
        key_terms=summary["key_terms"],
        transcript=transcript,
        qa_history=[{"question": "What is Newton's second law?", "answer": qa_res["answer"]}],
        slides_text=None,
    )
    out_md = "full_run_notes.md"
    out_pdf = "full_run_notes.pdf"
    out_csv = "full_run_flashcards.csv"
    # Write markdown file
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    # Convert to PDF
    markdown_to_pdf(md, out_pdf)
    # Generate flashcards CSV (extractive fallback if LLM not present)
    export_flashcards_csv(summary["bullets"], summary["key_terms"], [], out_path=out_csv)

    print("[OK] Generated artefacts:")
    print("   -", out_md)
    print("   -", out_pdf)
    print("   -", out_csv)


if __name__ == "__main__":
    run()
