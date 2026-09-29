"""Export lecture notes (transcript + summary + Q&A + slides) as Markdown, PDF, or Anki CSV."""

from __future__ import annotations
import csv
import io
import re
from typing import List, Dict, Optional
from datetime import datetime


def build_markdown(
    title: str,
    bullets: List[str],
    key_terms: List[str],
    transcript: str,
    qa_history: List[Dict],
    slides_text: Optional[str] = None,
) -> str:
    lines = [f"# {title}", "", f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}_", ""]
    lines += ["## Summary", ""]
    lines += [f"- {b}" for b in bullets] or ["- (no summary yet)"]
    lines += ["", "## Key Terms", ""]
    lines += [f"- {t}" for t in key_terms] or ["- (none)"]

    if slides_text and slides_text.strip():
        lines += ["", "## From Slides (OCR Capture)", "", slides_text.strip(), ""]

    if qa_history:
        lines += ["", "## Q&A History", ""]
        for qa in qa_history:
            lines += [f"**Q: {qa['question']}**", "", f"A: {qa['answer']}", ""]

    lines += ["", "## From Transcript", "", transcript if transcript.strip() else "(no transcript)"]
    return "\n".join(lines)


def markdown_to_pdf(markdown_text: str, out_path: str) -> str:
    """Very lightweight text->PDF using fpdf2.
    This version strips any characters that cannot be encoded with the default Latin‑1 font,
    preventing UnicodeEncodeError crashes when markdown contains exotic symbols (e.g., en‑dash).
    """
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)

    for raw_line in markdown_text.splitlines():
        # Remove characters not supported by Latin‑1 encoding used by the default font
        safe_line = raw_line.encode("latin-1", errors="ignore").decode("latin-1")
        line = safe_line.strip()
        if not line:
            pdf.ln(3)
            continue
        w = pdf.epw
        if line.startswith("# "):
            pdf.set_font("Helvetica", "B", 16)
            pdf.multi_cell(w, 10, line[2:])
            pdf.set_font("Helvetica", size=12)
        elif line.startswith("## "):
            pdf.set_font("Helvetica", "B", 13)
            pdf.multi_cell(w, 8, line[3:])
            pdf.set_font("Helvetica", size=12)
        elif line.startswith("- "):
            pdf.multi_cell(w, 7, "- " + line[2:])
        else:
            pdf.multi_cell(w, 7, line)

    pdf.output(out_path)
    return out_path


def export_flashcards_csv(
    bullets: List[str],
    key_terms: List[str],
    qa_history: List[Dict],
    out_path: str = "studypilot_flashcards.csv",
    llm=None
) -> str:
    """Generate an Anki-importable CSV file with Front (Question) and Back (Answer) columns."""
    cards: List[tuple[str, str]] = []

    # Include existing Q&A history items first
    for qa in qa_history:
        q = qa.get("question", "").strip()
        a = qa.get("answer", "").strip()
        if q and a:
            cards.append((q, a))

    # Generate flashcards from key terms & bullets
    if llm and hasattr(llm, "generate_text") and (bullets or key_terms):
        context = "Key Points:\n" + "\n".join(bullets) + "\nKey Terms:\n" + ", ".join(key_terms)
        prompt = (
            "Create 5 flashcards from these study notes for Anki spaced repetition. "
            "Output each flashcard on a new line in format: Front Question | Back Answer\n\n"
            f"Notes:\n{context}\n\nFlashcards:"
        )
        try:
            raw_cards = llm.generate_text(prompt)
            for line in raw_cards.splitlines():
                if "|" in line:
                    parts = line.split("|", 1)
                    q_text = parts[0].strip("- ").strip()
                    a_text = parts[1].strip()
                    if q_text and a_text:
                        cards.append((q_text, a_text))
        except Exception:
            pass

    # Fallback/extractive flashcard generation if list is small
    if len(cards) < 3:
        for term in key_terms:
            cards.append((f"What is the definition/significance of {term}?", f"{term} is a key concept covered in the lecture."))
        for idx, bullet in enumerate(bullets, 1):
            cards.append((f"Key Concept #{idx}", bullet))

    # Write RFC 4180 compliant CSV (compatible with Anki import)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Front", "Back"])
        for front, back in cards:
            writer.writerow([front, back])

    return out_path
