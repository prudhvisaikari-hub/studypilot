"""
StudyPilot - On-Device Lecture Summarizer & Doubt Solver
Entry point: `python app.py`

Runs by default with real Whisper ASR and real LLM summarization/RAG Q&A,
with automatic graceful fallback to mock mode and visible UI warning banners if
weights/dependencies are missing.
"""

from __future__ import annotations
import os
import logging
import gradio as gr

from backend.asr import get_asr_backend, segments_to_transcript_json, ASRInitError, ASRTranscriptionError, MockASR
from backend.summarizer import get_summarizer, SummarizerInitError, SummarizerError, ExtractiveSummarizer
from backend.rag import TfidfRAG, chunk_text
from backend.ocr import extract_slides_text
from backend.export import build_markdown, markdown_to_pdf, export_flashcards_csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("studypilot")

# ---- Configuration & Initial Fallback Handling --------------------------
DEFAULT_ASR = os.environ.get("STUDYPILOT_ASR", "whisper")
DEFAULT_SUMMARIZER = os.environ.get("STUDYPILOT_SUMMARIZER", "llm")
SAMPLE_LECTURE_PATH = os.path.join(os.path.dirname(__file__), "sample_data", "sample_lecture.txt")

ASR_WARNING = ""
SUMMARIZER_WARNING = ""
ACTUAL_ASR_BACKEND = DEFAULT_ASR
ACTUAL_SUMMARIZER_BACKEND = DEFAULT_SUMMARIZER

# Attempt to load ASR backend with automatic fallback
try:
    asr = get_asr_backend(DEFAULT_ASR)
    logger.info(f"Initialized requested ASR backend: {DEFAULT_ASR}")
except Exception as e:
    logger.warning(f"Failed to load requested ASR backend '{DEFAULT_ASR}'. Falling back to 'mock'. Error: {e}")
    asr = get_asr_backend("mock")
    ACTUAL_ASR_BACKEND = "mock (fallback)"
    ASR_WARNING = f"⚠️ Whisper ASR failed to load ('{DEFAULT_ASR}'): {e}. Falling back to mock/sample mode."

# Attempt to load Summarizer backend with automatic fallback
try:
    summarizer = get_summarizer(DEFAULT_SUMMARIZER)
    logger.info(f"Initialized requested Summarizer backend: {DEFAULT_SUMMARIZER}")
except Exception as e:
    logger.warning(f"Failed to load requested Summarizer backend '{DEFAULT_SUMMARIZER}'. Falling back to 'extractive'. Error: {e}")
    summarizer = get_summarizer("extractive")
    ACTUAL_SUMMARIZER_BACKEND = "extractive (fallback)"
    SUMMARIZER_WARNING = f"⚠️ LLM Summarizer failed to load ('{DEFAULT_SUMMARIZER}'): {e}. Falling back to Extractive mode."
# -------------------------------------------------------------------------

# In-memory session state
STATE = {
    "transcript": "",
    "bullets": [],
    "key_terms": [],
    "rag": None,
    "qa_history": [],
    "slides_text": "",
}


def transcribe(audio_file, use_sample, enable_diarization):
    path = SAMPLE_LECTURE_PATH if use_sample else audio_file
    if not path:
        return "Please upload/record an audio file, or check 'Use bundled sample lecture'.", ""

    try:
        segments = asr.transcribe(path, enable_diarization=enable_diarization)
    except FileNotFoundError:
        return f"File not found: {path}", ""
    except Exception as e:
        logger.error(f"ASR transcription error: {e}")
        return f"Transcription error: {str(e)}", ""

    if not segments:
        return "(Silent or empty audio provided. No spoken text detected.)", "[]"

    if enable_diarization:
        transcript_text = "\n".join(f"[{s.speaker}] ({s.start}s - {s.end}s): {s.text}" for s in segments)
    else:
        transcript_text = " ".join(s.text for s in segments)

    STATE["transcript"] = transcript_text
    # Include slide text into RAG index if present
    full_rag_context = transcript_text
    if STATE["slides_text"]:
        full_rag_context += f"\n\nSlides:\n{STATE['slides_text']}"

    STATE["rag"] = TfidfRAG(chunk_text(full_rag_context))
    STATE["qa_history"] = []
    return transcript_text, segments_to_transcript_json(segments)


def process_slides(image_files):
    if not image_files:
        return "No slide image files uploaded."
    paths = [f.name if hasattr(f, "name") else f for f in image_files]
    extracted = extract_slides_text(paths)
    STATE["slides_text"] = extracted

    # Re-index RAG if transcript already exists
    if STATE["transcript"]:
        full_context = f"{STATE['transcript']}\n\nSlides:\n{extracted}"
        STATE["rag"] = TfidfRAG(chunk_text(full_context))

    return extracted if extracted else "No text could be extracted from uploaded slides."


def summarize():
    if not STATE["transcript"]:
        return "Transcribe a lecture first.", ""
    try:
        result = summarizer.summarize(STATE["transcript"])
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        return f"Summarization failed: {str(e)}", ""

    STATE["bullets"] = result["bullets"]
    STATE["key_terms"] = result["key_terms"]
    bullets_md = "\n".join(f"- {b}" for b in result["bullets"])
    terms_md = ", ".join(result["key_terms"])
    return bullets_md, terms_md


def ask(question):
    if not STATE["rag"]:
        return "Transcribe a lecture first."
    if not question or not question.strip():
        return "Type a question first."

    llm_arg = summarizer if "llm" in ACTUAL_SUMMARIZER_BACKEND.lower() else None
    result = STATE["rag"].answer(question, top_k=3, llm=llm_arg)

    answer_text = result["answer"]
    if result.get("warning"):
        answer_text += f"\n\n⚠️ [{result['warning']}]"

    STATE["qa_history"].append({"question": question, "answer": answer_text})
    return answer_text


def export_markdown():
    md = build_markdown(
        title="StudyPilot Notes",
        bullets=STATE["bullets"],
        key_terms=STATE["key_terms"],
        transcript=STATE["transcript"],
        qa_history=STATE["qa_history"],
        slides_text=STATE["slides_text"],
    )
    out_path = "studypilot_notes.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    return out_path


def export_pdf():
    md = build_markdown(
        title="StudyPilot Notes",
        bullets=STATE["bullets"],
        key_terms=STATE["key_terms"],
        transcript=STATE["transcript"],
        qa_history=STATE["qa_history"],
        slides_text=STATE["slides_text"],
    )
    out_path = "studypilot_notes.pdf"
    markdown_to_pdf(md, out_path)
    return out_path


def export_flashcards():
    llm_arg = summarizer if "llm" in ACTUAL_SUMMARIZER_BACKEND.lower() else None
    out_path = export_flashcards_csv(
        bullets=STATE["bullets"],
        key_terms=STATE["key_terms"],
        qa_history=STATE["qa_history"],
        llm=llm_arg
    )
    return out_path


# ---- Gradio UI Construction --------------------------------------------
with gr.Blocks(title="StudyPilot") as demo:
    gr.Markdown("# 🎓 StudyPilot — On-Device Lecture Summarizer & Doubt Solver")
    gr.Markdown(
        f"_ASR backend: **{ACTUAL_ASR_BACKEND}** · Summarizer backend: **{ACTUAL_SUMMARIZER_BACKEND}** · "
        "All processing runs locally, no cloud calls._"
    )

    # Display Warning Banner if any backend dropped back to mock/extractive
    if ASR_WARNING or SUMMARIZER_WARNING:
        warning_msg = "### Warning / Fallback Notice:\n"
        if ASR_WARNING:
            warning_msg += f"- {ASR_WARNING}\n"
        if SUMMARIZER_WARNING:
            warning_msg += f"- {SUMMARIZER_WARNING}\n"
        gr.Markdown(warning_msg)

    with gr.Tab("1. Record / Upload & Diarization"):
        audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Lecture audio")
        use_sample = gr.Checkbox(label="Use bundled sample lecture (no audio needed, for demo/testing)", value=False)
        enable_diarization = gr.Checkbox(
            label="Enable Speaker Diarization (Approximation: pause/turn-taking heuristic)",
            value=False
        )
        transcribe_btn = gr.Button("Transcribe Lecture", variant="primary")
        transcript_out = gr.Textbox(label="Transcript Output", lines=10)
        raw_json_out = gr.Textbox(label="Raw Segments (Debug JSON)", lines=4, visible=True)
        transcribe_btn.click(
            transcribe,
            [audio_in, use_sample, enable_diarization],
            [transcript_out, raw_json_out]
        )

    with gr.Tab("2. Summary & Slide OCR"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Lecture Summarization")
                summarize_btn = gr.Button("Generate Summary", variant="primary")
                bullets_out = gr.Markdown(label="Key Takeaways")
                terms_out = gr.Textbox(label="Key Terms / Glossary")
                summarize_btn.click(summarize, [], [bullets_out, terms_out])

            with gr.Column():
                gr.Markdown("### Slide / Screen OCR Capture")
                slide_in = gr.File(label="Upload Lecture Slide Images", file_count="multiple", file_types=["image"])
                ocr_btn = gr.Button("OCR Extract Slides Text")
                ocr_out = gr.Textbox(label="Extracted Slide Text (From Slides)", lines=8)
                ocr_btn.click(process_slides, [slide_in], [ocr_out])

    with gr.Tab("3. Ask a Question (RAG Q&A)"):
        question_in = gr.Textbox(label="Your question about the lecture transcript or slides")
        ask_btn = gr.Button("Ask Question", variant="primary")
        answer_out = gr.Textbox(label="Answer & Grounding Status", lines=6)
        ask_btn.click(ask, [question_in], [answer_out])

    with gr.Tab("4. Export Notes & Flashcards"):
        gr.Markdown("Export your comprehensive notes (Summary + Key Terms + Slides + Q&A + Transcript).")
        with gr.Row():
            md_btn = gr.Button("Export Markdown")
            pdf_btn = gr.Button("Export PDF")
            flashcard_btn = gr.Button("Export Flashcards (Anki CSV)", variant="primary")
        md_file = gr.File(label="Markdown File")
        pdf_file = gr.File(label="PDF File")
        csv_file = gr.File(label="Anki Flashcards CSV File")

        md_btn.click(export_markdown, [], md_file)
        pdf_btn.click(export_pdf, [], pdf_file)
        flashcard_btn.click(export_flashcards, [], csv_file)

if __name__ == "__main__":
    demo.launch()
