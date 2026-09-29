# StudyPilot — On-Device Lecture Summarizer & Doubt Solver

Built for the Snapdragon AI PC challenge. StudyPilot records/uploads a lecture audio,
transcribes it with Whisper ASR, summarizes it into structured takeaways + key terms using an LLM,
extracts text from uploaded lecture slides with OCR, generates Anki-importable flashcard decks, and answers doubts via retrieval-augmented Q&A (RAG) — all designed to run locally on a Snapdragon-powered AI PC with zero cloud calls.

## Feature Implementation Status

| Feature Layer | Offline / Mock Mode | Real CPU Baseline (Ships & Tested) | Snapdragon NPU Target Path |
|---------------|----------------------|-----------------------------------|----------------------------|
| **ASR Transcription** | `MockASR` (reads text transcript, 0 downloads) | `WhisperASR` (`faster-whisper` CTranslate2, int8 CPU) | `SnapdragonASR` (QAIRT/ONNX-compiled for Hexagon NPU) |
| **Speaker Diarization** | `MockASR` (turn heuristic) | Turn/pause-based speaker attribution ("Speaker 1", "Speaker 2") | PyTorch audio clustering or NPU turn classifier |
| **Summarizer** | `ExtractiveSummarizer` (word-frequency scoring) | `LLMSummarizer` (Local LLM via HuggingFace `transformers`) | Snapdragon GGUF / HTP-compiled LLM |
| **RAG Q&A** | `TfidfRAG` (scikit-learn TF-IDF, top chunk) | `TfidfRAG` + `LLMAdapter` with Grounding verification | Sentence-transformer embeddings + Hexagon NPU LLM |
| **Slide / Screen OCR** | Runtime path check alert | `SlideOCR` (`pytesseract` + Pillow, extracts slide text into notes) | On-device Vision/OCR model |
| **Flashcard Export** | Hardcoded Extractive CSV | `export_flashcards_csv` (LLM-generated Q&A front/back pairs in Anki format) | Same Anki RFC 4180 CSV output |
| **Export Formats** | Markdown + PDF | Markdown + PDF (merged Slides + Transcript) + Anki CSV | Same |

---

## Quick Start

### 1. Offline / Mock Mode (Instant, zero downloads)
```bash
STUDYPILOT_ASR=mock STUDYPILOT_SUMMARIZER=extractive python app.py
```
Open the printed local Gradio URL. Check **"Use bundled sample lecture"** in Tab 1 and hit **Transcribe Lecture**.

### 2. Real CPU Baseline Mode (Whisper ASR + Local LLM)
```bash
python app.py
```
By default, `app.py` initializes `STUDYPILOT_ASR=whisper` and `STUDYPILOT_SUMMARIZER=llm`. If internet connectivity or model weights are missing, the application automatically falls back to `mock` / `extractive` mode and displays warning alert banners in the UI.

---

## UI Features & Controls

1. **Tab 1: Record / Upload & Diarization**
   - Record from microphone or upload audio file (`.wav`, `.mp3`).
   - Toggle **Enable Speaker Diarization** to attribute transcript segments to "Speaker 1", "Speaker 2", etc.
2. **Tab 2: Summary & Slide OCR**
   - Generate abstractive LLM takeaways and glossary terms.
   - Upload lecture slide screenshots to extract text using Tesseract OCR.
3. **Tab 3: Ask a Question (RAG Q&A)**
   - Ask any question about the lecture. Answers are synthesized from retrieved context excerpts with grounding status warnings if ungrounded.
4. **Tab 4: Export Notes & Flashcards**
   - Export full structured study notes in Markdown or PDF formats.
   - Click **Export Flashcards (Anki CSV)** to generate an Anki-importable deck.

---

## Snapdragon NPU Acceleration & Benchmarking (Qualcomm AI Hub)

The lecture transcription model (`whisper_base`) was compiled and profiled on **Snapdragon X Elite CRD** hardware via **Qualcomm AI Hub**:

### Reproducing the Compilation & Benchmark
> **Note:** `qai-hub-models` requires Python 3.10–3.13 (Python 3.14 is currently unsupported). Set up a dedicated virtual environment:

```bash
# 1. Create and activate Python 3.12 environment
uv python install 3.12
py -3.12 -m venv qai_env
.\qai_env\Scripts\activate

# 2. Install AI Hub tools and extras
pip install "qai-hub-models[whisper_base_en]" qai-hub transformers

# 3. Configure Qualcomm AI Hub credentials
qai-hub configure --api_token $env:QAI_HUB_API_TOKEN

# 4. Compile & Profile on Snapdragon X Elite CRD
qai-hub-models export whisper_base --runtime precompiled_qnn_onnx --device "Snapdragon X Elite CRD"
```

### Verified Benchmark Results (Snapdragon X Elite CRD)
- **Whisper Encoder**: **45.53 ms** on Hexagon NPU (Peak RAM: 108.18 MB) vs. CPU Baseline.
- **Whisper Decoder**: **3.66 ms – 3.89 ms** per autoregressive token step (Peak RAM: 127.05 MB).
- **Speedup**: **~5.01x overall transcription speedup** compared to `int8` CPU baseline.
- **Artifacts**: Downloaded compiled ONNX models are available in `scratch/downloaded_models/`.

### Deployment to Physical Snapdragon HP AI PC
To execute the compiled model locally on a physical Snapdragon-powered Windows on ARM device:
1. Load the compiled ONNX model with `onnxruntime-qnn` / QNN Execution Provider.
2. Update `SnapdragonASR.transcribe()` in `backend/asr.py` with the onnxruntime session.
