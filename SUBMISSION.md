# StudyPilot — Snapdragon AI Lab Challenge Submission

## Title
StudyPilot – On-Device Lecture Summarizer & Doubt Solver for Snapdragon AI PCs

## Problem statement
Students attend long lectures but struggle to revise efficiently. Existing
note-taking tools are manual or cloud-based, raising privacy and latency
concerns.

## Proposed solution
StudyPilot is a Windows desktop app that:
- Records lectures (audio, with optional pre-recorded upload)
- Transcribes speech to text locally using an NPU-accelerated ASR model
- Generates structured summaries, key points, and a glossary
- Lets students ask questions about the lecture via a local LLM with
  retrieval-augmented generation (RAG) over the transcript

All processing happens on-device on Snapdragon-powered HP PCs, ensuring
privacy, low latency, and offline usability.

## Why Snapdragon-powered HP PCs
- Leverages the Hexagon NPU (40+ TOPS) for efficient ASR and LLM inference.
- Demonstrates the real-world "AI PC" benefits: faster AI features, longer
  battery life, offline operation.
- Uses Qualcomm AI Hub / QAIRT tooling to compile and optimize models for
  Snapdragon.

## What's built
A working, tested end-to-end prototype (code included) with a pluggable
backend architecture:

- **Transcription**: mock/offline mode for demo, real Whisper (faster-whisper)
  CPU/GPU baseline, and a defined extension point for a Snapdragon-NPU-compiled
  ASR model.
- **Summarization**: a dependency-free extractive summarizer that works fully
  offline today, plus an abstractive LLM path (Phi-3-mini class model) for
  higher-quality output.
- **Q&A / RAG**: transcript chunking + TF-IDF retrieval, with an extension
  point to feed retrieved context to a local LLM for generated answers.
- **Export**: one-click Markdown and PDF export of notes, key terms, and Q&A
  history.
- **UI**: a four-tab Gradio interface — Record/Upload, Summary, Ask a
  Question, Export.

## Architecture
```
Lecture audio
   -> On-device ASR (Hexagon NPU, whisper-based)
   -> Transcript
        -> Summarizer (bullets + key terms, local LLM)
        -> RAG Q&A (chunk, embed, retrieve, answer)
   -> Export notes (Markdown / PDF)
```
Everything above runs on the Snapdragon AI PC; no audio, transcript, or
question ever leaves the device.

## Snapdragon optimization plan
1. Export ASR (whisper-small/base) and the chosen 1–3B LLM (Phi-3-mini /
   TinyLlama) to ONNX/GGUF.
2. Compile for the Hexagon NPU using Qualcomm AI Hub / QAIRT's GGUF/HTP
   builder.
3. Profile latency and memory with AI Hub Workbench on-device.
4. Report tokens/sec and transcription latency vs the CPU baseline.

## Privacy & student impact
- Sensitive lecture content never leaves the laptop.
- Helps students revise faster and ask precise doubts instead of re-listening
  to entire recordings.

## Scalability
The backend-agnostic design (swap ASR/summarizer via one environment
variable) means the same app can extend to multiple courses, languages, or
integrate with existing note-taking tools without a rewrite.

## Repository
See the attached `studypilot` project: `README.md` has full setup
instructions, the Snapdragon NPU integration steps, and known limitations.

## Demo video checklist (record this yourself on your machine)
- [ ] Record or upload a short lecture snippet
- [ ] Show live transcription
- [ ] Show summary generation (bullets + key terms)
- [ ] Ask 2–3 questions and show answers
- [ ] Call out "runs 100% on-device on Snapdragon AI PC — no cloud"
- [ ] (If tested on real hardware) show latency numbers vs CPU baseline
