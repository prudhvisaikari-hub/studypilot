"""
Summarization module for StudyPilot.

  - ExtractiveSummarizer: pure-Python, zero downloads. Uses word-frequency
    scoring (a small TextRank-style heuristic) to pick the most important
    sentences and extract candidate key terms. Works offline, instantly.

  - LLMSummarizer: abstractive, high-quality summaries using a local
    small LLM (e.g. Phi-3-mini-4k-instruct or Qwen2.5-0.5B-Instruct) via `transformers`.
"""

from __future__ import annotations
import re
import json
import logging
from collections import Counter
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class SummarizerError(Exception):
    """Raised when summarization fails."""
    pass


class SummarizerInitError(SummarizerError):
    """Raised when LLM summarizer initialization fails."""
    pass


_STOPWORDS = set("""
a an the is are was were be been being to of in on for with as at by from
this that these those it its it's and or but if then than so such not no
we you they he she i my your our their his her them us do does did done
can could will would should may might must shall have has had having
""".split())


def _sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 0]


def _word_freqs(text: str) -> Counter:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-']+", text.lower())
    words = [w for w in words if w not in _STOPWORDS and len(w) > 2]
    return Counter(words)


class ExtractiveSummarizer:
    """Pure-Python extractive summarizer for offline / mock mode."""

    def summarize(self, text: str, max_sentences: int = 6) -> Dict[str, List[str]]:
        sents = _sentences(text)
        if not sents:
            return {"bullets": [], "key_terms": []}

        freqs = _word_freqs(text)
        max_freq = max(freqs.values()) if freqs else 1

        scored = []
        for idx, s in enumerate(sents):
            words = re.findall(r"[a-zA-Z][a-zA-Z\-']+", s.lower())
            if not words:
                continue
            score = sum(freqs.get(w, 0) for w in words) / max_freq / len(words)
            position_boost = 1.15 if idx < max(1, len(sents) // 8) else 1.0
            scored.append((score * position_boost, idx, s))

        top = sorted(scored, key=lambda x: x[0], reverse=True)[:max_sentences]
        top_sorted_by_position = [s for _, _, s in sorted(top, key=lambda x: x[1])]

        key_terms = [w.title() for w, _ in freqs.most_common(12)]

        return {
            "bullets": top_sorted_by_position,
            "key_terms": key_terms,
        }


class LLMSummarizer:
    """Abstractive summarizer backed by a local small LLM via HuggingFace transformers.
    Supports robust JSON extraction with regex fallback parsing.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.model_name = model_name
        self.pipe = None
        self.tokenizer = None
        self.model = None

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import torch

            logger.info(f"Loading LLM model '{model_name}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=350,
                temperature=0.3,
                do_sample=False,
            )
            logger.info("LLMSummarizer loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load LLM model '{model_name}': {e}")
            raise SummarizerInitError(
                f"Failed to load LLM summarizer '{model_name}'. Error: {str(e)}"
            ) from e

    def generate_text(self, prompt: str) -> str:
        """Expose a clean generate_text interface for RAG / Q&A adapter calls."""
        if not self.pipe:
            raise SummarizerError("LLM pipeline is not initialized.")
        try:
            messages = [{"role": "user", "content": prompt}]
            if hasattr(self.tokenizer, "apply_chat_template"):
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                formatted_prompt = prompt

            output = self.pipe(formatted_prompt)
            gen_text = output[0]["generated_text"]
            if isinstance(gen_text, list):
                gen_text = gen_text[-1].get("content", "")
            elif gen_text.startswith(formatted_prompt):
                gen_text = gen_text[len(formatted_prompt):].strip()
            return gen_text.strip()
        except Exception as e:
            logger.error(f"LLM text generation failed: {e}")
            raise SummarizerError(f"LLM generation failed: {str(e)}") from e

    def summarize(self, text: str, max_sentences: int = 6) -> Dict[str, List[str]]:
        if not text or not text.strip():
            return {"bullets": [], "key_terms": []}

        prompt = (
            "You are a helpful study assistant. Analyze the following lecture transcript. "
            "Return a valid JSON object with exactly two keys:\n"
            '1. "bullets": a list of 4-6 concise key takeaways as strings.\n'
            '2. "key_terms": a list of 5-10 important technical terms as strings.\n\n'
            "JSON Format ONLY:\n"
            '{"bullets": ["..."], "key_terms": ["..."]}\n\n'
            f"Transcript:\n{text[:4000]}"
        )

        raw_output = self.generate_text(prompt)
        return self._parse_response(raw_output, fallback_text=text)

    def _parse_response(self, raw_output: str, fallback_text: str) -> Dict[str, List[str]]:
        """Robust parser: JSON first, then Regex fallback to ensure no silent empty outputs."""
        bullets: List[str] = []
        key_terms: List[str] = []

        # Strategy 1: Attempt direct or regex JSON extraction
        json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if isinstance(data, dict):
                    bullets = [str(b).strip() for b in data.get("bullets", []) if str(b).strip()]
                    key_terms = [str(k).strip() for k in data.get("key_terms", []) if str(k).strip()]
            except json.JSONDecodeError:
                pass

        # Strategy 2: Regex section fallback if JSON parsing failed or produced empty lists
        if not bullets or not key_terms:
            current_section = None
            for line in raw_output.splitlines():
                line_str = line.strip()
                if not line_str:
                    continue
                if "bullet" in line_str.lower() or "takeaway" in line_str.lower() or "summary" in line_str.lower():
                    current_section = "bullets"
                    continue
                elif "term" in line_str.lower() or "glossary" in line_str.lower():
                    current_section = "terms"
                    continue

                clean_item = re.sub(r"^[\*\-\d\.\:\s]+", "", line_str).strip()
                if clean_item and len(clean_item) > 2:
                    if current_section == "bullets" or (not current_section and len(clean_item) > 20):
                        bullets.append(clean_item)
                    elif current_section == "terms" or (not current_section and len(clean_item) <= 20):
                        key_terms.append(clean_item)

        # Strategy 3: Hard safety fallback to ExtractiveSummarizer if LLM response was completely unparseable
        if not bullets or not key_terms:
            logger.warning("LLM response parsing yielded incomplete results. Applying Extractive fallback.")
            extractive = ExtractiveSummarizer().summarize(fallback_text)
            if not bullets:
                bullets = extractive["bullets"]
            if not key_terms:
                key_terms = extractive["key_terms"]

        return {"bullets": bullets[:8], "key_terms": key_terms[:12]}


def get_summarizer(name: str = "extractive", **kwargs):
    name = name.lower()
    if name == "extractive":
        return ExtractiveSummarizer()
    if name == "llm":
        model_name = kwargs.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct")
        return LLMSummarizer(model_name=model_name)
    raise ValueError(f"Unknown summarizer backend: {name}")
