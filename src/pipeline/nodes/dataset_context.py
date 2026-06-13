"""
Node 1.5 — Dataset Context Builder + Dynamic VLM Prompt Generator

Runs AFTER yolo_inference and BEFORE vlm_annotation.

Purpose: Use Groq LLM to generate a tailored Gemini system prompt based on
what we already know BEFORE annotation starts:
  - known defect class names (so Gemini distinguishes novel vs known)
  - inspection domain hint (steel_casting, pcb, plastic_moulding, etc.)
  - number of unknown images found by YOLO
  - novelty ratio from this YOLO run

This makes the first VLM bbox detection flexible and domain-aware, rather
than using a one-size-fits-all static prompt for every defect type.

If Groq is unavailable or fails, falls back silently to the static prompt.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
import config as cfg
from src.utils import get_logger

log = get_logger("dataset_context")


# ── Pre-annotation context builder ─────────────────────────

def _build_pre_annotation_context(state: dict) -> dict:
    """
    Build a context snapshot from what is known BEFORE VLM annotation starts.
    Only uses state available after yolo_inference.
    """
    known_names = state.get("known_defect_names", [])
    unknown_paths = state.get("unknown_image_paths", [])
    all_paths = state.get("all_image_paths", [])

    total_images = len(all_paths) if all_paths else 0
    unknown_count = len(unknown_paths) if unknown_paths else 0
    novelty_ratio = round(unknown_count / total_images, 3) if total_images > 0 else 0.0

    return {
        "inspection_domain": cfg.INSPECTION_DOMAIN,
        "known_class_names": known_names,
        "known_class_count": len(known_names),
        "total_input_images": total_images,
        "unknown_image_count": unknown_count,
        "novelty_ratio": novelty_ratio,
        # Signal: high novelty → Gemini should explore new label names
        "high_novelty": novelty_ratio > 0.5,
    }


def _context_hash(context: dict) -> str:
    """Deterministic cache key for a context dict."""
    canon = json.dumps(context, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


# ── Groq meta-prompt ────────────────────────────────────────

_GROQ_SYSTEM = """You are a prompt engineer for an industrial vision AI system.
Your task is to write a DETECTION system prompt for a Gemini VLM.

The VLM receives raw inspection images and must return tight bounding-box
annotations of any manufacturing defects.

You receive a JSON context describing the current inspection run.
Write a Gemini system prompt that:

1. Clearly states the inspection domain so Gemini uses correct vocabulary.
2. Lists ALL known defect class names so Gemini can distinguish truly NEW
   defects from near-misses with existing classes.
3. If high_novelty is true or novelty_ratio > 0.5, tells Gemini to coin a
   NEW descriptive label rather than forcing a known class fit.
4. Always requires this exact JSON output:
   {
     "anomalies_found": bool,
     "findings": [
       {
         "box_2d": [ymin, xmin, ymax, xmax],
         "physical_traits": "concise defect description",
         "label": "defect_name",
         "confidence": 0.0-1.0,
         "severity": "low|medium|high|critical"
       }
     ]
   }
   box_2d values are in 0-1000 scale relative to image dimensions.
5. Instructs drawing TIGHT boxes (5-8% margin), covering structural
   violations, surface defects, and tonal/material anomalies.

OUTPUT: Return ONLY the Gemini system prompt text. No explanation, no markdown fences."""


def _generate_vlm_prompt_via_groq(context: dict) -> str | None:
    """
    Call Groq LLM to generate a tailored Gemini VLM detection prompt.
    Returns None on any failure — pipeline always falls back to static prompt.
    """
    if not cfg.GROQ_API_KEY:
        log.info("No GROQ_API_KEY — skipping dynamic prompt generation")
        return None

    ctx_hash = _context_hash(context)
    cache_path = cfg.VLM_PROMPT_CACHE_PATH
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            if ctx_hash in cache:
                log.info(f"VLM prompt cache hit (hash={ctx_hash})")
                return cache[ctx_hash]
        except (json.JSONDecodeError, OSError):
            pass

    try:
        from groq import Groq

        client = Groq(api_key=cfg.GROQ_API_KEY)
        response = client.chat.completions.create(
            model=cfg.GROQ_ADVISOR_MODEL,
            messages=[
                {"role": "system", "content": _GROQ_SYSTEM},
                {"role": "user", "content": json.dumps(context, indent=2)},
            ],
            temperature=0.3,
            max_tokens=1500,
        )
        prompt_text = response.choices[0].message.content.strip()

        if not prompt_text or len(prompt_text) < 50:
            log.warning("Groq returned empty/short prompt — using static fallback")
            return None

        # Cache result
        try:
            cache = {}
            if cache_path.exists():
                cache = json.loads(cache_path.read_text(encoding="utf-8"))
            cache[ctx_hash] = prompt_text
            cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        except Exception as e:
            log.warning(f"Failed to write VLM prompt cache: {e}")

        log.info(
            f"Groq generated dynamic VLM detection prompt "
            f"({len(prompt_text)} chars, hash={ctx_hash})"
        )
        return prompt_text

    except ImportError:
        log.warning("groq package not installed — using static VLM prompt")
        return None
    except Exception as e:
        log.warning(f"Groq prompt generation failed: {e}")
        return None


# ── LangGraph node ──────────────────────────────────────────

def dataset_context_node(state: dict) -> dict:
    """
    LangGraph node — runs AFTER yolo_inference, BEFORE vlm_annotation.

    Builds a pre-annotation context from known defects + domain hint,
    calls Groq to generate a domain-specific Gemini detection prompt,
    and writes it into state so vlm_annotation picks it up.

    Reads:
        state["known_defect_names"]
        state["unknown_image_paths"]
        state["all_image_paths"]

    Writes:
        state["dataset_context"]     — context snapshot dict
        state["vlm_system_prompt"]   — Groq-generated prompt (if successful)
    """
    from src.utils import LogStream

    # Build pre-annotation context
    try:
        context = _build_pre_annotation_context(state)
        log.info(
            f"Pre-VLM context: {context['unknown_image_count']} unknown images, "
            f"domain='{context['inspection_domain']}', "
            f"novelty_ratio={context['novelty_ratio']}, "
            f"{context['known_class_count']} known classes"
        )
        LogStream.emit(
            f"Context: {context['unknown_image_count']} unknown images | "
            f"domain={context['inspection_domain']} | "
            f"{context['known_class_count']} known defect classes",
            level="info",
            source="dataset_context",
        )
    except Exception as e:
        log.warning(f"Context building failed — using empty context: {e}")
        context = {}

    # Generate dynamic detection prompt via Groq
    vlm_prompt = None
    try:
        vlm_prompt = _generate_vlm_prompt_via_groq(context)
        if vlm_prompt:
            LogStream.emit(
                "Dynamic VLM detection prompt generated via Groq",
                level="step",
                source="dataset_context",
            )
        else:
            LogStream.emit(
                "Static VLM detection prompt will be used",
                level="info",
                source="dataset_context",
            )
    except Exception as e:
        log.warning(f"Dynamic prompt generation failed: {e}")

    result: dict = {"dataset_context": context}
    if vlm_prompt:
        result["vlm_system_prompt"] = vlm_prompt

    return result
