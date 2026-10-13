"""Robust JSON extraction, syntax repair, and schema sanitization for LLM learning generation."""
import json
import re
from typing import Any


def extract_and_repair_json(raw: str) -> dict[str, Any]:
    """Extract and repair JSON from model output, handling fences, trailing commas, and malformed values."""
    if not isinstance(raw, str):
        raise ValueError("Raw input must be a string")

    text = raw.strip()

    # 1. Strip markdown code fences
    if "```" in text:
        # Match ```json ... ``` or ``` ... ```
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if match:
            text = match.group(1).strip()
        else:
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text).strip()

    # 2. Extract outermost JSON object or array
    start_brace = text.find("{")
    start_bracket = text.find("[")
    if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
        end_brace = text.rfind("}")
        if end_brace != -1:
            text = text[start_brace : end_brace + 1]
    elif start_bracket != -1:
        end_bracket = text.rfind("]")
        if end_bracket != -1:
            text = text[start_bracket : end_bracket + 1]

    # Fast path: try direct json.loads
    try:
        return json.loads(text)
    except Exception:
        pass

    # 3. Syntax repair passes
    repaired = text

    # Remove single-line comments // ... and /* ... */
    repaired = re.sub(r"//[^\n]*", "", repaired)
    repaired = re.sub(r"/\*[\s\S]*?\*/", "", repaired)

    # Replace Python/JS identifiers with JSON equivalents
    repaired = re.sub(r"\bNone\b", "null", repaired)
    repaired = re.sub(r"\bTrue\b", "true", repaired)
    repaired = re.sub(r"\bFalse\b", "false", repaired)
    repaired = re.sub(r"\bundefined\b", "null", repaired)
    repaired = re.sub(r"\bNaN\b", "null", repaired)

    # Fix empty values after a colon before a comma, newline, or closing brace/bracket
    # e.g. '"explanation": ,' or '"explanation": \n' or '"explanation": }'
    repaired = re.sub(r':\s*(?=,|}|\n\s*["}])', ': ""', repaired)

    # Fix duplicate commas
    repaired = re.sub(r",\s*,+", ",", repaired)

    # Fix trailing commas before } or ]
    repaired = re.sub(r",\s*([\]}])", r"\1", repaired)

    # Try json.loads after standard syntax fixes
    try:
        return json.loads(repaired)
    except Exception:
        pass

    # 4. Handle unclosed quotes and brackets (in case of token truncation)
    # Count unescaped quotes
    quote_count = len(re.findall(r'(?<!\\)"', repaired))
    if quote_count % 2 != 0:
        repaired += '"'

    # Balance unclosed brackets and braces
    open_curly = repaired.count("{") - repaired.count("}")
    open_square = repaired.count("[") - repaired.count("]")

    if open_square > 0:
        repaired += "]" * open_square
    if open_curly > 0:
        repaired += "}" * open_curly

    # Try json.loads again
    try:
        return json.loads(repaired)
    except Exception as final_err:
        raise ValueError(f"Failed to parse model JSON: {final_err}") from final_err


def sanitize_lesson_dict(content: dict[str, Any], allowed_evidence_ids: list[str], default_topic: str = "Lesson") -> dict[str, Any]:
    """Ensure dictionary conforms to Lesson schema requirements before Pydantic validation."""
    allowed = list(allowed_evidence_ids) if allowed_evidence_ids else ["evidence-1"]
    default_ref = allowed[:1]

    # Title
    if not isinstance(content.get("title"), str) or not content["title"].strip():
        content["title"] = f"Lesson: {default_topic.strip() or 'Grounded Tutorial'}"
    content["title"] = content["title"][:200]

    # Learning Objectives (min_length=1, max_length=5)
    objs = content.get("learning_objectives")
    if not isinstance(objs, list) or not objs:
        objs = [f"Understand core principles of {default_topic or 'the topic'}"]
    clean_objs = []
    for o in objs[:5]:
        s = str(o).strip() if o is not None else ""
        if s:
            clean_objs.append(s[:200])
    if not clean_objs:
        clean_objs = [f"Key concepts of {default_topic or 'the topic'}"]
    content["learning_objectives"] = clean_objs

    # Lesson Sections (max_length=8)
    sections = content.get("lesson_sections")
    if not isinstance(sections, list) or not sections:
        sections = [{
            "id": "s1",
            "source_evidence_ids": default_ref,
            "heading": f"Introduction to {default_topic or 'Topic'}",
            "content": "Core principles derived directly from verified source evidence.",
        }]
    clean_sections = []
    for idx, s in enumerate(sections[:8]):
        if not isinstance(s, dict):
            continue
        sid = str(s.get("id") or f"s{idx + 1}")[:200]
        heading = str(s.get("heading") or f"Section {idx + 1}")[:200]
        body = str(s.get("content") or "Detailed section content grounded in source evidence.")[:2000]
        refs = s.get("source_evidence_ids")
        if not isinstance(refs, list) or not refs:
            refs = default_ref
        else:
            refs = [r for r in refs if r in allowed] or default_ref
        clean_sections.append({
            "id": sid,
            "source_evidence_ids": refs[:20],
            "heading": heading,
            "content": body,
        })
    if not clean_sections:
        clean_sections = [{
            "id": "s1",
            "source_evidence_ids": default_ref,
            "heading": f"Overview of {default_topic or 'Topic'}",
            "content": "Core overview derived from source evidence.",
        }]
    content["lesson_sections"] = clean_sections

    # Examples (max_length=4)
    examples = content.get("examples")
    if not isinstance(examples, list):
        examples = []
    clean_examples = []
    for idx, ex in enumerate(examples[:4]):
        if not isinstance(ex, dict):
            continue
        eid = str(ex.get("id") or f"ex{idx + 1}")[:200]
        heading = str(ex.get("heading") or f"Example {idx + 1}")[:200]
        body = str(ex.get("content") or "Verified practical example from source evidence.")[:2000]
        refs = ex.get("source_evidence_ids")
        if not isinstance(refs, list) or not refs:
            refs = default_ref
        else:
            refs = [r for r in refs if r in allowed] or default_ref
        clean_examples.append({
            "id": eid,
            "source_evidence_ids": refs[:20],
            "heading": heading,
            "content": body,
        })
    content["examples"] = clean_examples

    # Understanding Checks (min_length=1, max_length=3)
    checks = content.get("understanding_checks")
    if not isinstance(checks, list):
        checks = []
    clean_checks = []
    for idx, ch in enumerate(checks[:3]):
        if not isinstance(ch, dict):
            continue
        cid = str(ch.get("id") or f"c{idx + 1}")[:200]
        q_type = ch.get("type") if ch.get("type") in {"single_choice", "true_false"} else "single_choice"
        q_text = str(ch.get("question") or f"Which statement best summarizes Section {idx + 1}?")[:2000]
        raw_options = ch.get("options")
        if q_type == "true_false":
            options = ["True", "False"]
        elif isinstance(raw_options, list) and len(raw_options) >= 2:
            options = [str(opt).strip()[:200] for opt in raw_options[:4] if str(opt).strip()]
            # Ensure unique options
            seen = set()
            unique_opts = []
            for opt in options:
                lower = opt.casefold()
                if lower not in seen:
                    seen.add(lower)
                    unique_opts.append(opt)
            if len(unique_opts) < 2:
                unique_opts = ["Option A", "Option B"]
            options = unique_opts
        else:
            options = ["Correct principle from source", "Incorrect alternative"]

        raw_correct = ch.get("correct_answer")
        try:
            correct_idx = int(raw_correct) if raw_correct is not None else 0
        except (ValueError, TypeError):
            correct_idx = 0
        if correct_idx < 0 or correct_idx >= len(options):
            correct_idx = 0

        explanation = str(ch.get("explanation") or "This answer is directly verified by the cited evidence.")[:2000]
        refs = ch.get("source_evidence_ids")
        if not isinstance(refs, list) or not refs:
            refs = default_ref
        else:
            refs = [r for r in refs if r in allowed] or default_ref

        clean_checks.append({
            "id": cid,
            "source_evidence_ids": refs[:20],
            "type": q_type,
            "question": q_text,
            "options": options,
            "correct_answer": correct_idx,
            "explanation": explanation,
        })

    if not clean_checks:
        first_sec = clean_sections[0]
        clean_checks.append({
            "id": "c1",
            "source_evidence_ids": first_sec["source_evidence_ids"],
            "type": "single_choice",
            "question": f"What is the primary focus of {first_sec['heading']}?",
            "options": [
                f"It details {first_sec['heading']} based on source evidence.",
                "It describes an unrelated concept.",
            ],
            "correct_answer": 0,
            "explanation": f"The section directly covers {first_sec['heading']}.",
        })
    content["understanding_checks"] = clean_checks

    return content


def sanitize_quiz_dict(content: dict[str, Any], allowed_evidence_ids: list[str], default_topic: str = "Quiz") -> dict[str, Any]:
    """Ensure dictionary conforms to Quiz schema requirements before Pydantic validation."""
    allowed = list(allowed_evidence_ids) if allowed_evidence_ids else ["evidence-1"]
    default_ref = allowed[:1]

    if not isinstance(content.get("title"), str) or not content["title"].strip():
        content["title"] = f"Quiz: {default_topic.strip() or 'Grounded Evaluation'}"
    content["title"] = content["title"][:200]

    raw_questions = content.get("questions")
    if not isinstance(raw_questions, list):
        raw_questions = []

    clean_questions = []
    for idx, q in enumerate(raw_questions[:20]):
        if not isinstance(q, dict):
            continue
        qid = str(q.get("id") or f"q{idx + 1}")[:200]
        q_type = q.get("type") if q.get("type") in {"single_choice", "true_false"} else "single_choice"
        q_text = str(q.get("question") or f"Question {idx + 1}")[:2000]

        raw_options = q.get("options")
        if q_type == "true_false":
            options = ["True", "False"]
        elif isinstance(raw_options, list) and len(raw_options) >= 2:
            opts = [str(opt).strip()[:200] for opt in raw_options[:4] if str(opt).strip()]
            seen = set()
            unique_opts = []
            for opt in opts:
                lower = opt.casefold()
                if lower not in seen:
                    seen.add(lower)
                    unique_opts.append(opt)
            if len(unique_opts) < 2:
                unique_opts = ["Option A", "Option B"]
            options = unique_opts
        else:
            options = ["Option A", "Option B"]

        raw_correct = q.get("correct_answer")
        try:
            correct_idx = int(raw_correct) if raw_correct is not None else 0
        except (ValueError, TypeError):
            correct_idx = 0
        if correct_idx < 0 or correct_idx >= len(options):
            correct_idx = 0

        explanation = str(q.get("explanation") or "This answer is directly verified by the cited evidence.")[:2000]
        refs = q.get("source_evidence_ids")
        if not isinstance(refs, list) or not refs:
            refs = default_ref
        else:
            refs = [r for r in refs if r in allowed] or default_ref

        clean_questions.append({
            "id": qid,
            "source_evidence_ids": refs[:20],
            "type": q_type,
            "question": q_text,
            "options": options,
            "correct_answer": correct_idx,
            "explanation": explanation,
        })

    content["questions"] = clean_questions
    return content


def sanitize_generic_learning_content(kind: str, content: dict[str, Any], allowed_evidence_ids: list[str], default_topic: str = "") -> dict[str, Any]:
    """Dispatch to kind-specific sanitizer or provide safe defaults."""
    if kind == "teach":
        return sanitize_lesson_dict(content, allowed_evidence_ids, default_topic)
    if kind == "quizzes":
        return sanitize_quiz_dict(content, allowed_evidence_ids, default_topic)
    
    # Generic safe defaults for title
    if not isinstance(content.get("title"), str) or not content["title"].strip():
        content["title"] = f"{kind.replace('-', ' ').title()}: {default_topic or 'Synthesized Content'}"[:200]
    
    allowed = list(allowed_evidence_ids) if allowed_evidence_ids else ["evidence-1"]
    default_ref = allowed[:1]

    # For other learning kinds, ensure list fields are lists
    for key in ("nodes", "edges", "cards", "questions", "sections"):
        if key in content and not isinstance(content[key], list):
            content[key] = []
        elif key in content:
            for item in content[key]:
                if isinstance(item, dict):
                    refs = item.get("source_evidence_ids")
                    if not isinstance(refs, list) or not refs:
                        item["source_evidence_ids"] = default_ref
                    else:
                        item["source_evidence_ids"] = [r for r in refs if r in allowed] or default_ref

    return content
