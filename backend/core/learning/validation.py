"""Deterministic structure/citation checks and independent content review."""
import json
import re

from core.guardrails.citations import validate_citations
from core.guardrails.injection import assess_document_content
from core.guardrails.secrets import contains_secret_pattern
from core.guardrails.verification import VerificationStatus, verify_claims
from core.learning.json_repair import extract_and_repair_json
from core.learning.models import Review


class LearningError(ValueError):
    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


def normalized(text):
    return re.sub(r"\W+", " ", text.casefold()).strip()


def units(content):
    for key in ("nodes", "edges", "cards", "questions", "sections", "lesson_sections", "examples", "understanding_checks"):
        for item in content.get(key, []):
            yield key, item


def validate_structure(kind, content, evidence, request, config):
    allowed = {item.evidence.evidence_id for item in evidence}
    if not list(units(content)):
        raise LearningError("The selected evidence does not support this learning activity.")
    serialized = json.dumps(content)
    if contains_secret_pattern(serialized) or assess_document_content(serialized).detected:
        raise LearningError("The generated content failed safety verification. Try another topic.")
    for key in {key for key, _ in units(content)}:
        items = content[key]
        if len({item["id"] for item in items}) != len(items):
            raise LearningError("The model returned duplicate item IDs. Regenerate the activity.")
    for _, item in units(content):
        if not set(item["source_evidence_ids"]).issubset(allowed):
            raise LearningError("The generated content cited evidence outside the supplied scope.")
    if kind == "knowledge-graph":
        if len(content["nodes"]) > config.learning_max_nodes or len(content["edges"]) > config.learning_max_edges:
            raise LearningError("The graph exceeds configured size limits. Choose a narrower topic.")
        # Merge only exact concept+description duplicates, never merely similar names.
        aliases, seen, unique = {}, {}, []
        for node in content["nodes"]:
            fingerprint = (normalized(node["name"]), normalized(node["description"]), node["status"])
            if fingerprint in seen:
                original = seen[fingerprint]
                aliases[node["id"]] = original["id"]
                original["source_evidence_ids"] = sorted(set(original["source_evidence_ids"] + node["source_evidence_ids"]))
            else:
                seen[fingerprint] = node
                unique.append(node)
        content["nodes"] = unique
        node_ids = {node["id"] for node in unique}
        edges, seen_edges = [], set()
        for edge in content["edges"]:
            for key in ("source_node_id", "target_node_id"):
                edge[key] = aliases.get(edge[key], edge[key])
                if edge[key] not in node_ids:
                    raise LearningError("The graph contains a relationship with a missing concept.")
            signature = (edge["source_node_id"], normalized(edge["relation"]), edge["target_node_id"], edge["status"])
            if signature not in seen_edges and edge["source_node_id"] != edge["target_node_id"]:
                edges.append(edge)
                seen_edges.add(signature)
        content["edges"] = edges
    elif kind == "topic-tree":
        if len(content["nodes"]) > config.learning_max_nodes:
            raise LearningError("The outline exceeds the configured size limit.")
        by_id = {item["id"]: item for item in content["nodes"]}
        sibling_names = set()
        for node in content["nodes"]:
            visited = {node["id"]}
            parent = node["parent_id"]
            while parent is not None:
                if parent not in by_id or parent in visited:
                    raise LearningError("The outline contains a cycle or missing parent.")
                visited.add(parent)
                parent = by_id[parent]["parent_id"]
            if len(visited) > 6:
                raise LearningError("The outline is too deeply nested.")
            signature = (node["parent_id"], normalized(node["title"]))
            if signature in sibling_names:
                raise LearningError("The outline contains ambiguous duplicate topics.")
            sibling_names.add(signature)
            sources = [item.evidence for item in evidence if item.evidence.evidence_id in node["source_evidence_ids"]]
            if not any(node["title"] == item.section for item in sources):
                node["inferred"] = True
    elif kind in {"flashcards", "quizzes"}:
        key = "cards" if kind == "flashcards" else "questions"
        limit = config.learning_max_cards if kind == "flashcards" else config.learning_max_questions
        if len(content[key]) > min(request.count, limit):
            raise LearningError("The generated activity exceeds its requested count.")
        seen, unique = set(), []
        for item in content[key]:
            fingerprint = normalized(item["question"])
            if fingerprint not in seen:
                unique.append(item)
                seen.add(fingerprint)
        content[key] = unique
    elif kind == "teach":
        if not content["lesson_sections"]:
            raise LearningError("The evidence did not support a lesson.")
    return content


def review_content(content, evidence, reasoner, context, output_limit):
    """Check facts with existing claim verification, then review all units independently.

    The second pass does not receive the proposed quiz/check answer keys.
    Complete option truth tables must match exactly one correct answer.
    """
    tasks = []
    for group, item in units(content):
        identity = f"{group}/{item['id']}"
        task = {**item, "id": identity, "kind": group}
        task.pop("correct_answer", None)
        if "options" in item:
            task.pop("explanation", None)
        tasks.append(task)
        facts = [item[key] for key in ("description", "summary", "answer", "explanation", "content") if key in item]
        if group == "edges":
            names = {node["id"]: node["name"] for node in content["nodes"]}
            facts = [f"{names[item['source_node_id']]} {item['relation']} {names[item['target_node_id']]}"]
        # Reuse the same citation ledger and lexical-support verification as chat.
        for fact in facts:
            cited = " ".join(
                sentence + " " + " ".join(f"[evidence:{ref}]" for ref in item["source_evidence_ids"])
                for sentence in re.split(r"(?<=[.!?])\s+", fact) if sentence.strip()
            )
            citations = validate_citations(cited, {entry.evidence.evidence_id for entry in evidence})
            is_inferred = item.get("inferred") is True or item.get("status") == "inferred"
            if not citations.valid and not (is_inferred and not item.get("source_evidence_ids")):
                raise LearningError("A generated statement used an invalid evidence citation. Regenerate with a narrower topic.")
            # The existing verifier intentionally uses lexical support for chat
            # prose. Learning summaries can be faithful paraphrases with little
            # shared wording, so the independent NIM review below is the
            # semantic gate for plain prose. Keep the deterministic verifier as
            # a hard gate for numerical/formula claims where lexical evidence
            # and source-preserved notation are essential.
            verified = verify_claims(cited, evidence, citations)
            numeric_pattern = re.compile(r"(?:\d+(?:\.\d+)?|%|=|\b(?:kg|km|mb|gb|usd|seconds?|minutes?|hours?)\b)", re.I)
            unverified_claims = [claim.text for claim in verified.claims if not claim.verified]
            for unverified in unverified_claims:
                if numeric_pattern.search(unverified):
                    raise LearningError("A generated numeric or formula statement could not be verified against its cited sources. Regenerate with a narrower topic.")
            
            # Reject new numerical literals (including altered units/formulas in review).
            clean_fact = re.sub(r"(?<=\d),(?=\d)", "", fact)
            source_text = " ".join(f"{entry.evidence.filename} {entry.evidence.text or ''}" for entry in evidence if entry.evidence.evidence_id in item["source_evidence_ids"])
            clean_source = re.sub(r"(?<=\d),(?=\d)", "", source_text)
            fact_nums = {float(n) for n in re.findall(r"\b\d+(?:\.\d+)?\b", clean_fact)}
            all_source_text = " ".join(f"{entry.evidence.filename} {entry.evidence.text or ''}" for entry in evidence)
            clean_all_source = re.sub(r"(?<=\d),(?=\d)", "", all_source_text)
            all_source_nums = {float(n) for n in re.findall(r"\b\d+(?:\.\d+)?\b", clean_all_source)}
            unsupported_nums = {n for n in (fact_nums - all_source_nums) if n > 12}
            if unsupported_nums:
                raise LearningError(f"Generated numeric values ({', '.join(str(n) for n in unsupported_nums)}) were not found in the cited evidence.")
        if "options" in item:
            tasks.append({"id": identity + "/explanation", "content": item["explanation"], "source_evidence_ids": item["source_evidence_ids"]})
    instruction = (
        "Independently audit each supplied item against ONLY its cited evidence. Return one assessment per exact id. "
        "supported requires every factual clause, premise, formula, unit and condition to be supported. "
        "For inferred relationships, require a defensible connection and reject contradictions. "
        "Reject ambiguous questions, answer-leaking questions, unsupported false statements, and materially misleading summaries. "
        "For questions/checks independently solve the question, checking EVERY option: option_correct is a boolean "
        "per option indicating whether it is a correct answer. True/false statements must be provably true or false, "
        "not merely absent. For other items option_correct is []. An analogy must be labelled and preserve the supported mechanism. "
        "Audit objectives/title for unsupported promises too. Task data follows as JSON:\n"
        + json.dumps({"title": content["title"], "objectives": content.get("learning_objectives", []), "items": tasks})
    )
    raw = reasoner.structured(instruction, context, Review.model_json_schema(), max_tokens=6000)
    if len(raw) > output_limit:
        raise LearningError("Model review exceeded the output limit.")
    try:
        review_dict = extract_and_repair_json(raw)
        review = Review.model_validate(review_dict)
    except ValueError as error:
        raise LearningError("The model returned an invalid verification response. Retry generation.") from error
    assessments = {item.id: item for item in review.assessments}
    main_task_ids = {item["id"] for item in tasks if not item["id"].endswith("/explanation")}
    if not main_task_ids.issubset(set(assessments)):
        raise LearningError("The model did not verify every generated item. Retry generation.")
    for group, item in units(content):
        identity = f"{group}/{item['id']}"
        assessment = assessments.get(identity)
        if not assessment:
            raise LearningError(f"Item '{identity}' was not evaluated by the review model.")
        if not assessment.supported or not assessment.unambiguous:
            raise LearningError(f"Item '{identity}' was evaluated as supported={assessment.supported}, unambiguous={assessment.unambiguous}. Ground all claims strictly in cited evidence.")
        expected = [index == item["correct_answer"] for index in range(len(item["options"]))] if "options" in item else []
        if assessment.option_correct != expected:
            raise LearningError(f"Question/check item '{identity}' answer key could not be validated. Ensure exactly one option is unambiguously correct according to cited evidence.")
        if "options" in item:
            explanation = assessments.get(identity + "/explanation")
            if explanation and (not explanation.supported or not explanation.unambiguous):
                raise LearningError(f"Explanation for '{identity}' was evaluated as supported={explanation.supported}, unambiguous={explanation.unambiguous}. Ground explanation directly in cited evidence.")
    return content
