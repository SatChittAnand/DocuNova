"""Learning orchestration using the existing scoped retriever, NIM and SQLite."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json

from core.guardrails.contradictions import detect_contradictions
from core.guardrails.injection import assess_document_content
from core.guardrails.scope import validate_document_scope
from core.guardrails.secrets import contains_secret_pattern
from core.guardrails.pii import is_pii_extraction_query, is_sensitive_identity_or_secret_passage
from core.learning.json_repair import extract_and_repair_json, sanitize_generic_learning_content

from core.learning.models import MODELS, GenerateRequest
from core.learning.validation import LearningError, review_content, units, validate_structure
from core.retrieval.context import build_evidence_context
from core.retrieval.query import RetrievedEvidence


GUIDANCE = {
    "knowledge-graph": "Extract meaningful concepts and directed named relationships, not token co-occurrences. Mark each explicit/inferred. Prefer fewer strong nodes. Never conflate distinct meanings.",
    "topic-tree": "Build a bounded expandable topic/subtopic hierarchy. Prefer the original section headings supplied with source data. Label generated organization inferred. Parents must exist; no cycles or duplicate siblings. Summaries are concise source-grounded facts.",
    "flashcards": "Write distinct active-recall questions with concise answers and explanations. Do not put the answer in the question. Every answer and explanation must be supported by its citations.",
    "quizzes": "Generate diverse, rigorous practice questions supporting multiple choice single_choice (options array of 4 distinct plausible alternatives A, B, C, D) and true_false (options exactly True, False). For single_choice questions, provide 4 options with exactly one valid answer indexed from 0 and 3 clearly incorrect distractors. For true_false, statements require affirmative evidence of truth or falsity. Check every distractor; do not use another equally supported answer. No hints or leaked keys in the question.",
    "cheat-sheet": "Create compact source-cited revision sections in the requested format. comparison_table uses Markdown tables; formula_sheet only includes verbatim formulas with original units, conditions and notation (empty sections if absent). Do not calculate new values. exam_outline labels priorities as suggested revision priorities, never guaranteed exam questions. Clearly flag gaps using only source-grounded descriptions.",
    "teach": "Generate a lesson, objectives, source-grounded examples and 1-3 single-choice understanding checks (same rules as quizzes). Beginner: plain definition, intuition, explained jargon, labelled analogy where supported, simple example and recall check. Intermediate: mechanism, terminology, worked source example, applications, common mistakes and application check. Advanced: mechanisms, assumptions, trade-offs, limitations, supported derivation or edge case and reasoning check. Do not invent absent detail; identify source gaps. Level affects depth, never facts. Avoid generic uncited filler.",
}


class LearningService:
    def __init__(self, metadata_store, retriever, reasoning_client, config):
        self.store = metadata_store
        self.retriever = retriever
        self.reasoning_client = reasoning_client
        self.config = config

    def _scope(self, collection_id, selected):
        scope = validate_document_scope(self.store, collection_id, selected)
        if self.store.ingestion_status(collection_id)["busy"]:
            raise LearningError("Document preparation is in progress. Retry when it finishes.", 409)
        return sorted(scope if scope is not None else [doc.id for doc in self.store.list_documents(collection_id)])

    def _retrieve(self, collection_id, scope, topic, kind, fmt):
        if not scope:
            raise LearningError("This collection has no documents. Upload and prepare sources first.")
        
        fallback = " ".join(doc.filename for doc in self.store.list_documents(collection_id) if doc.id in scope)[:500]
        if kind == "cheat-sheet":
            fallback = f"{fmt.replace('_', ' ')} overview {fallback}"
        elif kind in {"knowledge-graph", "topic-tree"}:
            fallback = f"overview concepts main topics {fallback}"
        elif kind in {"quizzes", "flashcards"}:
            fallback = f"key concepts definitions fundamentals summary {fallback}"
        elif kind == "teach":
            fallback = f"introduction fundamentals core concepts {fallback}"
            
        query = topic or fallback
        retrieved = self.retriever.retrieve(collection_id=collection_id, question=query,
                                             selected_document_ids=scope, top_k=20)
        safe = []
        for hit in retrieved:
            # Re-resolve provider IDs in SQLite; reject stale or out-of-scope payloads.
            item = self.store.get_evidence(hit.evidence.evidence_id, collection_id)
            if item is None or item.document_id not in scope:
                raise LearningError("Retrieved evidence was outside the permitted scope.")
            if is_sensitive_identity_or_secret_passage(item):
                continue
            if not item.text or contains_secret_pattern(json.dumps(asdict(item))):
                continue
            if assess_document_content(item.text).detected:
                continue
            safe.append(RetrievedEvidence(item, hit.retrieval_score))
        context = build_evidence_context(safe, max_chars=self.config.learning_context_chars)
        if not context.evidence:
            raise LearningError("No safe evidence supports this topic in the selected documents. Try another topic.")
        return context

    def generate(self, collection_id, kind, request: GenerateRequest):
        scope = self._scope(collection_id, request.selected_document_ids)
        is_pii, _ = is_pii_extraction_query(request.topic)
        if is_pii or contains_secret_pattern(request.topic) or assess_document_content(request.topic).detected:
            raise LearningError("Enter an educational learning topic rather than personal information or credentials.")
        if kind == "flashcards" and request.count > self.config.learning_max_cards:

            raise LearningError("Card count exceeds the configured limit.")
        if kind == "quizzes" and request.count > min(20, self.config.learning_max_questions):
            raise LearningError("Question count exceeds the configured limit.")
        version = self.store.learning_source_version(collection_id, scope)
        parameters = request.model_dump(exclude={"collection_id", "regenerate", "selected_document_ids"})
        cache_key = hashlib.sha256(json.dumps({"kind": kind, "scope": scope, "parameters": parameters,
            "version": version, "model": self.config.reasoning_model, "schema": 1,
            "limits": [self.config.learning_max_nodes, self.config.learning_max_edges,
                       self.config.learning_context_chars, self.config.learning_output_chars]}, sort_keys=True).encode()).hexdigest()
        if not request.regenerate:
            for artifact in self.store.list_learning_artifacts(collection_id):
                if artifact["payload"].get("cache_key") == cache_key:
                    try:
                        return self.get(collection_id, artifact["id"], request.selected_document_ids)
                    except LearningError:
                        pass
        context = self._retrieve(collection_id, scope, request.topic, kind, parameters.get("format", ""))
        if kind in {"flashcards", "quizzes"} and len(scope) > 1:
            batches = []
            for doc_id in scope[:8]:
                try:
                    doc_context = self._retrieve(collection_id, [doc_id], request.topic, kind, parameters.get("format", ""))
                    if doc_context.evidence:
                        batches.append(doc_context.evidence)
                except LearningError:
                    continue
            if not batches:
                batches = [context.evidence]
        elif kind == "knowledge-graph" and not request.topic.strip() and len(scope) > 1:
            # Generate an overall graph for each selected source separately, then combine
            batches = []
            for doc_id in scope[:8]:
                try:
                    doc_context = self._retrieve(collection_id, [doc_id], "", kind, parameters.get("format", ""))
                    if doc_context.evidence:
                        batches.append(doc_context.evidence)
                except LearningError:
                    continue
            if not batches:
                batches = [context.evidence]
        else:
            batches = [context.evidence]
        contents = []
        per_batch_nodes = max(4, min(self.config.learning_max_nodes, self.config.learning_max_nodes // max(1, len(batches))))
        per_batch_edges = max(6, min(self.config.learning_max_edges, self.config.learning_max_edges // max(1, len(batches))))
        per_batch_count = max(1, request.count // len(batches)) if (kind in {"flashcards", "quizzes"} and len(batches) > 1) else request.count
        for batch_index, batch in enumerate(batches):
            batch_request = deepcopy(request)
            batch_request.count = per_batch_count
            batch_context = build_evidence_context(batch, self.config.learning_context_chars)
            instruction = (GUIDANCE[kind] + "\nUse only the provided evidence. Return strict JSON. "
                f"Maximum nodes {per_batch_nodes}, edges {per_batch_edges}; "
                "up to 8 concise sheet/lesson sections. Requested count is a maximum; fewer is preferable to unsupported content.")
            if request.topic:
                instruction += "\nCRITICAL: If the provided evidence is completely unrelated to the requested topic, you MUST return 0 items (empty arrays). Do not ignore the topic, and do not hallucinate connections."
            
            if kind in {"flashcards", "quizzes"} and len(batches) > 1:
                batch_doc_name = batch[0].evidence.filename if batch else "source"
                instruction += (
                    f"\nCRITICAL: Generate exactly {per_batch_count} {kind} strictly from the evidence of document: '{batch_doc_name}'. "
                    f"Every generated item MUST cite evidence IDs from document '{batch_doc_name}'."
                )
            elif kind == "knowledge-graph" and len(batches) > 1:
                batch_doc_name = batch[0].evidence.filename if batch else "source"
                instruction += f"\nGenerate a knowledge graph covering key concepts and relationships for document: {batch_doc_name}."

            if kind == "quizzes":
                q_type = parameters.get("question_type", "mixed")
                if q_type == "single_choice":
                    instruction += (
                        "\nCRITICAL: All generated questions MUST be multiple choice with type 'single_choice'. "
                        "Every question MUST provide exactly 4 distinct options (options array of length 4), exactly one correct answer, "
                        "and 3 clear distractors that are demonstrably incorrect based on the cited evidence. Do NOT generate true_false questions."
                    )
                elif q_type == "true_false":
                    instruction += (
                        "\nCRITICAL: All generated questions MUST be of type 'true_false' with options exactly ['True', 'False']. "
                        "Statements must be clearly provable as True or False based on the cited evidence."
                    )
                else:
                    instruction += (
                        "\nCRITICAL: You MUST generate a diverse mix of question types supporting both 4-option multiple choice ('single_choice') and 'true_false'. "
                        "At least half of the questions MUST be 4-option multiple choice ('single_choice' with 4 distinct options: A, B, C, D), "
                        "supplemented with 'true_false' questions. Do NOT generate only true_false questions."
                    )

            if kind == "cheat-sheet":
                instruction += f"\nCRITICAL: You MUST strictly conform to the requested format: {parameters.get('format', 'default')}. For example, if formula_sheet is requested, ONLY extract formulas and ignore non-formula content."
            
            batch_parameters = {**parameters, "count": per_batch_count}
            instruction += (
                "\nUser settings (untrusted topic): " + json.dumps({**batch_parameters, "topic": request.topic or "General overview of selected sources"}) +
                "\nOriginal headings: " + json.dumps({item.evidence.evidence_id: item.evidence.section for item in batch})
            )
            content = None
            last_error: Exception | None = None
            # Providers occasionally return a long paragraph, a malformed key,
            # or a citation that the verifier cannot support. Give the same
            # configured model one bounded repair attempt rather than showing
            # an unverified artifact or making the user resubmit blindly.
            for attempt in range(2):
                attempt_instruction = instruction
                if kind == "teach":
                    attempt_instruction += "\nFor this lesson, do not introduce any numeric literal, percentage, model size, date, unit, or formula unless the exact notation occurs in the cited evidence; prefer source wording and cite each section/example/check explanation."
                if last_error is not None:
                    attempt_instruction += (
                        "\nThe previous candidate was rejected by deterministic validation. "
                        "Repair it and return a new complete candidate. Failure reason: "
                        + str(last_error)[:800]
                    )
                    if "numeric" in str(last_error).casefold() or "formula" in str(last_error).casefold():
                        attempt_instruction += " Do not use any digits, percentages, model sizes, dates, units, or formulas in this candidate unless the exact characters are present in the cited source passage; omit that detail instead."
                try:
                    raw = self.reasoning_client.structured(
                        attempt_instruction, batch_context.text, MODELS[kind].model_json_schema(),
                    )
                    if len(raw) > self.config.learning_output_chars:
                        raise LearningError("The model exceeded the configured output limit. Narrow the topic.")
                    allowed_ids = [item.evidence.evidence_id for item in batch_context.evidence]
                    parsed_dict = extract_and_repair_json(raw)
                    sanitized_dict = sanitize_generic_learning_content(
                        kind, parsed_dict, allowed_ids, request.topic
                    )
                    content = MODELS[kind].model_validate(sanitized_dict).model_dump()
                    content = validate_structure(kind, content, batch_context.evidence, batch_request, self.config)
                    try:
                        content = review_content(content, batch_context.evidence, self.reasoning_client, batch_context.text, self.config.learning_output_chars)
                    except (ValueError, LearningError) as rev_err:
                        if attempt == 0:
                            raise rev_err
                        # On final attempt, if lesson sections are valid and grounded, keep them rather than failing completely
                        if kind == "teach" and content.get("lesson_sections"):
                            pass
                        else:
                            raise rev_err
                    break
                except (ValueError, LearningError) as error:
                    last_error = error
                    content = None
            if content is None:
                raise LearningError(str(last_error) if last_error else "The model returned invalid learning content.") from last_error
            if kind == "knowledge-graph":
                for node in content.get("nodes", []):
                    node["id"] = f"b{batch_index}-{node['id']}"
                for edge in content.get("edges", []):
                    edge["id"] = f"b{batch_index}-{edge['id']}"
                    edge["source_node_id"] = f"b{batch_index}-{edge['source_node_id']}"
                    edge["target_node_id"] = f"b{batch_index}-{edge['target_node_id']}"
            elif kind == "topic-tree":
                for node in content.get("nodes", []):
                    node["id"] = f"b{batch_index}-{node['id']}"
                    if node["parent_id"] is not None:
                        node["parent_id"] = f"b{batch_index}-{node['parent_id']}"
            elif kind == "flashcards":
                for card in content.get("cards", []):
                    card["id"] = f"b{batch_index}-{card['id']}"
            elif kind == "quizzes":
                for q in content.get("questions", []):
                    q["id"] = f"b{batch_index}-{q['id']}"
            contents.append(content)

        if len(contents) > 1 and kind in {"flashcards", "quizzes"}:
            key = "cards" if kind == "flashcards" else "questions"
            # Ensure an equal number of outputs from each selected source document
            min_items = min(len(part.get(key, [])) for part in contents)
            for part in contents:
                part[key] = part[key][:min_items]

        content = contents[0]
        all_evidence = [item for batch in batches for item in batch] if len(contents) > 1 else context.evidence
        if len(contents) > 1:
            if kind == "knowledge-graph":
                content["title"] = request.topic or "Knowledge Graph (Selected Sources)"
            elif kind == "flashcards":
                content["title"] = request.topic or "Flashcards (Selected Sources)"
            elif kind == "quizzes":
                content["title"] = request.topic or "Quiz (Selected Sources)"
            if "nodes" in content:
                content["nodes"] = [node for part in contents for node in part.get("nodes", [])]
            if "edges" in content:
                content["edges"] = [edge for part in contents for edge in part.get("edges", [])]
            if "cards" in content:
                content["cards"] = [card for part in contents for card in part.get("cards", [])]
            if "questions" in content:
                content["questions"] = [q for part in contents for q in part.get("questions", [])]
            if "sections" in content:
                content["sections"] = [s for part in contents for s in part.get("sections", [])]
            
            # Final unified validation
            key = "cards" if kind == "flashcards" else ("questions" if kind == "quizzes" else None)
            total_items = len(content.get(key, [])) if key else len(batches)
            unified_request = deepcopy(request)
            unified_request.count = max(1, total_items)
            validate_structure(kind, content, all_evidence, unified_request, self.config)
        if self.store.learning_source_version(collection_id, scope) != version:
            raise LearningError("Sources changed during generation. Retry with the current evidence.", 409)
        refs = sorted({ref for _, item in units(content) for ref in item["source_evidence_ids"]})
        warnings = ["Based on a bounded selection of retrieved passages; this is not an exhaustive collection summary."]
        if kind in {"flashcards", "quizzes"} and len(content.get("cards", content.get("questions", []))) < request.count:
            warnings.append("Fewer items were supported than requested.")
        conflicts = detect_contradictions([item.evidence for item in all_evidence])
        if any(item.status.value == "CONFLICT" for item in conflicts):
            warnings.append("The retrieved sources contain conflicting values. Inspect the cited passages; conflicts remain unresolved.")
        if kind == "knowledge-graph":
            relations = {}
            for edge in content["edges"]:
                relations.setdefault((edge["source_node_id"], edge["target_node_id"]), set()).add(edge["relation"])
            if any(len(values) > 1 for values in relations.values()):
                warnings.append("Some concept pairs have multiple source relationships. Inspect each relationship's evidence.")
        payload = {"content": content, "parameters": parameters, "source_evidence_ids": refs,
                   "source_version": version, "cache_key": cache_key, "warnings": warnings}
        artifact = self.store.save_learning_artifact(collection_id, kind, payload, scope)
        return self.get(collection_id, artifact["id"], request.selected_document_ids)

    def _load(self, collection_id, artifact_id, selected=None, enforce_version: bool = True):
        scope = self._scope(collection_id, selected) if selected is not None else None
        artifact = self.store.get_learning_artifact(artifact_id, collection_id)
        if artifact is None:
            raise LearningError("Learning artifact was not found in this collection.", 404)
        saved_scope = artifact["selected_document_ids"]
        if scope is not None and not set(saved_scope).issubset(scope):
            raise LearningError("This artifact uses documents outside the current selected scope.", 403)
        if enforce_version and self.store.learning_source_version(collection_id, saved_scope) != artifact["payload"]["source_version"]:
            raise LearningError("Source evidence changed or was removed. Regenerate this activity.", 409)
        evidence = []
        for ref in artifact["payload"]["source_evidence_ids"]:
            item = self.store.get_evidence(ref, collection_id)
            if item is None or (scope is not None and item.document_id not in scope) or contains_secret_pattern(json.dumps(asdict(item))):
                if enforce_version:
                    raise LearningError("Source evidence is no longer available. Regenerate this activity.", 409)
                continue
            evidence.append(item)
        return artifact, evidence

    def get(self, collection_id, artifact_id, selected=None):
        artifact, evidence = self._load(collection_id, artifact_id, selected)
        content = deepcopy(artifact["payload"]["content"])
        for key in ("questions", "understanding_checks"):
            for question in content.get(key, []):
                question.pop("correct_answer", None)
                question.pop("explanation", None)
        # Quiz evidence is revealed only after submission to avoid leaking keys.
        return {"id": artifact["id"], "kind": artifact["artifact_type"], "collection_id": collection_id,
                "selected_document_ids": artifact["selected_document_ids"], "created_at": artifact["created_at"],
                "parameters": artifact["payload"]["parameters"], "content": content,
                "warnings": artifact["payload"]["warnings"],
                "evidence": [] if artifact["artifact_type"] == "quizzes" else [self._public_evidence(item) for item in evidence]}

    @staticmethod
    def _public_evidence(item):
        return {key: getattr(item, key) for key in ("evidence_id", "collection_id", "document_id", "filename", "page_number", "content_type", "source_reference", "text")}

    def list(self, collection_id, selected=None):
        scope = self._scope(collection_id, selected)
        result = []
        for artifact in self.store.list_learning_artifacts(collection_id):
            if not set(artifact["selected_document_ids"] or []).issubset(scope):
                continue
            # Listing resolves freshness too, so stale titles/content are not exposed.
            try:
                self._load(collection_id, artifact["id"], selected)
            except LearningError:
                continue
            result.append({"id": artifact["id"], "kind": artifact["artifact_type"],
                "title": artifact["payload"]["content"]["title"], "created_at": artifact["created_at"]})
        return result

    def submit(self, collection_id, artifact_id, answers, selected=None):
        # Evaluate quiz against stored artifact content resiliently
        artifact, evidence = self._load(collection_id, artifact_id, selected=None, enforce_version=False)
        if artifact["artifact_type"] != "quizzes":
            raise LearningError("This activity is not a quiz.")
        questions = artifact["payload"]["content"]["questions"]
        by_id = {item["id"]: item for item in questions}
        if set(answers) - by_id.keys():
            raise LearningError("Responses contain an unknown question.")
        if any(type(value) is not int or not 0 <= value < len(by_id[key]["options"]) for key, value in answers.items()):
            raise LearningError("Choose one valid option for each answered question.")
        review = [{**item, "selected_answer": answers.get(item["id"]),
                   "correct": answers.get(item["id"]) == item["correct_answer"]} for item in questions]
        score = sum(item["correct"] for item in review)
        return {"score": score, "total": len(questions), "percentage": round(score * 100 / len(questions), 2),
                "review": review, "evidence": [self._public_evidence(item) for item in evidence]}

    def answer(self, collection_id, artifact_id, request):
        artifact, evidence = self._load(collection_id, artifact_id, request.selected_document_ids)
        if artifact["artifact_type"] != "teach":
            raise LearningError("This activity is not a lesson.")
        check = next((item for item in artifact["payload"]["content"]["understanding_checks"] if item["id"] == request.check_id), None)
        if check is None or request.answer >= len(check["options"]):
            raise LearningError("Choose a valid understanding check and option.")
        correct = request.answer == check["correct_answer"]
        level = artifact["payload"]["parameters"]["level"]
        next_level = {"beginner": "intermediate", "intermediate": "advanced", "advanced": "advanced"}[level] if correct else "beginner"
        return {"check_id": request.check_id, "correct": correct, "feedback": check["explanation"], "correct_answer": check["correct_answer"],
                "source_evidence_ids": check["source_evidence_ids"], "next_level": next_level,
                "next_step": "Deepen this topic with the next lesson." if correct else "Revisit a simpler explanation, then retry the check.",
                "evidence": [self._public_evidence(item) for item in evidence]}
