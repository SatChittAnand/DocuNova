"""Strict generation contracts; collection/document/evidence remain shared models."""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Text = Annotated[str, Field(min_length=1, max_length=2000)]
Title = Annotated[str, Field(min_length=1, max_length=200)]
Refs = Annotated[list[str], Field(min_length=1, max_length=20)]
Kind = Literal["knowledge-graph", "topic-tree", "flashcards", "quizzes", "cheat-sheet", "teach"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class GenerateRequest(StrictModel):
    collection_id: str | None = None
    selected_document_ids: list[str] | None = Field(default=None, max_length=100)
    topic: str = Field(default="", max_length=240)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    level: Literal["beginner", "intermediate", "advanced"] = "beginner"
    count: int = Field(default=6, ge=1, le=30)
    format: Literal["quick_revision", "key_concepts", "formula_sheet", "comparison_table", "exam_outline", "last_minute"] = "quick_revision"
    question_type: Literal["mixed", "single_choice", "true_false"] = "mixed"
    regenerate: bool = False


class SubmitRequest(StrictModel):
    collection_id: str | None = None
    selected_document_ids: list[str] | None = Field(default=None, max_length=100)
    answers: dict[str, int] = Field(max_length=30)


class TeachAnswerRequest(StrictModel):
    collection_id: str | None = None
    selected_document_ids: list[str] | None = Field(default=None, max_length=100)
    check_id: str = Field(max_length=200)
    answer: int = Field(ge=0, le=10)


class Sourced(StrictModel):
    id: Title
    source_evidence_ids: Refs


class Section(Sourced):
    heading: Title
    content: Text


class GraphNode(Sourced):
    name: Title
    description: Text
    status: Literal["explicit", "inferred"]
    source_evidence_ids: list[str] = Field(default_factory=list, max_length=20)


class GraphEdge(Sourced):
    source_node_id: Title
    target_node_id: Title
    relation: Title
    status: Literal["explicit", "inferred"]
    source_evidence_ids: list[str] = Field(default_factory=list, max_length=20)


class Graph(StrictModel):
    title: Title
    nodes: list[GraphNode] = Field(max_length=100)
    edges: list[GraphEdge] = Field(max_length=200)


class TopicNode(Sourced):
    parent_id: str | None
    title: Title
    summary: Text
    inferred: bool
    source_evidence_ids: list[str] = Field(default_factory=list, max_length=20)


class TopicTree(StrictModel):
    title: Title
    nodes: list[TopicNode] = Field(max_length=100)


class Card(Sourced):
    question: Text
    answer: Text
    explanation: Text


class Deck(StrictModel):
    title: Title
    cards: list[Card] = Field(max_length=30)


class Question(Sourced):
    type: Literal["single_choice", "true_false"]
    question: Text
    options: list[Title] = Field(min_length=2, max_length=4)
    correct_answer: int = Field(ge=0, le=3)
    explanation: Text

    @model_validator(mode="after")
    def valid_options(self):
        if self.correct_answer >= len(self.options):
            raise ValueError("Invalid answer key")
        if len({option.casefold() for option in self.options}) != len(self.options):
            raise ValueError("Ambiguous duplicate options")
        if self.type == "true_false" and self.options != ["True", "False"]:
            raise ValueError("True/false options must be True, False")
        return self


class Quiz(StrictModel):
    title: Title
    questions: list[Question] = Field(max_length=20)


class Sheet(StrictModel):
    title: Title
    sections: list[Section] = Field(max_length=20)


class Lesson(StrictModel):
    title: Title
    learning_objectives: list[Title] = Field(min_length=1, max_length=5)
    lesson_sections: list[Section] = Field(max_length=8)
    examples: list[Section] = Field(max_length=4)
    understanding_checks: list[Question] = Field(min_length=1, max_length=3)


class Assessment(StrictModel):
    id: Title
    supported: bool
    unambiguous: bool
    # For every quiz/check option: is this an acceptable answer to the question?
    # This is independently derived by the reviewer, never copied from the key.
    option_correct: list[bool] = Field(max_length=4)


class Review(StrictModel):
    assessments: list[Assessment] = Field(max_length=200)


MODELS = {"knowledge-graph": Graph, "topic-tree": TopicTree, "flashcards": Deck,
          "quizzes": Quiz, "cheat-sheet": Sheet, "teach": Lesson}
