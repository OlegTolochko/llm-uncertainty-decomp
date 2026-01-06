"""AmbigQA dataset loader.

Each question can have:
- singleAnswer: One unambiguous answer (list of acceptable answer strings)
- multipleQAs: Multiple disambiguated question-answer pairs

Data files expected at:
- data/dev.json
- data/train.json
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Literal, Optional, Union


@dataclass
class QAPair:
    """A disambiguated question with its answer(s)."""

    question: str
    answers: List[str]  # Multiple acceptable answer strings


@dataclass
class AmbigQAItem:
    """A single item from the AmbigQA dataset."""

    id: str
    question: str  # Original (possibly ambiguous) question
    nq_answer: List[str]  # Original Natural Questions answer(s)
    nq_doc_title: str

    annotation_type: Literal["singleAnswer", "multipleQAs"]

    # For singleAnswer: list of acceptable answers
    # For multipleQAs: empty (use qa_pairs instead)
    single_answers: List[str] = field(default_factory=list)

    # For multipleQAs: list of disambiguated QA pairs
    # For singleAnswer: empty
    qa_pairs: List[QAPair] = field(default_factory=list)

    @property
    def is_ambiguous(self) -> bool:
        """Whether the question has multiple interpretations."""
        return self.annotation_type == "multipleQAs"

    @property
    def all_answers(self) -> List[str]:
        """All acceptable answers (flattened from single_answers or qa_pairs)."""
        if self.annotation_type == "singleAnswer":
            return self.single_answers
        else:
            answers = []
            for qap in self.qa_pairs:
                answers.extend(qap.answers)
            return list(set(answers))  # Deduplicate

    def __repr__(self) -> str:
        ans_preview = self.all_answers[:2]
        return (
            f"AmbigQAItem(id={self.id!r}, "
            f"question={self.question[:50]!r}..., "
            f"type={self.annotation_type}, "
            f"answers={ans_preview})"
        )


def _parse_item(raw: dict) -> AmbigQAItem:
    """Parse a raw JSON item into an AmbigQAItem."""
    annotations = raw.get("annotations", [])

    # Default to singleAnswer if no annotations
    if not annotations:
        return AmbigQAItem(
            id=raw["id"],
            question=raw["question"],
            nq_answer=raw.get("nq_answer", []),
            nq_doc_title=raw.get("nq_doc_title", ""),
            annotation_type="singleAnswer",
            single_answers=raw.get("nq_answer", []),
        )

    ann = annotations[0]
    ann_type = ann.get("type", "singleAnswer")

    if ann_type == "singleAnswer":
        return AmbigQAItem(
            id=raw["id"],
            question=raw["question"],
            nq_answer=raw.get("nq_answer", []),
            nq_doc_title=raw.get("nq_doc_title", ""),
            annotation_type="singleAnswer",
            single_answers=ann.get("answer", []),
        )
    else:  # multipleQAs
        qa_pairs = [
            QAPair(question=qap["question"], answers=qap["answer"])
            for qap in ann.get("qaPairs", [])
        ]
        return AmbigQAItem(
            id=raw["id"],
            question=raw["question"],
            nq_answer=raw.get("nq_answer", []),
            nq_doc_title=raw.get("nq_doc_title", ""),
            annotation_type="multipleQAs",
            qa_pairs=qa_pairs,
        )


def load_ambigqa(
    split: Literal["dev", "train"] = "dev",
    data_dir: Union[str, Path] = "data",
    limit: Optional[int] = None,
) -> List[AmbigQAItem]:
    """Load the AmbigQA dataset.

    Args:
        split: Which split to load ("dev" or "train")
        data_dir: Directory containing the JSON files
        limit: Maximum number of items to load (None for all)

    Returns:
        List of AmbigQAItem objects
    """
    path = Path(data_dir) / f"{split}.json"

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if limit is not None:
        raw_data = raw_data[:limit]

    return [_parse_item(item) for item in raw_data]


def iter_ambigqa(
    split: Literal["dev", "train"] = "dev",
    data_dir: Union[str, Path] = "data",
) -> Iterator[AmbigQAItem]:
    """Iterate over the AmbigQA dataset (memory-efficient for large files).

    Args:
        split: Which split to load ("dev" or "train")
        data_dir: Directory containing the JSON files

    Yields:
        AmbigQAItem objects one at a time
    """
    path = Path(data_dir) / f"{split}.json"

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    for item in raw_data:
        yield _parse_item(item)


def filter_ambiguous(items: List[AmbigQAItem]) -> List[AmbigQAItem]:
    """Filter to only ambiguous questions (multipleQAs)."""
    return [item for item in items if item.is_ambiguous]


def filter_unambiguous(items: List[AmbigQAItem]) -> List[AmbigQAItem]:
    """Filter to only unambiguous questions (singleAnswer)."""
    return [item for item in items if not item.is_ambiguous]


def dataset_stats(items: List[AmbigQAItem]) -> dict:
    """Get basic statistics about the dataset."""
    ambig = sum(1 for item in items if item.is_ambiguous)
    return {
        "total": len(items),
        "ambiguous": ambig,
        "unambiguous": len(items) - ambig,
        "ambiguous_pct": round(100 * ambig / len(items), 1) if items else 0,
    }


if __name__ == "__main__":
    data = load_ambigqa("dev", limit=10)

    print(f"Loaded {len(data)} items")
    print(f"Stats: {dataset_stats(data)}")

    print("\n--- Example unambiguous question ---")
    for item in data:
        if not item.is_ambiguous:
            print(f"Q: {item.question}")
            print(f"A: {item.single_answers}")
            break

    print("\n--- Example ambiguous question ---")
    for item in data:
        if item.is_ambiguous:
            print(f"Q: {item.question}")
            print("Disambiguations:")
            for qap in item.qa_pairs:
                print(f"  - {qap.question}")
                print(f"    -> {qap.answers}")
            break
