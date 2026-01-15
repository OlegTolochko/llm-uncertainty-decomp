from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm_inference import inference
from ambigqa import AmbigQAItem, load_ambigqa, filter_ambiguous, filter_unambiguous
from sys_prompts import ambigqa_clarification_sys_prompt, ambigqa_target_sys_prompt


@dataclass
class PipelineResult:
    """Result of running the uncertainty pipeline on a single item."""

    query: str
    clarifications: List[str]
    model_answers: List[List[str]]  # [clarification][sample]

    ground_truth_answers: List[str] = field(default_factory=list)
    is_ambiguous: bool = False
    item_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "query": self.query,
            "is_ambiguous": self.is_ambiguous,
            "clarifications": self.clarifications,
            "model_answers": self.model_answers,
            "ground_truth_answers": self.ground_truth_answers,
        }


def generate_clarifications(
    query: str,
    sys_prompt: str,
    clarification_llm: str = "google/gemini-3-flash-preview",
):
    user_content = f"Question: {query}"
    output_raw = inference(
        model_url=clarification_llm, content=user_content, system=sys_prompt
    )
    output_text = output_raw.text

    return output_text


def process_clarifications(
    clarifications: List[str],
    sys_prompt: str,
    target_llm: str,
    m: int,
    temperature: float,
    max_workers: int = 16,
) -> List[List[str]]:
    """Generate m answer samples for each clarification.
    
    Returns:
        all_outputs_text: List of lists, shape [n_clarifications][m_samples]
    """
    n = len(clarifications)

    def call_one(i: int, j: int) -> Tuple[int, int, str]:
        user_content = f"Task\nQuestion: {clarifications[i]}"
        r = inference(
            model_url=target_llm,
            content=user_content,
            system=sys_prompt,
            n=1,
            temperature=temperature,
        )
        return i, j, r.text

    all_outputs_text: List[List[str]] = [[""] * m for _ in range(n)]

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(call_one, i, j) for i in range(n) for j in range(m)]
        for fut in as_completed(futures):
            i, j, text = fut.result()
            all_outputs_text[i][j] = text

    return all_outputs_text


def parse_ambigqa_response(response_text: str, original_query: str) -> List[str]:
    text = response_text.strip()

    clarifications_marker = None
    for marker in ["—Clarifications:", "-Clarifications:", "Clarifications:"]:
        if marker in text:
            clarifications_marker = marker
            break

    if clarifications_marker is None:
        print(f"Warning: Output format malformed. Returning original query.")
        return [original_query]

    clarification_section = text.split(clarifications_marker)[-1].strip()

    if "No clarification needed" in clarification_section.lower():
        return [original_query]

    # Extract individual clarifications: match "-1 ..." or "—1 ..."
    pattern = r"^[-—]\d+\s+(.+)$"
    matches = re.findall(pattern, clarification_section, re.MULTILINE)

    if not matches:
        return [original_query]

    cleaned_clarifications = [m.strip() for m in matches if m.strip()]
    cleaned_clarifications = cleaned_clarifications[
        :10
    ]  # Limit to first 10 clarifications
    return cleaned_clarifications


def process_ambigQA(
    query: str,
    sys_prompt: str,
    clarification_llm: str = "google/gemini-3-flash-preview",
):
    raw_response = generate_clarifications(
        query, sys_prompt, clarification_llm=clarification_llm
    )

    W_clarifications = parse_ambigqa_response(raw_response, query)

    print(f"Raw Response:\n{raw_response}\n")
    print(f"Final List W: {W_clarifications}")

    return W_clarifications


def pipeline(query: str, sys_prompt_clarify: str, sys_prompt_answer: str):
    """Simple pipeline for a single query."""
    W_clarifications = process_ambigQA(query, sys_prompt_clarify)

    all_outputs_text = process_clarifications(
        clarifications=W_clarifications,
        sys_prompt=sys_prompt_answer,
        m=3,
        target_llm="google/gemini-3-flash-preview",
        temperature=0.5,
    )

    return PipelineResult(
        query=query,
        clarifications=W_clarifications,
        model_answers=all_outputs_text,
    )


def pipeline_on_ambigqa_item(
    item: AmbigQAItem,
    sys_prompt_clarify: str,
    sys_prompt_answer: str,
    clarification_llm: str = "google/gemini-3-flash-preview",
    target_llm: str = "google/gemini-3-flash-preview",
    m: int = 3,
    temperature: float = 0.5,
) -> PipelineResult:
    """Run the pipeline on a single AmbigQA item."""

    # 1: Generate clarifications from the original question
    W_clarifications = process_ambigQA(
        item.question, sys_prompt_clarify, clarification_llm=clarification_llm
    )

    # 2: Get m samples for each clarification
    all_outputs_text = process_clarifications(
        clarifications=W_clarifications,
        sys_prompt=sys_prompt_answer,
        target_llm=target_llm,
        m=m,
        temperature=temperature,
    )

    return PipelineResult(
        query=item.question,
        clarifications=W_clarifications,
        model_answers=all_outputs_text,
        ground_truth_answers=item.all_answers,
        is_ambiguous=item.is_ambiguous,
        item_id=item.id,
    )


def run_ambigqa_evaluation(
    sys_prompt_clarify: str,
    sys_prompt_answer: str,
    split: str = "dev",
    start_index: int = 0,
    limit: Optional[int] = 10,
    output_file: Optional[str] = None,
    only_ambiguous: bool = False,
    only_unambiguous: bool = False,
    temperature: float = 0.5,
    m: int = 10,
    target_llm: str = "google/gemini-3-flash-preview",
    clarification_llm: str = "google/gemini-3-flash-preview",
) -> List[PipelineResult]:
    """Run the pipeline on AmbigQA dataset and collect results.

    Args:
        split: Dataset split ("dev" or "train")
        start_index: Starting index in the dataset (0-based)
        limit: Max items to process from start_index (None for all remaining)
        sys_prompt_clarify: System prompt for clarification generation
        sys_prompt_answer: System prompt for answer generation
        output_file: Path to save results as JSON (optional)
        only_ambiguous: Filter to only ambiguous questions
        only_unambiguous: Filter to only unambiguous questions
        temperature: Temperature for the model
        m: Number of samples to generate for each clarification

    Returns:
        List of PipelineResult objects
    """
    data = load_ambigqa(split)

    if only_ambiguous:
        data = filter_ambiguous(data)
    elif only_unambiguous:
        data = filter_unambiguous(data)

    if limit is not None:
        data = data[start_index : start_index + limit]
    else:
        data = data[start_index:]

    print(
        f"Processing {len(data)} items from {split} split (starting at index {start_index})..."
    )

    results = []
    for i, item in enumerate(data):
        print(f"\n[{i + 1}/{len(data)}] Processing: {item.question[:60]}...")

        try:
            result = pipeline_on_ambigqa_item(
                item=item,
                sys_prompt_clarify=sys_prompt_clarify,
                sys_prompt_answer=sys_prompt_answer,
                temperature=temperature,
                m=m,
                target_llm=target_llm,
                clarification_llm=clarification_llm,
            )
            results.append(result)

            print(f"  Ambiguous: {result.is_ambiguous}")
            print(f"  Clarifications: {len(result.clarifications)}")
            print(f"  Answers per clarification: {len(result.model_answers[0]) if result.model_answers else 0}")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Saves results if output file is provided
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_file}")

    if results:
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        ambig_results = [r for r in results if r.is_ambiguous]
        unambig_results = [r for r in results if not r.is_ambiguous]

        print(f"Ambiguous: {len(ambig_results)} items")
        print(f"Unambiguous: {len(unambig_results)} items")
        print(f"Total: {len(results)} items")

    return results
