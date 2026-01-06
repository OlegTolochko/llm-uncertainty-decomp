from dataclasses import dataclass, field
from typing import List, Optional
import re
import json

from sentence_transformers import SentenceTransformer
import numpy as np

from llm_inference import inference
from ambigqa import AmbigQAItem, load_ambigqa, filter_ambiguous, filter_unambiguous
from sys_prompts import ambigqa_clarification_sys_prompt, ambigqa_target_sys_prompt

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


@dataclass
class PipelineResult:
    """Result of running the uncertainty pipeline on a single item."""

    query: str
    clarifications: List[str]
    model_answers: List[List[str]]  # [clarification][sample]
    aleatoric: float
    epistemic: float

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
            "aleatoric": self.aleatoric,
            "epistemic": self.epistemic,
        }


def embed_sentences(sentences: List[str]):
    embeddings = model.encode(sentences)
    return embeddings


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
    target_llm: str = "google/gemini-3-flash-preview",
    m: int = 3,
    temperature: float = 0.0,
):
    all_outputs_embedded = []
    all_outputs_text = []
    similarity_matrices = []
    similarity_eigenvalues = []
    for clarification in clarifications:
        user_content = f"Task\nQuestion: {clarification}"

        # n=m first, fall back to multiple calls if provider doesn't support n>1
        outputs_raw = inference(
            model_url=target_llm,
            content=user_content,
            system=sys_prompt,
            n=m,
            temperature=temperature,
        )

        if isinstance(outputs_raw, list):
            outputs_text = [r.text for r in outputs_raw]
        else:
            # Provider returned single result despite n>1 request
            outputs_text = [outputs_raw.text]
            for _ in range(m - 1):
                additional = inference(
                    model_url=target_llm,
                    content=user_content,
                    system=sys_prompt,
                    n=1,
                    temperature=temperature,
                )
                outputs_text.append(additional.text)

        all_outputs_text.append(outputs_text)
        outputs_embedded = embed_sentences(outputs_text)  # (m, embed_dim)
        similarity_matrix = compute_similarity_matrix(outputs_embedded)  # (m, m)

        sim_matrix_norm = 1 / outputs_embedded.shape[0] * similarity_matrix
        sim_eig_values = np.linalg.eigvalsh(sim_matrix_norm)
        sim_eig_values = np.maximum(sim_eig_values, 1e-10)

        all_outputs_embedded.append(outputs_embedded)
        similarity_matrices.append(similarity_matrix)
        similarity_eigenvalues.append(sim_eig_values)

    return (
        all_outputs_embedded,
        all_outputs_text,
        similarity_matrices,
        similarity_eigenvalues,
    )


def compute_similarity_matrix(outputs_embedded: List[np.ndarray], gamma: float = 1.0):
    outputs_arr = np.asarray(outputs_embedded)[:, np.newaxis, :]  # (m, 1, embed_dim)
    outputs_arr_t = np.transpose(outputs_arr, (1, 0, 2))  # (1, m, embed_dim)
    element_wise_diff = outputs_arr - outputs_arr_t  # (m, m, embed_dim)
    dist_sq = np.sum(element_wise_diff**2, axis=2)
    sim_matrix = np.exp(-gamma * dist_sq)  # (m, m)
    return sim_matrix


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


def process_outer_loop(all_outputs_embedded: List[np.ndarray], gamma: float = 1.0):
    flattened_embeddings = np.vstack(all_outputs_embedded)  # (n*m, embed_dim)
    nm = flattened_embeddings.shape[0]

    K_out = compute_similarity_matrix(flattened_embeddings, gamma=gamma)
    K_out_norm = (1 / nm) * K_out
    outer_eigenvalues = np.linalg.eigvalsh(K_out_norm)
    outer_eigenvalues = np.maximum(outer_eigenvalues, 1e-10)

    return outer_eigenvalues


def compute_uncertainties(
    inner_eigenvalues: List[np.ndarray], outer_eigenvalues: np.ndarray, n: int
):
    aleatoric = (1 / n) * (
        np.sum(inner_eigenvalues * np.log(inner_eigenvalues))
    ) - np.sum(outer_eigenvalues * np.log(outer_eigenvalues))
    epistemic = -(1 / n) * np.sum(inner_eigenvalues * np.log(inner_eigenvalues))
    return aleatoric, epistemic


def pipeline(query: str, sys_prompt_clarify: str, sys_prompt_answer: str):
    W_clarifications = process_ambigQA(query, sys_prompt_clarify)

    all_outputs_embedded, all_outputs_text, inner_matrices, inner_eigenvalues = (
        process_clarifications(
            clarifications=W_clarifications,
            sys_prompt=sys_prompt_answer,
            m=3,
            target_llm="google/gemini-3-flash-preview",
        )
    )
    outer_eigenvalues = process_outer_loop(all_outputs_embedded)
    n = len(all_outputs_embedded)
    aleatoric, epistemic = compute_uncertainties(
        inner_eigenvalues=inner_eigenvalues, outer_eigenvalues=outer_eigenvalues, n=n
    )
    print(f"Aleatoric Uncertainty: {aleatoric}")
    print(f"Epistemic Uncertainty: {epistemic}")

    return PipelineResult(
        query=query,
        clarifications=W_clarifications,
        model_answers=all_outputs_text,
        aleatoric=aleatoric,
        epistemic=epistemic,
    )


def pipeline_on_ambigqa_item(
    item: AmbigQAItem,
    sys_prompt_clarify: str,
    sys_prompt_answer: str,
    clarification_llm: str = "google/gemini-3-flash-preview",
    target_llm: str = "google/gemini-3-flash-preview",
    m: int = 3,
    temperature: float = 0.0,
) -> PipelineResult:
    """Run the uncertainty pipeline on a single AmbigQA item."""

    # Step 1: Generate clarifications from the original question
    W_clarifications = process_ambigQA(
        item.question, sys_prompt_clarify, clarification_llm=clarification_llm
    )

    # Step 2: Get m samples for each clarification
    all_outputs_embedded, all_outputs_text, _, inner_eigenvalues = (
        process_clarifications(
            clarifications=W_clarifications,
            sys_prompt=sys_prompt_answer,
            target_llm=target_llm,
            m=m,
            temperature=temperature,
        )
    )

    # Step 3: Compute outer similarity
    outer_eigenvalues = process_outer_loop(all_outputs_embedded)

    # Step 4: Compute uncertainties
    n = len(all_outputs_embedded)
    aleatoric, epistemic = compute_uncertainties(
        inner_eigenvalues=inner_eigenvalues, outer_eigenvalues=outer_eigenvalues, n=n
    )

    return PipelineResult(
        query=item.question,
        clarifications=W_clarifications,
        model_answers=all_outputs_text,
        aleatoric=aleatoric,
        epistemic=epistemic,
        ground_truth_answers=item.all_answers,
        is_ambiguous=item.is_ambiguous,
        item_id=item.id,
    )


def run_ambigqa_evaluation(
    split: str = "dev",
    start_index: int = 0,
    limit: Optional[int] = 10,
    sys_prompt_clarify: str = "",
    sys_prompt_answer: str = "Answer concisely.",
    output_file: Optional[str] = None,
    only_ambiguous: bool = False,
    only_unambiguous: bool = False,
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

    Returns:
        List of PipelineResult objects
    """
    # Load dataset
    data = load_ambigqa(split)

    if only_ambiguous:
        data = filter_ambiguous(data)
    elif only_unambiguous:
        data = filter_unambiguous(data)

    # Apply start_index and limit
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
            )
            results.append(result)

            print(f"  Ambiguous: {result.is_ambiguous}")
            print(f"  Clarifications: {len(result.clarifications)}")
            print(f"  Aleatoric: {result.aleatoric:.4f}")
            print(f"  Epistemic: {result.epistemic:.4f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Save results if output file specified
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
        print(f"Model Answers: {results[0].model_answers}")

        if ambig_results:
            avg_aleatoric = np.mean([r.aleatoric for r in ambig_results])
            avg_epistemic = np.mean([r.epistemic for r in ambig_results])
            print(f"Ambiguous ({len(ambig_results)} items):")
            print(f"  Avg Aleatoric: {avg_aleatoric:.4f}")
            print(f"  Avg Epistemic: {avg_epistemic:.4f}")

        if unambig_results:
            avg_aleatoric = np.mean([r.aleatoric for r in unambig_results])
            avg_epistemic = np.mean([r.epistemic for r in unambig_results])
            print(f"Unambiguous ({len(unambig_results)} items):")
            print(f"  Avg Aleatoric: {avg_aleatoric:.4f}")
            print(f"  Avg Epistemic: {avg_epistemic:.4f}")

    return results


if __name__ == "__main__":
    sys_prompt_clarify = ambigqa_clarification_sys_prompt
    sys_prompt_answer = ambigqa_target_sys_prompt
    run_ambigqa_evaluation(
        split="dev",
        limit=5,
        start_index=1,
        sys_prompt_clarify=sys_prompt_clarify,
        sys_prompt_answer=sys_prompt_answer,
    )
