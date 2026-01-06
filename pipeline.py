from typing import List
import re

from sentence_transformers import SentenceTransformer
import numpy as np

from llm_inference import inference

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


def embed_sentences(sentences: List[str]):
    embeddings = model.encode(sentences)
    return embeddings


def generate_clarifications(
    query: str,
    sys_prompt: str,
    clarifictation_llm: str = "google/gemini-3-flash-preview",
):
    output_raw = inference(
        model_url=clarifictation_llm, content=query, system=sys_prompt
    )
    output_text = output_raw.text

    return output_text


def process_clarifications(
    clarifications: List[str],
    sys_prompt: str,
    target_llm: str = "google/gemini-3-flash-preview",
    n: int = 3,
):
    similarity_matrices = []
    similarity_eigenvalues = []
    for clarification in clarifications:
        outputs_raw = inference(
            model_url=target_llm, content=clarification, system=sys_prompt, n=n
        )
        outputs_text = [outputs_raw.text for outputs_raw in outputs_raw]
        outputs_embedded = embed_sentences(outputs_text)  # (n, embed_dim)
        similarity_matrix = compute_similarity_matrix(outputs_embedded)  # (n, n)

        sim_matrix_norm = 1 / outputs_embedded.shape[0] * similarity_matrix
        sim_eig_values = np.linalg.eigvalsh(sim_matrix_norm)
        similarity_matrices.append(similarity_matrix)
        similarity_eigenvalues.append(sim_eig_values)

    return similarity_matrices, similarity_eigenvalues


def compute_similarity_matrix(outputs_embedded: List[np.ndarray], gamma: float = 1.0):
    outputs_arr = np.asarray(outputs_embedded)[:, np.newaxis, :]  # (n, 1, embed_dim)
    outputs_arr_t = np.transpose(outputs_arr, (1, 0, 2))  # (1, n, embed_dim)
    element_wise_diff = outputs_arr - outputs_arr_t  # (n, n, embed_dim)
    sim_matrix = np.exp(
        -gamma * np.linalg.norm(element_wise_diff, axis=(0, 1))
    )  # (n, n)
    return sim_matrix


def parse_ambigqa_response(response_text: str, original_query: str) -> List[str]:
    text = response_text.strip()

    if "-Clarifications:" not in text:
        # Fallback: if the model failed to follow format, assume ambiguity
        print(f"Warning: Output format malformed. Returning original query.")
        return [original_query]

    clarification_section = text.split("-Clarifications:")[-1].strip()

    if "No clarification needed" in clarification_section:
        return [original_query]

    # extract individual clarifications
    pattern = r"-\d+\s+(.+)"
    matches = re.findall(pattern, clarification_section)

    if not matches:
        return [original_query]

    cleaned_clarifications = [m.strip() for m in matches]

    return cleaned_clarifications


def process_ambigQA(query: str, sys_prompt: str):
    raw_response = generate_clarifications(query, sys_prompt)

    W_clarifications = parse_ambigqa_response(raw_response, query)

    print(f"Raw Response:\n{raw_response}\n")
    print(f"Final List W: {W_clarifications}")

    return W_clarifications


def pipeline(query: str, sys_prompt_clarify: str, sys_prompt_answer: str):
    W_clarifications = process_ambigQA(query, sys_prompt_clarify)

    similarity_matrices, similarity_eigenvalues = process_clarifications(
        clarifications=W_clarifications,
        sys_prompt=sys_prompt_answer,
        n=3,
    )

    return similarity_matrices, similarity_eigenvalues
