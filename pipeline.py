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
    m: int = 3,
):
    all_outputs_embedded = []
    similarity_matrices = []
    similarity_eigenvalues = []
    for clarification in clarifications:
        outputs_raw = inference(
            model_url=target_llm, content=clarification, system=sys_prompt, n=m
        )
        outputs_text = [outputs_raw.text for outputs_raw in outputs_raw]
        outputs_embedded = embed_sentences(outputs_text)  # (m, embed_dim)
        similarity_matrix = compute_similarity_matrix(outputs_embedded)  # (m, m)

        sim_matrix_norm = 1 / outputs_embedded.shape[0] * similarity_matrix
        sim_eig_values = np.linalg.eigvalsh(sim_matrix_norm)

        all_outputs_embedded.append(outputs_embedded)
        similarity_matrices.append(similarity_matrix)
        similarity_eigenvalues.append(sim_eig_values)

    return all_outputs_embedded, similarity_matrices, similarity_eigenvalues


def compute_similarity_matrix(outputs_embedded: List[np.ndarray], gamma: float = 1.0):
    outputs_arr = np.asarray(outputs_embedded)[:, np.newaxis, :]  # (m, 1, embed_dim)
    outputs_arr_t = np.transpose(outputs_arr, (1, 0, 2))  # (1, m, embed_dim)
    element_wise_diff = outputs_arr - outputs_arr_t  # (m, m, embed_dim)
    dist_sq = np.sum(element_wise_diff**2, axis=2)
    sim_matrix = np.exp(-gamma * dist_sq)  # (m, m)
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

    all_outputs_embedded, inner_matrices, inner_eigenvalues = process_clarifications(
        clarifications=W_clarifications,
        sys_prompt=sys_prompt_answer,
        m=3,
        target_llm="google/gemini-3-flash-preview",
    )
    outer_eigenvalues = process_outer_loop(all_outputs_embedded)
    n = len(all_outputs_embedded)
    aleatoric, epistemic = compute_uncertainties(
        inner_eigenvalues=inner_eigenvalues, outer_eigenvalues=outer_eigenvalues, n=n
    )
    print(f"Aleatoric Uncertainty: {aleatoric}")
    print(f"Epistemic Uncertainty: {epistemic}")

    return aleatoric, epistemic
