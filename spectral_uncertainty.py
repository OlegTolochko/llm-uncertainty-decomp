"""Spectral uncertainty quantification based on embedding similarity matrices.

This module computes aleatoric and epistemic uncertainty using eigenvalue
decomposition of RBF kernel similarity matrices over answer embeddings.
"""

from typing import List, Tuple
import numpy as np


def compute_similarity_matrix(embeddings: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Compute RBF kernel similarity matrix from embeddings.

    Args:
        embeddings: Array of shape (n, embed_dim)
        gamma: RBF kernel bandwidth parameter

    Returns:
        Similarity matrix of shape (n, n)
    """
    outputs_arr = np.asarray(embeddings)[:, np.newaxis, :]  # (n, 1, embed_dim)
    outputs_arr_t = np.transpose(outputs_arr, (1, 0, 2))  # (1, n, embed_dim)
    element_wise_diff = outputs_arr - outputs_arr_t  # (n, n, embed_dim)
    dist_sq = np.sum(element_wise_diff**2, axis=2)
    sim_matrix = np.exp(-gamma * dist_sq)  # (n, n)
    return sim_matrix


def compute_inner_eigenvalues(
    answer_embeddings: List[np.ndarray], gamma: float = 1.0
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Compute eigenvalues for each clarification's answer set.

    Args:
        answer_embeddings: List of embedding arrays, shape [n_clarifications] where each is (m_samples, embed_dim)
        gamma: RBF kernel bandwidth parameter

    Returns:
        Tuple of (all_embeddings, inner_eigenvalues)
        - all_embeddings: List of embedding arrays per clarification
        - inner_eigenvalues: List of eigenvalue arrays per clarification
    """
    all_embeddings = []
    inner_eigenvalues = []

    for embeddings in answer_embeddings:
        similarity_matrix = compute_similarity_matrix(embeddings, gamma=gamma)

        # Normalize and compute eigenvalues
        sim_matrix_norm = (1 / embeddings.shape[0]) * similarity_matrix
        eig_values = np.linalg.eigvalsh(sim_matrix_norm)
        eig_values = np.maximum(eig_values, 1e-10)  # Numerical stability

        all_embeddings.append(embeddings)
        inner_eigenvalues.append(eig_values)

    return all_embeddings, inner_eigenvalues


def compute_outer_eigenvalues(
    all_embeddings: List[np.ndarray], gamma: float = 1.0
) -> np.ndarray:
    """Compute eigenvalues for the combined (outer) similarity matrix.

    Args:
        all_embeddings: List of embedding arrays per clarification
        gamma: RBF kernel parameter

    Returns:
        Eigenvalues of the outer similarity matrix
    """
    flattened_embeddings = np.vstack(all_embeddings)  # (n*m, embed_dim)
    nm = flattened_embeddings.shape[0]

    K_out = compute_similarity_matrix(flattened_embeddings, gamma=gamma)
    K_out_norm = (1 / nm) * K_out
    outer_eigenvalues = np.linalg.eigvalsh(K_out_norm)
    outer_eigenvalues = np.maximum(outer_eigenvalues, 1e-10)

    return outer_eigenvalues


def compute_spectral_uncertainty(
    inner_eigenvalues: List[np.ndarray], outer_eigenvalues: np.ndarray
) -> Tuple[float, float]:
    """Compute aleatoric and epistemic uncertainty from eigenvalues.

    Args:
        inner_eigenvalues: List of eigenvalue arrays per clarification
        outer_eigenvalues: Eigenvalues of the outer similarity matrix

    Returns:
        Tuple of (aleatoric, epistemic) uncertainty values
    """
    n = len(inner_eigenvalues)
    inner_eigenvalues_flat = np.concatenate(inner_eigenvalues)

    aleatoric = (1 / n) * np.sum(
        inner_eigenvalues_flat * np.log(inner_eigenvalues_flat)
    ) - np.sum(outer_eigenvalues * np.log(outer_eigenvalues))

    epistemic = -(1 / n) * np.sum(
        inner_eigenvalues_flat * np.log(inner_eigenvalues_flat)
    )

    return float(aleatoric), float(epistemic)


def compute_spectral_uncertainty_from_embeddings(
    answer_embeddings: List[np.ndarray], gamma: float = 1.0
) -> Tuple[float, float]:
    """Compute spectral uncertainty from pre-computed embeddings.

    This is the main entry point for computing spectral uncertainty.

    Args:
        answer_embeddings: List of embedding arrays, shape [n_clarifications] where each is (m_samples, embed_dim)
        gamma: RBF kernel bandwidth parameter

    Returns:
        Tuple of (aleatoric, epistemic) uncertainty values
    """
    all_embeddings, inner_eigenvalues = compute_inner_eigenvalues(
        answer_embeddings, gamma=gamma
    )
    outer_eigenvalues = compute_outer_eigenvalues(all_embeddings, gamma=gamma)

    return compute_spectral_uncertainty(inner_eigenvalues, outer_eigenvalues)


def compute_spectral_uncertainty_batch(
    results_data: List[dict], gamma: float = 1.0
) -> List[dict]:
    """Compute spectral uncertainty for a batch of results.

    Args:
        results_data: List of result dicts with 'answer_embeddings' key (list of np.ndarray)
        gamma: RBF kernel bandwidth parameter

    Returns:
        Same list with 'spectral_aleatoric' and 'spectral_epistemic' added
    """
    for item in results_data:
        answer_embeddings = item.get("answer_embeddings", [])
        if answer_embeddings and all(len(emb) > 0 for emb in answer_embeddings):
            aleatoric, epistemic = compute_spectral_uncertainty_from_embeddings(
                answer_embeddings, gamma=gamma
            )
            item["spectral_aleatoric"] = aleatoric
            item["spectral_epistemic"] = epistemic
        else:
            item["spectral_aleatoric"] = None
            item["spectral_epistemic"] = None

    return results_data
