import json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from sentence_transformers import SentenceTransformer

from spectral_uncertainty import compute_spectral_uncertainty_from_embeddings
from semantic_entropy import get_semantic_ids, cluster_assignment_entropy, EntailmentDeberta

_model: SentenceTransformer | None = None


def load_results(json_file: str) -> list[dict]:
    """Load results from JSON file."""
    with open(json_file, "r") as f:
        return json.load(f)


def get_model() -> SentenceTransformer:
    """Lazy load the sentence transformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    return _model


def embed_sentences(sentences: List[str]) -> np.ndarray:
    """Embed sentences using SentenceTransformer.
    
    Args:
        sentences: List of sentences to embed
        
    Returns:
        Array of shape (n_sentences, embed_dim)
    """
    model = get_model()
    embeddings = model.encode(sentences)
    return embeddings


def embed_model_answers(model_answers: List[List[str]]) -> List[np.ndarray]:
    """Embed all model answers for a single item.
    
    Args:
        model_answers: List of lists, shape [n_clarifications][m_samples]
        
    Returns:
        List of embedding arrays, shape [n_clarifications] where each is (m_samples, embed_dim)
    """
    answer_embeddings = []
    for answers in model_answers:
        embeddings = embed_sentences(answers)
        answer_embeddings.append(embeddings)
    return answer_embeddings


def compute_spectral_scores(data: list[dict]) -> np.ndarray:
    """Compute spectral aleatoric uncertainty for all items."""
    scores = []
    for item in data:
        model_answers = item.get("model_answers", [])
        if model_answers and all(len(answers) > 0 for answers in model_answers):
            answer_embeddings = embed_model_answers(model_answers)
            aleatoric, _ = compute_spectral_uncertainty_from_embeddings(answer_embeddings)
            scores.append(aleatoric)
        else:
            scores.append(0.0)
    return np.array(scores)


def compute_semantic_entropy_scores(data: list[dict]) -> np.ndarray:
    """Compute semantic entropy scores for all items."""
    model = EntailmentDeberta()
    scores = []
    for item in data:
        model_answers = item.get("model_answers", [])
        if model_answers and all(len(answers) > 0 for answers in model_answers):
            ids = get_semantic_ids(model_answers, model)
            uncertainty = cluster_assignment_entropy(ids)
            scores.append(uncertainty)
        else:
            scores.append(0.0)
    return np.array(scores)

def evaluate_ambiguity(json_file: str):
    """Evaluate ambiguity detection using different uncertainty methods."""
    data = load_results(json_file)

    print(f"--- Results for {json_file} ---")
    print(f"Total Samples: {len(data)}")

    y_true = np.array([1 if item["is_ambiguous"] else 0 for item in data])

    # Spectral Uncertainty
    print("\nComputing spectral uncertainty")
    spectral_scores = compute_spectral_scores(data)
    eval_uncertainty(
        y_true=y_true,
        y_scores=spectral_scores,
        method="Spectral Uncertainty (aleatoric)",
        short_name="spectral_aleatoric",
    )

    print("\nComputing Semantic Entropy")
    semantic_scores = compute_semantic_entropy_scores(data)
    eval_uncertainty(y_true=y_true, y_scores=semantic_scores, method="Semantic Entropy", short_name="semantic_entropy")


def eval_uncertainty(y_true: np.ndarray, y_scores: np.ndarray, method: str, short_name: str):
    """Evaluate and visualize uncertainty scores for ambiguity detection."""
    try:
        auroc = roc_auc_score(y_true, y_scores)
        aupr = average_precision_score(y_true, y_scores)

        print(f"\n{method}:")
        print(f"  Ambiguous Count: {sum(y_true)}")
        print(f"  AUROC: {auroc * 100:.2f}%")
        print(f"  AUPR:  {aupr * 100:.2f}%")

    except ValueError as e:
        print(
            f"Error computing metrics (needs both positive and negative classes): {e}"
        )

    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        y_scores[y_true == 1], fill=True, label="Ambiguous", color="red", alpha=0.3
    )
    sns.kdeplot(
        y_scores[y_true == 0], fill=True, label="Non-ambiguous", color="blue", alpha=0.3
    )
    plt.xlabel("Uncertainty")
    plt.title(method)
    plt.legend()
    plt.savefig(f"out/ambiguity_uncertainty_distribution_{short_name}.png")
    plt.close()
    print(f"  Plot saved to: out/ambiguity_uncertainty_distribution_{short_name}.png")


if __name__ == "__main__":
    evaluate_ambiguity("out/results.json")
