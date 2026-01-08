import json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_ambiguity(json_file: str):
    with open(json_file, "r") as f:
        data = json.load(f)

    y_true = np.array([1 if item["is_ambiguous"] else 0 for item in data])
    y_scores = np.array([item["aleatoric"] for item in data])

    try:
        auroc = roc_auc_score(y_true, y_scores)
        aupr = average_precision_score(y_true, y_scores)

        print(f"--- Results for {json_file} ---")
        print(f"Total Samples: {len(data)}")
        print(f"Ambiguous Count: {sum(y_true)}")
        print(f"AUROC: {auroc * 100:.2f}%")
        print(f"AUPR:  {aupr * 100:.2f}%")

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
    plt.xlabel("Aleatoric Uncertainty")
    plt.title("Uncertainty Distribution")
    plt.legend()
    plt.savefig("ambiguity_uncertainty_distribution.png")
    plt.close()


if __name__ == "__main__":
    evaluate_ambiguity("out/ambigqa_results.json")
