"""This code is an adjusted version of https://github.com/MLO-lab/spectral_uncertainty_decomposition/blob/main/src/uncertainty_metrics/semantic_entropy.py"""
import os
import logging

import numpy as np
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BaseEntailment:
    def save_prediction_cache(self):
        pass


class EntailmentDeberta(BaseEntailment):
    """NLI-based entailment using DeBERTa (3-way classification)."""
    def __init__(self):
        # Using deberta-v3-base-mnli-fever-anli which is more stable with newer transformers versions
        self.tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli").to(DEVICE)

    def check_implication(self, text1, text2, *args, **kwargs):
        inputs = self.tokenizer(text1, text2, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        # The model checks if text1 -> text2, i.e. if text2 follows from text1.
        # check_implication('The weather is good', 'The weather is good and I like you') --> 1
        # check_implication('The weather is good and I like you', 'The weather is good') --> 2
        outputs = self.model(**inputs)
        logits = outputs.logits
        # Deberta-mnli returns `neutral` and `entailment` classes at indices 1 and 2.
        largest_index = torch.argmax(torch.nn.functional.softmax(logits, dim=1))  # pylint: disable=no-member
        prediction = largest_index.cpu().item()
        if os.environ.get('DEBERTA_FULL_LOG', False):
            logging.info('Deberta Input: %s -> %s', text1, text2)
            logging.info('Deberta Prediction: %s', prediction)

        return prediction


class EntailmentEmbedding(BaseEntailment):
    """Embedding-based entailment using cosine similarity (faster, no 3-way classification)."""
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        equivalence_threshold: float = 0.85,
        contradiction_threshold: float = 0.3,
    ):
        """
        Args:
            model_name: Sentence transformer model to use for embeddings
            equivalence_threshold: Cosine similarity above this = entailment (2)
            contradiction_threshold: Cosine similarity below this = contradiction (0)
            Between thresholds = neutral (1)
        """
        self.model = SentenceTransformer(model_name, device=DEVICE)
        self.equivalence_threshold = equivalence_threshold
        self.contradiction_threshold = contradiction_threshold
        self._embedding_cache = {}
    
    def _get_embedding(self, text: str) -> np.ndarray:
        if text not in self._embedding_cache:
            self._embedding_cache[text] = self.model.encode(text, normalize_embeddings=True)
        return self._embedding_cache[text]
    
    def check_implication(self, text1, text2, *args, **kwargs):
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)
        
        # Cosine similarity (embeddings are normalized, so dot product = cosine sim)
        similarity = np.dot(emb1, emb2)
        
        if similarity >= self.equivalence_threshold:
            return 2  # Entailment
        elif similarity <= self.contradiction_threshold:
            return 0  # Contradiction
        else:
            return 1  # Neutral
import os
import logging

import numpy as np
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BaseEntailment:
    def save_prediction_cache(self):
        pass


class EntailmentDeberta(BaseEntailment):
    def __init__(self):
        # Using deberta-v3-base-mnli-fever-anli which is more stable with newer transformers versions
        self.tokenizer = AutoTokenizer.from_pretrained("MoritzLaworker/DeBERTa-v3-base-mnli-fever-anli")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "MoritzLaworker/DeBERTa-v3-base-mnli-fever-anli").to(DEVICE)

    def check_implication(self, text1, text2, *args, **kwargs):
        inputs = self.tokenizer(text1, text2, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        # The model checks if text1 -> text2, i.e. if text2 follows from text1.
        # check_implication('The weather is good', 'The weather is good and I like you') --> 1
        # check_implication('The weather is good and I like you', 'The weather is good') --> 2
        outputs = self.model(**inputs)
        logits = outputs.logits
        # Deberta-mnli returns `neutral` and `entailment` classes at indices 1 and 2.
        largest_index = torch.argmax(torch.nn.functional.softmax(logits, dim=1))  # pylint: disable=no-member
        prediction = largest_index.cpu().item()
        if os.environ.get('DEBERTA_FULL_LOG', False):
            logging.info('Deberta Input: %s -> %s', text1, text2)
            logging.info('Deberta Prediction: %s', prediction)

        return prediction


def get_semantic_ids(strings_list, model, strict_entailment=False, example=None):
    """Group list of predictions into semantic meaning."""

    def are_equivalent(text1, text2):

        implication_1 = model.check_implication(text1, text2, example=example)
        implication_2 = model.check_implication(text2, text1, example=example)  # pylint: disable=arguments-out-of-order
        assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])

        if strict_entailment:
            semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)

        else:
            implications = [implication_1, implication_2]
            # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
            semantically_equivalent = (0 not in implications) and ([1, 1] != implications)

        return semantically_equivalent

    # Initialise all ids with -1.
    semantic_set_ids = [-1] * len(strings_list)
    # Keep track of current id.
    next_id = 0
    for i, string1 in enumerate(strings_list):
        # Check if string1 already has an id assigned.
        if semantic_set_ids[i] == -1:
            # If string1 has not been assigned an id, assign it next_id.
            semantic_set_ids[i] = next_id
            for j in range(i+1, len(strings_list)):
                # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                if are_equivalent(string1, strings_list[j]):
                    semantic_set_ids[j] = next_id
            next_id += 1

    assert -1 not in semantic_set_ids

    return semantic_set_ids


def cluster_assignment_entropy(semantic_ids):
    """Estimate semantic uncertainty from how often different clusters get assigned.

    We estimate the categorical distribution over cluster assignments from the
    semantic ids. The uncertainty is then given by the entropy of that
    distribution. This estimate does not use token likelihoods, it relies soley
    on the cluster assignments. If probability mass is spread of between many
    clusters, entropy is larger. If probability mass is concentrated on a few
    clusters, entropy is small.

    Input:
        semantic_ids: List of semantic ids, e.g. [0, 1, 2, 1].
    Output:
        cluster_entropy: Entropy, e.g. (-p log p).sum() for p = [1/4, 2/4, 1/4].
    """

    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts/n_generations
    assert np.isclose(probabilities.sum(), 1)
    probabilities = probabilities[probabilities>0]
    entropy = - (probabilities * np.log(probabilities)).sum()
    return entropy