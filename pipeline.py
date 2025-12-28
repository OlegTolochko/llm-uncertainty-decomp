from typing import List
import re
from sentence_transformers import SentenceTransformer

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
