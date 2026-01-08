import json
from typing import Optional
import cyclopts

from llm_inference import inference
from pipeline import run_ambigqa_evaluation
from sys_prompts import (
    ambigqa_clarification_sys_prompt,
    ambigqa_target_sys_prompt,
)

app = cyclopts.App()


@app.command()
def test_llm_inference(
    model_url: str = "google/gemini-3-flash-preview",
    content: str = "how many r's are in the word strawberry?",
    temperature: float = 0.0,
    max_tokens: int = 256,
):
    res = inference(
        model_url=model_url,
        content=content,
        temperature=temperature,
        max_tokens=max_tokens,
        system="Answer concisely.",
    )
    print(
        json.dumps(
            {
                "text": res.text,
                "finish_reason": res.finish_reason,
                "model": res.model,
                "provider": res.provider,
                "usage": res.usage,
                "cost": res.cost,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


@app.command()
def ambigqa(
    split: str = "dev",
    start_index: int = 0,
    limit: Optional[int] = None,
    output_file: Optional[str] = None,
    only_ambiguous: bool = False,
    only_unambiguous: bool = False,
    temperature: float = 0.5,
    m: int = 10,
    target_llm: str = "microsoft/phi-4",
    clarification_llm: str = "openai/gpt-4o",
):
    """Run the pipeline on AmbigQA dataset and collect results.

    Args:
        split: Dataset split ("dev" or "train")
        start_index: Starting index in the dataset (0-based)
        limit: Max items to process from start_index (None for all remaining)
        output_file: Path to save results as JSON (optional)
        only_ambiguous: Filter to only ambiguous questions
        only_unambiguous: Filter to only unambiguous questions
        temperature: LLM sampling temperature
        m: samples per clarification
        target_llm: LLM model for answering questions
        clarification_llm: LLM model for generating clarifications
    """
    sys_prompt_clarify = ambigqa_clarification_sys_prompt
    sys_prompt_answer = ambigqa_target_sys_prompt
    run_ambigqa_evaluation(
        sys_prompt_clarify=sys_prompt_clarify,
        sys_prompt_answer=sys_prompt_answer,
        split=split,
        limit=limit,
        start_index=start_index,
        output_file=output_file,
        only_ambiguous=only_ambiguous,
        only_unambiguous=only_unambiguous,
        target_llm=target_llm,
        clarification_llm=clarification_llm,
        m=m,
        temperature=temperature,
    )


if __name__ == "__main__":
    app()
