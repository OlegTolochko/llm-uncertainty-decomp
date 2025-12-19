import json
import cyclopts

from llm_inference import inference

app = cyclopts.App()


@app.command()
def main(
    model_url: str = "deepseek/deepseek-v3.2",
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


if __name__ == "__main__":
    app()
