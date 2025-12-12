import cyclopts

from llm_inference import inference

app = cyclopts.App()


@app.command()
def main(
    model_url: str = "deepseek/deepseek-v3.2",
    content: str = "how many r's are in the word strawberry?",
    temperature: float = 1.0,
):
    response = inference(model_url=model_url, content=content, temperature=temperature)
    print(response)


if __name__ == "__main__":
    app()
