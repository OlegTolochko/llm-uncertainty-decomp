import cyclopts

app = cyclopts.App()


@app.command()
def main():
    print("Hello from llm-uncertainty-decomp!")


if __name__ == "__main__":
    app()
