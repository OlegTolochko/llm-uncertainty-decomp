import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_api_key = os.getenv("OPENROUTER_API_KEY")
if not _api_key:
    raise RuntimeError("OPENROUTER_API_KEY is not set in the environment (.env)")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=_api_key)


def inference(
    model_url: str,
    content: Optional[str] = None,
    *,
    messages: Optional[List[Dict[str, str]]] = None,
    system: Optional[str] = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: Optional[int] = None,
    n: int = 1,
):
    if messages is None:
        if content is None:
            raise ValueError("Provide either `content` or `messages`.")
        messages = [{"role": "user", "content": content}]

    if system:
        messages = [{"role": "system", "content": system}] + messages

    response = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", ""),
            "X-Title": os.getenv("OPENROUTER_SITE_NAME", ""),
        },
        model=model_url,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=n,
    )
    return response
