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


@dataclass
class InferenceResult:
    text: str
    finish_reason: Optional[str]
    model: Optional[str]
    provider: Optional[str]
    usage: Optional[Dict[str, Any]]
    cost: Optional[float]
    raw: Any


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
    try:
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
        choice0 = response.choices[0]
        text = (choice0.message.content or "").strip()
        usage = None
        cost = None
        provider = None
        model = getattr(response, "model", None)

        usage_obj = getattr(response, "usage", None)
        if usage_obj is not None:
            usage = (
                    usage_obj.model_dump()
                if hasattr(usage_obj, "model_dump")
                else dict(usage_obj)
            )
            cost = usage.get("cost")

        provider = getattr(response, "provider", None)

        return InferenceResult(
            text=text,
            finish_reason=getattr(choice0, "finish_reason", None),
            model=model,
            provider=provider,
            usage=usage,
            cost=cost,
            raw=response,
        )
    except Exception as e:
        raise
