from openai import AsyncOpenAI
from enum import Enum
from typing import Any, AsyncGenerator, Literal
import os
import logging

logger = logging.getLogger(__name__)

class Author(Enum):
    USER = "me"
    OLLAMA = "ollama"

class GPT4LLM:
    def __init__(self, 
        model="gpt-4o", 
        system: str | None = None,
        context: list[int] = [],
        format: Literal["", "json"] = "",
        keep_alive: int = 5):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.system = system
        self.context = context
        self.format = format
        self.keep_alive = keep_alive
        logger.info(f"GPT4LLM initialized with model: {self.model}")

    async def completion(self, prompt: str) -> str:
        logger.info(f"Sending completion request with prompt: {prompt}")
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": "You are a friendly assistant."}, {"role": "user", "content": prompt}],
            max_tokens=1024
        )
        logger.info(f"Received completion response: {response}")
        return response.choices[0].message.content.strip()

    async def stream(self, prompt: str, images: list[str] = [], msgs: list[tuple[Author, str]] = []) -> AsyncGenerator[str, Any]:
        logger.info(f"Starting stream with prompt: {prompt}")
        messages = [{"role": "system", "content": "You are a friendly assistant."}]
        for author, content in msgs:
            role = "user" if author == Author.USER else "assistant"
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": prompt})
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=2048,
            stream=True
        )
        async for chunk in response:
            if chunk.choices[0].delta.content is not None:
                logger.info(f"Stream chunk received: {chunk.choices[0].delta.content}")
                yield chunk.choices[0].delta.content
