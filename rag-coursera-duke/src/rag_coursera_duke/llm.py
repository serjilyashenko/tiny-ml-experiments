from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from .protocols import Message


def to_openai_message(message: Message) -> ChatCompletionMessageParam:
    if message.role == "system":
        return {"role": "system", "content": message.content}
    if message.role == "user":
        return {"role": "user", "content": message.content}
    return {"role": "assistant", "content": message.content}


class LLM:
    def __init__(self) -> None:
        self.client = OpenAI(
            base_url="http://127.0.0.1:8080/v1", # "http://<Your api-server IP>:port"
            api_key = "sk-no-key-required"
        )

    def generate(self, messages: list[Message]) -> str:
        response = self.client.chat.completions.create(
            model="LLaMA_CPP",
            stream=False,
            messages=[to_openai_message(message) for message in messages],
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise RuntimeError("The model returned an empty answer.")
        return content

    def close(self) -> None:
        self.client.close()
