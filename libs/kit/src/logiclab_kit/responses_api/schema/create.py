from typing import Literal, TypedDict

from agno.models.message import Message as AgnoMessage

from .types import StrictModel


class AgnoContent(TypedDict, total=False):
    text: str
    type: Literal["text"]


class TextContent(StrictModel):
    text: str
    type: Literal["output_text", "input_text"]

    def to_agno(self) -> AgnoContent:
        return AgnoContent(text=self.text, type="text")


class InputMessage(StrictModel):
    content: str | list[TextContent]
    role: Literal["user", "assistant", "system", "developer"]
    type: Literal["message"] | None = None

    def to_agno(self) -> AgnoMessage:
        if isinstance(self.content, str):
            return AgnoMessage(role=self.role, content=self.content)

        content: list[AgnoContent] = []
        for c in self.content:
            content.append(c.to_agno())
        return AgnoMessage(role=self.role, content=content)


class AgnoCreateRequest(StrictModel):
    input: list[AgnoMessage] = []
    model: str | None = None
    stream: bool = False
    temperature: float = 1.0


class CreateRequest(StrictModel):
    input: str | list[InputMessage] | None = None
    instructions: str | None = None
    model: str | None = None
    stream: bool | None = None
    temperature: float | None = None

    def to_agno(self) -> AgnoCreateRequest:
        input: list[AgnoMessage] = []

        if self.instructions is not None:
            input.append(AgnoMessage(role="system", content=self.instructions))

        if self.input is not None:
            if isinstance(self.input, str):
                input.append(AgnoMessage(role="user", content=self.input))
            else:
                for message in self.input:
                    input.append(message.to_agno())

        stream = self.stream if self.stream is not None else False
        temperature = self.temperature if self.temperature is not None else 1.0

        return AgnoCreateRequest(
            input=input,
            model=self.model,
            stream=stream,
            temperature=temperature,
        )
