import uuid
from typing import Any

from agno.models.metrics import Metrics
from openai.types.responses import (
    ResponseOutputText,
    ResponseCreatedEvent,
    ResponseOutputMessage,
    ResponseTextDoneEvent,
    ResponseCompletedEvent,
    ResponseTextDeltaEvent,
    ResponseOutputItemDoneEvent,
    ResponseContentPartDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseContentPartAddedEvent,
)

from .response_builder import ResponseBuilder


class SequenceNumberCounter:
    def __init__(self, start: int = 1) -> None:
        self._value = start - 1

    def __call__(self) -> int:
        self._value += 1
        return self._value


class OutputItemContext:
    def __init__(
        self,
        output_index: int,
        sequence_number: SequenceNumberCounter,
        content_index_start: int = 0,
    ) -> None:
        self._item_id = "msg_" + str(uuid.uuid4())[:8]
        self._output_index = output_index
        self._content_index = content_index_start - 1
        self._sequence_number = sequence_number
        self._content = []

    @property
    def item_id(self) -> str:
        return self._item_id

    @property
    def output_index(self) -> int:
        return self._output_index

    @property
    def content_index(self) -> int:
        self._content_index += 1
        return self._content_index

    @property
    def sequence_number(self) -> int:
        return self._sequence_number()

    def add_content(self, content) -> None:
        self._content.append(content)

    @property
    def content(self) -> list[Any]:
        return self._content


class ResponseTextPart:
    def __init__(
        self,
        ctx: OutputItemContext,
    ) -> None:
        self._ctx = ctx
        self._text = ""
        self._content_index = ctx.content_index

    def enter(self) -> ResponseContentPartAddedEvent:
        ctx = self._ctx
        part = ResponseOutputText(
            annotations=[],
            text="",
            type="output_text",
            logprobs=None,
        )
        return ResponseContentPartAddedEvent(
            content_index=self._content_index,
            item_id=ctx.item_id,
            output_index=ctx.output_index,
            part=part,
            sequence_number=ctx.sequence_number,
            type="response.content_part.added",
        )

    def add(self, delta: str) -> ResponseTextDeltaEvent:
        ctx = self._ctx
        self._text += delta
        return ResponseTextDeltaEvent(
            content_index=self._content_index,
            delta=delta,
            item_id=ctx.item_id,
            logprobs=[],
            output_index=ctx.output_index,
            sequence_number=ctx.sequence_number,
            type="response.output_text.delta",
        )

    def done(self, text: str | None) -> ResponseTextDoneEvent:
        ctx = self._ctx
        if text is not None:
            self._text = text
        return ResponseTextDoneEvent(
            content_index=self._content_index,
            item_id=ctx.item_id,
            logprobs=[],
            output_index=ctx.output_index,
            sequence_number=ctx.sequence_number,
            text=self._text,
            type="response.output_text.done",
        )

    def exit(self) -> ResponseContentPartDoneEvent:
        ctx = self._ctx
        part = ResponseOutputText(
            annotations=[],
            text=self._text,
            type="output_text",
            logprobs=None,
        )
        ctx.add_content(part)
        return ResponseContentPartDoneEvent(
            content_index=self._content_index,
            item_id=ctx.item_id,
            output_index=ctx.output_index,
            part=part,
            sequence_number=ctx.sequence_number,
            type="response.content_part.done",
        )


class ResponseOutputItem:
    def __init__(
        self,
        output_index: int,
        sequence_number: SequenceNumberCounter,
    ) -> None:
        self._ctx = OutputItemContext(output_index, sequence_number, 0)
        self._content_item: ResponseOutputMessage | None = None

    def enter(self) -> ResponseOutputItemAddedEvent:
        ctx = self._ctx
        item = ResponseOutputMessage(
            id=ctx.item_id,
            content=[],
            role="assistant",
            status="in_progress",
            type="message",
        )
        return ResponseOutputItemAddedEvent(
            item=item,
            output_index=ctx.output_index,
            sequence_number=ctx.sequence_number,
            type="response.output_item.added",
        )

    def new_text_part(self) -> ResponseTextPart:
        return ResponseTextPart(self._ctx)

    def exit(self) -> ResponseOutputItemDoneEvent:
        ctx = self._ctx
        self._content_item = ResponseOutputMessage(
            id=ctx.item_id,
            content=ctx.content,
            role="assistant",
            status="completed",
            type="message",
        )
        return ResponseOutputItemDoneEvent(
            item=self._content_item,
            output_index=ctx.output_index,
            sequence_number=ctx.sequence_number,
            type="response.output_item.done",
        )

    @property
    def content_item(self) -> ResponseOutputMessage | None:
        return self._content_item


class ResponseStream:
    def __init__(self, id: str, created_at: int, model: str) -> None:
        self._output_items = []
        self._output_index: int = -1
        self._sequence_number = SequenceNumberCounter(start=1)
        self._response = ResponseBuilder(id, created_at, model).status("in_progress")

    def enter(self) -> ResponseCreatedEvent:
        return ResponseCreatedEvent(
            response=self._response.build(),
            sequence_number=self._sequence_number(),
            type="response.created",
        )

    def new_output_item(self) -> ResponseOutputItem:
        self._output_index += 1
        item = ResponseOutputItem(self._output_index, self._sequence_number)
        self._output_items.append(item)
        return item

    def exit(self, metrics: Metrics | None) -> ResponseCompletedEvent:
        response = (
            self._response.output(
                [item.content_item for item in self._output_items if item.content_item is not None]
            )
            .status("completed")
            .metrics(metrics)
            .build()
        )
        return ResponseCompletedEvent(
            response=response,
            sequence_number=self._sequence_number(),
            type="response.completed",
        )
