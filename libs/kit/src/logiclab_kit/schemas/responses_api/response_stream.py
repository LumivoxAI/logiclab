import uuid
from typing import Any
from datetime import datetime

from openai.types.responses import (
    Response,
    ResponseUsage,
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
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)


def new_resp_id() -> str:
    return f"resp_{uuid.uuid4()}"


def created_at() -> int:
    return int(datetime.now().timestamp())


class ResponseStream:
    def __init__(self, model: str) -> None:
        self._text = ""
        self._output_index: int = 0
        self._content_index: int = 0
        self._sequence_number: int = 0
        self._response = Response(
            id=new_resp_id(),
            created_at=created_at(),
            error=None,
            incomplete_details=None,
            instructions=None,
            metadata=None,
            model=model,
            object="response",
            output=[],
            parallel_tool_calls=True,  # ???
            temperature=None,
            tool_choice="auto",  # ???
            tools=[],
            top_p=None,
            background=None,
            conversation=None,
            max_output_tokens=None,
            max_tool_calls=None,
            previous_response_id=None,
            prompt=None,
            prompt_cache_key=None,
            prompt_cache_retention=None,
            reasoning=None,
            safety_identifier=None,
            service_tier=None,
            status="in_progress",
            top_logprobs=None,
            truncation=None,
            usage=None,
            user=None,
        )

    @property
    def sequence_number(self) -> int:
        self._sequence_number += 1
        return self._sequence_number

    def response_created(self) -> ResponseCreatedEvent:
        return ResponseCreatedEvent(
            response=self._response,
            sequence_number=self.sequence_number,
            type="response.created",
        )

    def response_output_item_added(self) -> ResponseOutputItemAddedEvent:
        self._item_id = "msg_" + str(uuid.uuid4())[:8]
        item = ResponseOutputMessage(
            id=self._item_id,
            content=[],
            role="assistant",
            status="in_progress",
            type="message",
        )
        return ResponseOutputItemAddedEvent(
            item=item,
            output_index=self._output_index,
            sequence_number=self.sequence_number,
            type="response.output_item.added",
        )

    def response_content_part_added(self) -> ResponseContentPartAddedEvent:
        part = ResponseOutputText(
            annotations=[],
            text="",
            type="output_text",
            logprobs=None,
        )
        return ResponseContentPartAddedEvent(
            content_index=self._content_index,
            item_id=self._item_id,
            output_index=self._output_index,
            part=part,
            sequence_number=self.sequence_number,
            type="response.content_part.added",
        )

    def response_output_text_delta(self, delta: str) -> ResponseTextDeltaEvent:
        self._text += delta
        return ResponseTextDeltaEvent(
            content_index=self._content_index,
            delta=delta,
            item_id=self._item_id,
            logprobs=[],
            output_index=self._output_index,
            sequence_number=self.sequence_number,
            type="response.output_text.delta",
        )

    def response_output_text_done(self) -> ResponseTextDoneEvent:
        return ResponseTextDoneEvent(
            content_index=self._content_index,
            item_id=self._item_id,
            logprobs=[],
            output_index=self._output_index,
            sequence_number=self.sequence_number,
            text=self._text,
            type="response.output_text.done",
        )

    def response_content_part_done(self) -> ResponseContentPartDoneEvent:
        self._content_part = ResponseOutputText(
            annotations=[],
            text=self._text,
            type="output_text",
            logprobs=None,
        )
        return ResponseContentPartDoneEvent(
            content_index=self._content_index,
            item_id=self._item_id,
            output_index=self._output_index,
            part=self._content_part,
            sequence_number=self.sequence_number,
            type="response.content_part.done",
        )

    def response_output_item_done(self) -> ResponseOutputItemDoneEvent:
        self._content_item = ResponseOutputMessage(
            id=self._item_id,
            content=[self._content_part],
            role="assistant",
            status="completed",
            type="message",
        )
        return ResponseOutputItemDoneEvent(
            item=self._content_item,
            output_index=self._output_index,
            sequence_number=self.sequence_number,
            type="response.output_item.done",
        )

    def response_completed(self, metrics: Any) -> ResponseCompletedEvent:
        self._response.status = "completed"
        self._response.output = [self._content_item]
        if metrics:
            self._response.usage = ResponseUsage(
                input_tokens=metrics.input_tokens,
                input_tokens_details=InputTokensDetails(cached_tokens=0),
                output_tokens=metrics.output_tokens,
                output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                total_tokens=metrics.total_tokens,
            )
        return ResponseCompletedEvent(
            response=self._response,
            sequence_number=self.sequence_number,
            type="response.completed",
        )
