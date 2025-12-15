from typing import Self, NoReturn, AsyncIterator

from pydantic import BaseModel
from agno.run.agent import (
    RunEvent,
    RunOutputEvent,
    RunContentEvent,
    RunStartedEvent,
    RunCompletedEvent,
    RunContentCompletedEvent,
)
from starlette.types import Send
from structlog.typing import FilteringBoundLogger
from fastapi.responses import StreamingResponse as FastAPIStreamingResponse

from .schema import CreateRequest, ResponseStream, ResponseTextPart, ResponseOutputItem


class EmptyAsyncIterator:
    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> NoReturn:
        raise StopAsyncIteration


class RunOutputEventIterator:
    def __init__(
        self,
        iterator: AsyncIterator[RunOutputEvent],
        logger: FilteringBoundLogger,
    ) -> None:
        self._exhausted = False
        self._iterator = iterator.__aiter__()
        self._event = None
        self._logger = logger

    async def _read(self) -> None:
        if self._exhausted:
            raise StopAsyncIteration

        try:
            self._event = await self._iterator.__anext__()
            self._logger.info("RunOutputEvent", body=self._event.to_dict())
        except StopAsyncIteration:
            self._exhausted = True
            raise

    async def next_id(self) -> str:
        if self._event is None:
            await self._read()

        return self._event.event

    async def next(self) -> RunOutputEvent:
        if self._event is None:
            await self._read()

        event = self._event
        self._event = None
        return event


class StreamingResponse(FastAPIStreamingResponse):
    def __init__(
        self,
        output: AsyncIterator[RunOutputEvent],
        request: CreateRequest,
        logger: FilteringBoundLogger,
    ) -> None:
        super().__init__(
            content=EmptyAsyncIterator(),
            status_code=200,
            headers=None,
            media_type="text/event-stream",
            background=None,
        )
        self._events = RunOutputEventIterator(output, logger)
        self._request = request
        self._send: Send | None = None
        self._stream: ResponseStream | None = None
        self._oitem: ResponseOutputItem | None = None

    async def send_model(self, model: BaseModel) -> None:
        data = f"data: {model.model_dump_json()}\n\n".encode("utf-8")
        await self._send({"type": "http.response.body", "body": data, "more_body": True})

    async def send_string(self, s: str) -> None:
        data = f"data: {s}\n\n".encode("utf-8")
        await self._send({"type": "http.response.body", "body": data, "more_body": True})

    async def stream_response(self, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.raw_headers,
            }
        )
        self._send = send
        await self._do()
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def _do(self) -> None:
        id = await self._events.next_id()
        if id != RunEvent.run_started.value:
            raise ValueError(f"Unexpected event: {id}")

        event: RunStartedEvent = await self._events.next()
        self._stream = ResponseStream(
            id=event.run_id,
            created_at=event.created_at,
            model=self._request.model,
        )
        await self.send_model(self._stream.enter())

        self._oitem = self._stream.new_output_item()
        await self.send_model(self._oitem.enter())

        await self._content()

        id = await self._events.next_id()
        if id != RunEvent.run_completed.value:
            raise ValueError(f"Unexpected event: {id}")

        event: RunCompletedEvent = await self._events.next()
        await self.send_model(self._oitem.exit())
        await self.send_model(self._stream.exit(event.metrics))

        await self.send_string("[DONE]")

    async def _content(self) -> None:
        text_part: ResponseTextPart | None = None

        while True:
            id = await self._events.next_id()

            if id == RunEvent.run_content.value:
                event: RunContentEvent = await self._events.next()
                if event.content_type != "str":
                    raise ValueError(f"Unexpected content type: {event.content_type}")
                content = event.content
                if content is None:
                    raise ValueError(f"Unexpected content: None")
                content = str(content)
                if content == "":
                    continue
                if text_part is None:
                    text_part = self._oitem.new_text_part()
                    await self.send_model(text_part.enter())
                await self.send_model(text_part.add(content))
                continue

            if id == RunEvent.run_content_completed.value:
                event: RunContentCompletedEvent = await self._events.next()
                if text_part is None:
                    raise ValueError(f"Unexpected event: {id}, text_part is None")
                await self.send_model(text_part.done(None))
                await self.send_model(text_part.exit())
                text_part = None
                continue

            if id == RunEvent.run_completed.value:
                if text_part is not None:
                    raise ValueError(f"Unexpected event: {id}, text_part is not None")

                return

            raise ValueError(f"Unexpected event: {id}")
