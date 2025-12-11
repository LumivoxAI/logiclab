import json
from typing import Any, AsyncIterator

import httpx


def _parse_content(content: bytes) -> Any:
    try:
        text = content.decode("utf-8", errors="replace")
    except Exception:
        return "<binary>"

    # Parse JSON
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # Parse SSE (Server-Sent Events)
    stripped_text = text.strip()
    if stripped_text.startswith("data:"):
        parsed_events = []

        for event in text.split("\n\n"):
            event = event.strip()
            if not event.startswith("data:"):
                continue

            payload = event[5:].strip()

            if payload == "[DONE]":
                parsed_events.append(payload)
                continue

            try:
                parsed_events.append(json.loads(payload))
            except (json.JSONDecodeError, TypeError):
                parsed_events.append(payload)

        if parsed_events:
            return parsed_events[0] if len(parsed_events) == 1 else parsed_events

    return text


class _LoggingStream(httpx.AsyncByteStream):
    def __init__(self, response: httpx.Response, logger: Any) -> None:
        self._response = response
        self._stream = response.stream
        self._logger = logger
        self._chunks = []
        self._logged = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._stream:
                self._chunks.append(chunk)
                yield chunk
        finally:
            self._log_content()

    async def aclose(self) -> None:
        self._log_content()
        await self._stream.aclose()

    def _log_content(self) -> None:
        if self._logged:
            return
        self._logged = True

        if len(self._chunks) == 0:
            body = None
        elif len(self._chunks) == 1:
            body = _parse_content(self._chunks[0])
        else:
            body = [_parse_content(chunk) for chunk in self._chunks]

        r = self._response

        self._logger.info(
            "httpx_response",
            url=str(r.url),
            status_code=r.status_code,
            headers=dict(r.headers),
            body=body,
        )


class HttpClientManager:
    def __init__(
        self,
        logger: Any,
        log_request: bool = False,
        log_response: bool = False,
        proxy: httpx._types.ProxyTypes | None = None,
    ) -> None:
        self._proxy = proxy
        self._logger = logger

        self._event_hooks = {}
        if log_request:
            self._event_hooks["request"] = [self._log_request]
        if log_response:
            self._event_hooks["response"] = [self._log_response]

        self._async_client: httpx.AsyncClient | None = None

    async def _log_request(self, request: httpx.Request) -> None:
        headers = dict(request.headers)
        if "authorization" in headers:
            headers["authorization"] = "Bearer [REDACTED]"

        body = None
        if request.content:
            body = _parse_content(request.content)

        self._logger.info(
            "httpx_request", method=request.method, url=str(request.url), headers=headers, body=body
        )

    async def _log_response(self, response: httpx.Response) -> None:
        response.stream = _LoggingStream(response, self._logger)

    def _make_async_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            verify=True,
            cert=None,
            http1=True,
            http2=True,
            proxy=self._proxy,
            follow_redirects=False,
            event_hooks=self._event_hooks,
        )

    def get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = self._make_async_client()

        return self._async_client

    async def close(self) -> None:
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None
