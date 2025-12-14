from typing import Self

from agno.models.metrics import Metrics
from openai.types.responses import (
    Response,
    ResponseUsage,
    ResponseStatus,
    ResponseOutputItem,
)
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)


class ResponseBuilder:
    def __init__(
        self,
        id: str,
        created_at: int,
        model: str,
    ) -> None:
        self._response = Response(
            id=id,
            created_at=created_at,
            error=None,
            incomplete_details=None,
            instructions=None,
            metadata=None,
            model=model,
            object="response",
            output=[],
            parallel_tool_calls=True,  # ?
            temperature=None,
            tool_choice="auto",  # ?
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
            status=None,
            top_logprobs=None,
            truncation=None,
            usage=None,
            user=None,
        )

    def status(self, status: ResponseStatus) -> Self:
        self._response.status = status
        return self

    def metrics(self, metrics: Metrics | None) -> Self:
        if metrics is not None:
            self._response.usage = ResponseUsage(
                input_tokens=metrics.input_tokens,
                input_tokens_details=InputTokensDetails(cached_tokens=0),
                output_tokens=metrics.output_tokens,
                output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                total_tokens=metrics.total_tokens,
            )

        return self

    def output(self, output: list[ResponseOutputItem]) -> Self:
        self._response.output = output
        return self

    def build(self) -> Response:
        return self._response
