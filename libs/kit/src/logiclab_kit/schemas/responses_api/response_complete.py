import uuid
from typing import Any
from datetime import datetime

from agno.run.agent import RunOutput
from openai.types.responses import (
    Response,
    ResponseUsage,
    ResponseOutputText,
    ResponseOutputMessage,
)
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)


def make_response_complete(msg: RunOutput, model: str) -> Response:
    last_msg = msg.messages[-1]
    content_part = ResponseOutputText(
        annotations=[],
        text=last_msg.get_content_string(),
        type="output_text",
        logprobs=None,
    )
    content_item = ResponseOutputMessage(
        id=last_msg.id,
        content=[content_part],
        role="assistant",
        status="completed",
        type="message",
    )

    if msg.metrics is not None:
        metrics = msg.metrics
    elif last_msg.metrics is not None:
        metrics = last_msg.metrics
    else:
        metrics = None

    if metrics is not None:
        usage = ResponseUsage(
            input_tokens=metrics.input_tokens,
            input_tokens_details=InputTokensDetails(cached_tokens=metrics.cache_read_tokens),
            output_tokens=metrics.output_tokens,
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
            total_tokens=metrics.total_tokens,
        )
    else:
        usage = None

    return Response(
        id=msg.run_id,
        created_at=msg.created_at,
        error=None,
        incomplete_details=None,
        instructions=None,
        metadata=None,
        model=model,
        object="response",
        output=[content_item],
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
        status="completed",
        top_logprobs=None,
        truncation=None,
        usage=usage,
        user=None,
    )
