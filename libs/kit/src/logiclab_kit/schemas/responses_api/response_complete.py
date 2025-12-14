from agno.run.agent import RunOutput
from openai.types.responses import (
    Response,
    ResponseOutputText,
    ResponseOutputMessage,
)

from .response_builder import ResponseBuilder


def make_response_complete(output: RunOutput, model: str) -> Response:
    msg = output.messages[-1]
    content_part = ResponseOutputText(
        annotations=[],
        text=msg.get_content_string(),
        type="output_text",
        logprobs=None,
    )
    content_item = ResponseOutputMessage(
        id=msg.id,
        content=[content_part],
        role="assistant",
        status="completed",
        type="message",
    )

    return (
        ResponseBuilder(output.run_id, output.created_at, model)
        .status("completed")
        .metrics(output.metrics if output.metrics is not None else msg.metrics)
        .output([content_item])
        .build()
    )
