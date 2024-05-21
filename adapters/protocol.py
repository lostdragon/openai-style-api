from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional, Union, Iterable
import time


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class ChatMessage(BaseModel):
    role: Literal["assistant", "system", "function"]
    content: Optional[str]
    function_call: Optional[Dict] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionContentPartTextParam(BaseModel):
    text: str
    """The text content."""

    type: str = "text"
    """The type of the content part."""


class ImageURL(BaseModel):
    url: str
    """Either a URL of the image or the base64 encoded image data."""

    detail: Literal["auto", "low", "high"] = "auto"
    """Specifies the detail level of the image.

    Learn more in the
    [Vision guide](https://platform.openai.com/docs/guides/vision/low-or-high-fidelity-image-understanding).
    """


class ChatCompletionContentPartImageParam(BaseModel):
    image_url: ImageURL

    type: str = "image_url"
    """The type of the content part."""


ChatCompletionContentPartParam = Union[ChatCompletionContentPartTextParam, ChatCompletionContentPartImageParam]


class ChatCompletionUserMessageParam(BaseModel):
    content: Union[str, Iterable[ChatCompletionContentPartParam]]
    """The contents of the user message."""

    role: str = "user"
    """The role of the messages author, in this case `user`."""

    name: Optional[str] = None
    """An optional name for the participant.

    Provides the model information to differentiate between participants of the same
    role.
    """


ChatCompletionMessageParam = Union[
    ChatMessage,
    ChatCompletionUserMessageParam,
]


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "gpt-3.5-turbo"
    messages: List[ChatCompletionUserMessageParam]
    functions: Optional[List[Dict]] = None
    temperature: Optional[float] = None  # between 0 and 2    Defaults to 1
    top_p: Optional[float] = None  # Defaults to 1
    max_length: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "content_filter", "length", "function_call"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "content_filter", "length"]]
    logprobs: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = f"chatcmpl-{str(time.time())}"
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[
        Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]
    ]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None


class Error(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    status_code: int | None
    error: Error
