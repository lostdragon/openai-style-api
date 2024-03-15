from typing import Dict, Iterator, List
from adapters.base import ModelAdapter
from adapters.protocol import ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ErrorResponse
import time

import cachetools.func
import jwt
from loguru import logger
from utils.util import num_tokens_from_string

from utils.sse_client import SSEClient
from utils.http_util import post, stream

API_TOKEN_TTL_SECONDS = 3 * 60

CACHE_TTL_SECONDS = API_TOKEN_TTL_SECONDS - 30


@cachetools.func.ttl_cache(maxsize=10, ttl=CACHE_TTL_SECONDS)
def generate_token(apikey: str):
    try:
        api_key, secret = apikey.split(".")
    except Exception as e:
        raise Exception("invalid api_key", e)

    payload = {
        "api_key": api_key,
        "exp": int(round(time.time() * 1000)) + API_TOKEN_TTL_SECONDS * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }

    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )


headers = {
    "Accept": "application/json",
    "Content-Type": "application/json; charset=UTF-8",
}


class ZhiPuApiModel(ModelAdapter):
    """
    API 模型适配器
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = kwargs.pop("api_key", None)
        self.model = kwargs.pop("model", None)
        self.prompt = kwargs.pop(
            "prompt", "You need to follow the system settings:{system}")
        self.config_args = kwargs

    def chat_completions(self, request: ChatCompletionRequest) -> Iterator[ChatCompletionResponse]:
        '''
        https://open.bigmodel.cn/dev/api#http
        https://open.bigmodel.cn/dev/api#sdk
        '''
        # 发起post请求
        model = self.model if self.model else request.model
        invoke_method = "sse-invoke" if request.stream else "invoke"
        url = f"https://open.bigmodel.cn/api/paas/v3/model-api/{model}/{invoke_method}"
        token = generate_token(self.api_key)
        params = self.convert_params(request)
        if request.stream:
            data = stream(url, {"Authorization": token}, params)
            event_data = SSEClient(data)
            for event in event_data.events():
                logger.debug(f"chat_completions event: {event}")
                yield ChatCompletionResponse(**self.convert_response_stream(event, model))
        else:
            global headers
            headers.update({"Authorization": token})
            response = post(url, headers, params)
            logger.debug(f"chat_completions data: {response.json()}")
            if response.status_code == 200:
                yield ChatCompletionResponse(**self.convert_response(response.json(), model))
            else:
                yield ErrorResponse(status_code=response.status_code, **response.json())

    def convert_response(self, resp, model):
        resp = resp["data"]
        req_id = resp["request_id"]
        openai_response = {
            "id": f"chatcmpl-{req_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "usage": {
                "prompt_tokens": resp["usage"]["prompt_tokens"],
                "completion_tokens": resp["usage"]["completion_tokens"],
                "total_tokens": resp["usage"]["total_tokens"],
            },
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": resp["choices"][0]["content"],
                    },
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
        }
        return openai_response

    def convert_response_stream(self, event_data, model):
        completion = event_data.data
        completion_tokens = num_tokens_from_string(completion)
        finish_reason = "stop" if event_data.event == "finish" else None
        openai_response = {
            "id": f"chatcmpl-{event_data.id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": completion_tokens,
                "total_tokens": completion_tokens,
            },
            "choices": [
                {
                    "delta": {
                        "role": "assistant",
                        "content": completion,
                    },
                    "index": 0,
                    "finish_reason": finish_reason,
                }
            ],
        }
        return openai_response

    def convert_params(self, request: ChatCompletionRequest) -> Dict:
        """
        将请求参数转换为 API 请求参数
        """
        req_args = request.model_dump(exclude_none=True, exclude_defaults=True)
        req_args.update(self.config_args)
        params = {
            "prompt": self.convert_messages_to_prompt(request.messages),
        }
        if req_args.get("temperature"):
            params["temperature"] = req_args.get("temperature")
        if req_args.get("top_p"):
            top_p = req_args.get("top_p")  # zhipu 范围在(0,1)开区间，默认值0.7
            if top_p == 1:
                top_p = 0.7
            params["top_p"] = top_p
        return params

    def convert_messages_to_prompt(self, messages: List[ChatMessage]) -> List[Dict[str, str]]:
        prompt = []
        for message in messages:
            role = message.role
            if role in ["function"]:
                raise Exception(f"不支持的功能:{role}")
            if role == "system":  # 将system转为user   这里可以使用  CharacterGLM
                role = "user"
                content = self.prompt.format(system=message.content)
                prompt.append({"role": role, "content": content})
                prompt.append({"role": "assistant", "content": "ok"})
            else:
                content = message.content
                prompt.append({"role": role, "content": content})
        return prompt
