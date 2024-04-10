import json
import time
from typing import Dict, Iterator, List
import uuid
from adapters.base import ModelAdapter
from adapters.protocol import ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ErrorResponse
import requests
from utils.http_util import post, stream
from loguru import logger
from utils.util import num_tokens_from_string

"""
 curl -x http://127.0.0.1:7890 https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key= \
    -H 'Content-Type: application/json' \
    -X POST \
    -d '{
      "contents": [
        {"role":"user",
         "parts":[{
           "text": "你好"}]},
        {"role": "model",
         "parts":[{
           "text": "你好"}]},
        {"role": "user",
         "parts":[{
           "text": "你是谁？"}]},
      ]
    }'


{
  "candidates": [
    {
      "content": {
        "parts": [
          {
            "text": "In the tranquil village of Étoiles-sur-Mer, nestled amidst the rolling hills of 17th-century France, lived a young girl named Marie. She was known for her kind heart, inquisitive nature, and an extraordinary bond with a magical backpack she inherited from her grandmother."
          }
        ],
        "role": "model"
      },
      "finishReason": "STOP",
      "index": 0,
      "safetyRatings": [
        {
          "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
          "probability": "NEGLIGIBLE"
        },
        {
          "category": "HARM_CATEGORY_HATE_SPEECH",
          "probability": "NEGLIGIBLE"
        },
        {
          "category": "HARM_CATEGORY_HARASSMENT",
          "probability": "NEGLIGIBLE"
        },
        {
          "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
          "probability": "NEGLIGIBLE"
        }
      ]
    }
  ],
  "promptFeedback": {
    "safetyRatings": [
      {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "probability": "NEGLIGIBLE"
      },
      {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "probability": "NEGLIGIBLE"
      },
      {
        "category": "HARM_CATEGORY_HARASSMENT",
        "probability": "NEGLIGIBLE"
      },
      {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "probability": "NEGLIGIBLE"
      }
    ]
  }
}
"""

error_code_map = {
    400: 'InvalidRequestError',
    401: 'AuthenticationError',
    403: 'PermissionError',
    404: 'InvalidRequestError',
    409: 'TryAgain',
    415: 'InvalidRequestError',
    429: 'RateLimit',
    500: 'InternalServerError',
    503: 'Unavailable',
}


class GeminiAdapter(ModelAdapter):
    def __init__(self, **kwargs):
        super().__init__()
        self.api_key = kwargs.pop("api_key", None)
        self.prompt = kwargs.pop(
            "prompt", "You need to follow the system settings:{system}"
        )
        self.proxies = kwargs.pop("proxies", None)
        self.model = "gemini-pro"
        self.config_args = kwargs

    def chat_completions(
            self, request: ChatCompletionRequest
    ) -> Iterator[ChatCompletionResponse | ErrorResponse]:
        method = "generateContent"
        headers = {"Content-Type": "application/json"}
        # if request.stream:
        #     method = "streamGenerateContent"

        model = request.model
        if model not in ['gemini-pro', 'gemini-1.5-pro-latest']:
            model = 'gemini-pro'

        url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:{method}?key="
                + self.api_key
        )
        params = self.convert_2_gemini_param(request)
        response = post(url, headers=headers, proxies=self.proxies, params=params)

        if response.status_code != 200:
            openai_response = self.error_convert(response.json())
            yield ErrorResponse(status_code=response.status_code, **openai_response)
        else:
            if request.stream:  # 假的stream
                openai_response = self.response_convert_stream(response.json())
                yield ChatCompletionResponse(**openai_response)
            else:
                openai_response = self.response_convert(response.json())
                yield ChatCompletionResponse(**openai_response)

    @staticmethod
    def parse_response(data: dict) -> tuple:
        completion_tokens = 0
        completion = ''
        if 'promptFeedback' in data and 'blockReason' in data['promptFeedback']:  # 可选 SAFETY, OTHER
            finish_reason = 'content_filter'
        elif data["candidates"][0]['finishReason'] == 'STOP':
            completion = data["candidates"][0]["content"]["parts"][0]["text"]
            completion_tokens = num_tokens_from_string(completion)
            finish_reason = 'stop'
        elif data["candidates"][0]['finishReason'] == 'MAX_TOKENS':
            finish_reason = 'length'
        else:
            # SAFETY, RECITATION, OTHER
            finish_reason = 'content_filter'

        return finish_reason, completion_tokens, completion

    def response_convert_stream(self, data: dict) -> dict:
        finish_reason, completion_tokens, completion = self.parse_response(data)
        openai_response = {
            "id": str(uuid.uuid1()),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.model,
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

    def response_convert(self, data: dict) -> dict:
        finish_reason, completion_tokens, completion = self.parse_response(data)
        openai_response = {
            "id": str(uuid.uuid1()),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": completion_tokens,
                "total_tokens": completion_tokens,
            },
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": completion,
                    },
                    "index": 0,
                    "finish_reason": finish_reason,
                }
            ],
        }
        return openai_response

    """
    [
        {"role":"user",
         "parts":[{
           "text": "你好"}]},
        {"role": "model",
         "parts":[{
           "text": "你好"}]},
        {"role": "user",
         "parts":[{
           "text": "你是谁？"}]},
      ]
    """

    @staticmethod
    def error_response(code: str, message: str) -> dict:
        return {
            'error': {
                'code': code,
                'message': message,
            }
        }

    def error_convert(self, data):
        # {
        #     "error": {
        #         "code": 503,
        #         "message": "The model is overloaded. Please try again later.",
        #         "status": "UNAVAILABLE"
        #     }
        # }
        code = error_code_map.get(data['error']['code'], error_code_map[500])

        return self.error_response(code, data['error']['message'])

    def convert_messages_to_prompt(
            self, messages: List[ChatMessage]
    ) -> List[Dict[str, str]]:
        prompt = []
        for message in messages:
            role = message.role
            if role in ["function"]:
                raise Exception(f"不支持的功能:{role}")
            if role == "system":  # 将system转为user   这里可以使用  CharacterGLM
                role = "user"
                content = self.prompt.format(system=message.content)
                prompt.append({"role": role, "parts": [{"text": content}]})
                prompt.append({"role": "model", "parts": [{"text": "ok"}]})
            elif role == "assistant":
                prompt.append({"role": "model", "parts": [{"text": message.content}]})
            else:
                content = message.content
                prompt.append({"role": role, "parts": [{"text": content}]})
        return prompt

    def convert_2_gemini_param(self, request: ChatCompletionRequest):
        contents = self.convert_messages_to_prompt(request.messages)
        param = {
            "contents": contents,
            # "generationConfig": {
            #     "temperature": request.temperature if request.temperature else 0.9,  # 0-1之间
            #     "topK": 1,
            #     "topP": request.top_p if request.top_p else 1,  # 0-1之间
            #     "maxOutputTokens": request.max_length if request.max_length else 2048,
            #     "stopSequences": request.stop if request.stop else [],
            # },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                }
            ]
        }
        return param
