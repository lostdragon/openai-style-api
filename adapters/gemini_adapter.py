import base64
import json
import mimetypes
import time
from typing import Dict, Iterator, List
import uuid

import re
import requests

from adapters.base import ModelAdapter
from adapters.protocol import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse, ChatCompletionMessageParam
from utils.http_util import post, stream
from loguru import logger

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
    _models = None

    def __init__(self, **kwargs):
        super().__init__()
        self.api_key = kwargs.pop("api_key", None)
        self.proxies = kwargs.pop("proxies", None)
        self.model = "gemini-1.5-flash-latest"
        self.config_args = kwargs
        self.safe_settings = [
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

    def chat_completions(
            self, request: ChatCompletionRequest
    ) -> Iterator[ChatCompletionResponse | ErrorResponse]:
        method = "streamGenerateContent" if request.stream else "generateContent"
        headers = {"Content-Type": "application/json"}

        if request.model in self.get_models():
            self.model = request.model

        params = self.convert_2_gemini_params(request)
        url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:{method}?key="
                + self.api_key
        )

        if request.stream:
            url = url + '&alt=sse'
            response = stream(url, headers=headers, proxies=self.proxies, params=params)
        else:
            response = post(url, headers=headers, proxies=self.proxies, params=params)

        if response.status_code != 200:
            openai_response = self.error_convert(response.json())
            yield ErrorResponse(status_code=response.status_code, **openai_response)
        else:
            last_chunk = None
            if request.stream:
                message_id = str(uuid.uuid1())
                for chunk in response.iter_lines(chunk_size=1024):
                    # 移除头部data: 字符
                    decoded_line = chunk.decode('utf-8')
                    logger.debug(f"decoded_line: {decoded_line}")
                    decoded_line = decoded_line.lstrip("data:").strip()
                    if "[DONE]" == decoded_line:
                        break
                    if decoded_line:
                        last_chunk = json.loads(decoded_line)
                        yield ChatCompletionResponse(**self.response_convert_stream(message_id, last_chunk))

                yield ChatCompletionResponse(**self.response_convert_stream(message_id, last_chunk, is_last=True))
            else:
                openai_response = self.response_convert(response.json())
                yield ChatCompletionResponse(**openai_response)

    def get_models(self):
        if self._models is None:
            self._models = []
            url = (
                    f"https://generativelanguage.googleapis.com/v1beta/models?key="
                    + self.api_key
            )
            headers = {"Content-Type": "application/json"}
            response = requests.get(url, headers=headers, proxies=self.proxies)
            if response.status_code == 200:
                result = response.json()
                for model in result['models']:
                    matches = re.match('models/(gemini.*)', model['name'])
                    if matches:
                        self._models.append(matches.group(1))

        return self._models

    @staticmethod
    def parse_response(data: dict) -> tuple:
        completion_tokens = 0
        completion = ''
        finish_reason = None
        if data:
            if 'promptFeedback' in data and 'blockReason' in data['promptFeedback']:  # 可选 SAFETY, OTHER
                finish_reason = 'content_filter'
            elif data["candidates"][0]['finishReason'] == 'STOP':
                completion = data["candidates"][0]["content"]["parts"][0]["text"]
                finish_reason = 'stop'
            elif data["candidates"][0]['finishReason'] == 'MAX_TOKENS':
                finish_reason = 'length'
            else:
                # SAFETY, RECITATION, OTHER
                finish_reason = 'content_filter'

        return finish_reason, completion_tokens, completion

    def response_convert_stream(self, message_id: str, data: dict, is_last: bool = False) -> dict:
        completion = ''
        finish_reason = None
        if data:
            if 'promptFeedback' in data and 'blockReason' in data['promptFeedback']:  # 可选 SAFETY, OTHER
                finish_reason = 'content_filter'
            elif data["candidates"][0]['finishReason'] == 'STOP':
                completion = data["candidates"][0]["content"]["parts"][0]["text"]
                finish_reason = 'stop'
            elif data["candidates"][0]['finishReason'] == 'MAX_TOKENS':
                finish_reason = 'length'
            else:
                # SAFETY, RECITATION, OTHER
                finish_reason = 'content_filter'

        if not is_last:
            finish_reason = None
        else:
            completion = ''

        openai_response = {
            "id": message_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.model,
            "usage": {
                "prompt_tokens": data['usageMetadata'].get('promptTokenCount', 0) if data['usageMetadata'] else 0,
                "completion_tokens": data['usageMetadata'].get('candidatesTokenCount', 0) if data[
                    'usageMetadata'] else 0,
                "total_tokens": data['usageMetadata'].get('totalTokenCount', 0) if data['usageMetadata'] else 0,
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

    @staticmethod
    def get_image_format(data: bytes) -> str:
        data = data[:20]
        if b'GIF' in data:
            return 'gif'
        elif b'PNG' in data:
            return 'png'
        elif b'JFIF' in data:
            return 'jpg'
        return ''

    @classmethod
    def get_content_type(cls, data: bytes, return_format: bool = False) -> str | tuple:
        image_format = cls.get_image_format(data)
        if image_format:
            mimetype = mimetypes.guess_type('image.{}'.format(image_format))[0]
        else:
            mimetype = None

        if return_format:
            return image_format, mimetype
        else:
            return mimetype

    @classmethod
    def get_image(cls, image_url: str) -> (str, str):
        """
        根据图片URL获取图片信息
        :param image_url: 图片URL
        :type image_url: str
        :return:
        :rtype:
        """
        if re.match("^(http|https)://", image_url):
            content = requests.get(image_url).content
        else:
            matches = re.match('data:(.*?);base64,(.*)', image_url)
            if matches:
                image_media_type = matches.group(1)
                image_data = matches.group(2)
                return image_media_type, image_data
            else:
                content = base64.b64decode(image_url)
        if content:
            image_media_type = cls.get_content_type(content)
            if image_media_type:
                image_data = base64.b64encode(content).decode("utf-8")
                return image_media_type, image_data

        return None, None

    def convert_messages_to_prompt(
            self, messages: List[ChatCompletionMessageParam]
    ) -> Dict:
        contents = []
        system_parts = []

        for message in messages:
            role = message.role
            if role in ["function"]:
                raise Exception(f"不支持的功能:{role}")
            else:
                parts = []
                if isinstance(message.content, str):
                    parts.append({"text": message.content})
                else:
                    for c in message.content:
                        if c.type == 'text':
                            parts.append({'text': c.text})
                        elif c.type == 'image_url':
                            image_media_type, image_data = self.get_image(c.image_url.url)
                            if image_media_type and image_data:
                                parts.append({'inlineData': {'mimeType': image_media_type, 'data': image_data}})

                if parts:
                    if role == 'system':
                        system_parts = parts
                    elif role == 'assistant':
                        contents.append({"role": "model", "parts": parts})
                    else:
                        contents.append({"role": role, "parts": parts})

        result = {
            'contents': contents,
        }

        if system_parts:
            result['systemInstruction'] = {"role": "", "parts": system_parts}

        return result

    def convert_2_gemini_params(self, request: ChatCompletionRequest):
        params = self.convert_messages_to_prompt(request.messages)
        params.update({
            "generationConfig": {
                "temperature": request.temperature if request.temperature else 0.9,  # 0-1之间
                "topK": 1,
                "topP": request.top_p if request.top_p else 1,  # 0-1之间
                "maxOutputTokens": request.max_length if request.max_length else 2048,
                "stopSequences": request.stop if request.stop else [],
            },
            "safetySettings": self.safe_settings
        })
        return params
