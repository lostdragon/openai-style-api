import requests
import json
from loguru import logger

api_timeout_seconds = 300


def post(
        api_url, headers: dict, params: dict, timeout=api_timeout_seconds, proxies=None
):
    resp = None
    try:
        resp = requests.post(
            url=api_url,
            headers=headers,
            data=json.dumps(params),
            timeout=timeout,
            proxies=proxies,
        )
        return resp
    except Exception as e:
        logger.exception("请求异常", e)
        raise e
    finally:
        logger.debug(
            f"【http.post】 请求url：{api_url}, headers:{headers}, params:{params}, resp:{resp_text(resp)}"
        )


def resp_text(resp):
    resp_str = f"status_code:{resp.status_code}: {resp.text}"
    return resp_str


def stream(
        api_url, headers: dict, params: dict, timeout=api_timeout_seconds, proxies=None
):
    resp = None
    try:
        resp = requests.post(
            api_url,
            stream=True,
            headers=headers,
            json=params,
            # data=json.dumps(params),
            timeout=timeout,
            proxies=proxies,
        )
        return resp
    except Exception as e:
        logger.exception("请求异常", e)
    finally:
        logger.debug(
            f"【http.stream】 请求url：{api_url}, headers:{headers}, params:{params}"
        )
