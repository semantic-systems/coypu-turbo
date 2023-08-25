"""This module provides a ChatGPT-compatible Restful API for chat completion.

Usage:

python3 -m fastchat.serve.api

Reference: https://platform.openai.com/docs/api-reference/chat/create
"""

from typing import Optional, Dict, List, Any, Union

import json
import logging
import requests

import fastapi
import httpx
import uvicorn
import aiohttp

from fastchat.protocol.openai_api_protocol import (
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice, DeltaMessage, ChatCompletionStreamResponse, UsageInfo)
# from fastchat.conversation import get_default_conv_template, SeparatorStyle
# from fastchat.serve.inference import compute_skip_echo_len
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseSettings

from fastchat.constants import ErrorCode, WORKER_API_TIMEOUT
import shortuuid


logger = logging.getLogger(__name__)


app = fastapi.FastAPI()
headers = {"User-Agent": "FastChat API Server"}


class CustomChatCompletionRequest(ChatCompletionRequest):
    key: str
    temperature: Optional[float] = 1


# @app.post("/")
# async def create_chat_completion(request: CustomChatCompletionRequest):
#     authenticated = False
#
#     if request.key == 'M7ZQL9ELMSDXXE86': authenticated = True
#
#     if authenticated == False:
#         response = {'error': 'no valid API key'}
#         http_code = 401
#
#     else:
#         """Creates a completion for the chat message"""
#         payload, skip_echo_len = generate_payload(
#             request.model,
#             request.messages,
#             temperature=request.temperature,
#             max_tokens=request.max_tokens,
#             stop=request.stop)
#
#         choices = []
#         # TODO: batch the requests. maybe not necessary if using CacheFlow worker
#         for i in range(request.n):
#             content = await chat_completion(request.model, payload, skip_echo_len)
#             choices.append(
#                 ChatCompletionResponseChoice(
#                     index=i,
#                     message=ChatMessage(role="assistant", content=content),
#                     # TODO: support other finish_reason
#                     finish_reason="stop")
#             )
#
#         # TODO: support usage field
#         # "usage": {
#         #     "prompt_tokens": 9,
#         #     "completion_tokens": 12,
#         #     "total_tokens": 21
#         # }
#         response = ChatCompletionResponse(choices=choices)
#         http_code = 200
#
#     return response, http_code

fetch_timeout = aiohttp.ClientTimeout(total=3 * 3600)


async def fetch_remote(url, pload=None, name=None):
    async with aiohttp.ClientSession(timeout=fetch_timeout) as session:
        async with session.post(url, json=pload) as response:
            chunks = []
            async for chunk, _ in response.content.iter_chunks():
                chunks.append(chunk)
        output = b"".join(chunks)

    if name is not None:
        res = json.loads(output)
        if name != "":
            res = res[name]
        return res

    return output


class AppSettings(BaseSettings):
    # The address of the model controller.
    controller_address: str = "http://vicuna_controller:5287"
    api_keys: Optional[List[str]] = None


app_settings = AppSettings()
async def check_model(request) -> Optional[JSONResponse]:
    controller_address = app_settings.controller_address
    ret = None

    models = await fetch_remote(controller_address + "/list_models", None, "models")
    if request.model not in models:
        ret = create_error_response(
            ErrorCode.INVALID_MODEL,
            f"Only {'&&'.join(models)} allowed now, your model {request.model}",
        )
    return ret


def create_error_response(code: int, message: str) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(message=message, code=code).dict(), status_code=400
    )

def check_requests(request) -> Optional[JSONResponse]:
    # Check all params
    if request.max_tokens is not None and request.max_tokens <= 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.max_tokens} is less than the minimum of 1 - 'max_tokens'",
        )
    if request.n is not None and request.n <= 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.n} is less than the minimum of 1 - 'n'",
        )
    if request.temperature is not None and request.temperature < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is less than the minimum of 0 - 'temperature'",
        )
    if request.temperature is not None and request.temperature > 2:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is greater than the maximum of 2 - 'temperature'",
        )
    if request.top_p is not None and request.top_p < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is less than the minimum of 0 - 'top_p'",
        )
    if request.top_p is not None and request.top_p > 1:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is greater than the maximum of 1 - 'temperature'",
        )
    if request.stop is not None and (
        not isinstance(request.stop, str) and not isinstance(request.stop, list)
    ):
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.stop} is not valid under any of the given schemas - 'stop'",
        )

    return None


async def get_worker_address(model_name: str) -> str:
    """
    Get worker address based on the requested model

    :param model_name: The worker's model name
    :return: Worker address from the controller
    :raises: :class:`ValueError`: No available worker for requested model
    """
    controller_address = app_settings.controller_address
    worker_addr = await fetch_remote(
        controller_address + "/get_worker_address", {"model": model_name}, "address"
    )

    # No available worker
    if worker_addr == "":
        raise ValueError(f"No available worker for {model_name}")
    logger.debug(f"model_name: {model_name}, worker_addr: {worker_addr}")
    return worker_addr


async def check_length(request, prompt, max_tokens, worker_addr):
    context_len = await fetch_remote(
        worker_addr + "/model_details", {"model": request.model}, "context_length"
    )
    token_num = await fetch_remote(
        worker_addr + "/count_token",
        {"model": request.model, "prompt": prompt},
        "count",
    )
    if token_num + max_tokens > context_len:
        return create_error_response(
            ErrorCode.CONTEXT_OVERFLOW,
            f"This model's maximum context length is {context_len} tokens. "
            f"However, you requested {max_tokens + token_num} tokens "
            f"({token_num} in the messages, "
            f"{max_tokens} in the completion). "
            f"Please reduce the length of the messages or completion.",
        )
    else:
        return None


async def DeltaMessage(payload: Dict[str, Any], worker_addr: str):
    controller_address = app_settings.controller_address
    async with httpx.AsyncClient() as client:
        delimiter = b"\0"
        async with client.stream(
            "POST",
            worker_addr + "/worker_generate_stream",
            headers=headers,
            json=payload,
            timeout=WORKER_API_TIMEOUT,
        ) as response:
            # content = await response.aread()
            async for raw_chunk in response.aiter_raw():
                for chunk in raw_chunk.split(delimiter):
                    if not chunk:
                        continue
                    data = json.loads(chunk.decode())
                    yield data


async def chat_completion_stream_generator(
    model_name: str, gen_params: Dict[str, Any], n: int, worker_addr: str
) -> Generator[str, Any, None]:
    """
    Event stream format:
    https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format
    """
    id = f"chatcmpl-{shortuuid.random()}"
    finish_stream_events = []
    for i in range(n):
        # First chunk with role
        choice_data = ChatCompletionResponseStreamChoice(
            index=i,
            delta=DeltaMessage(role="assistant"),
            finish_reason=None,
        )
        chunk = ChatCompletionStreamResponse(
            id=id, choices=[choice_data], model=model_name
        )
        yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

        previous_text = ""
        async for content in generate_completion_stream(gen_params, worker_addr):
            if content["error_code"] != 0:
                yield f"data: {json.dumps(content, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            decoded_unicode = content["text"].replace("\ufffd", "")
            delta_text = decoded_unicode[len(previous_text) :]
            previous_text = decoded_unicode

            if len(delta_text) == 0:
                delta_text = None
            choice_data = ChatCompletionResponseStreamChoice(
                index=i,
                delta=DeltaMessage(content=delta_text),
                finish_reason=content.get("finish_reason", None),
            )
            chunk = ChatCompletionStreamResponse(
                id=id, choices=[choice_data], model=model_name
            )
            if delta_text is None:
                if content.get("finish_reason", None) is not None:
                    finish_stream_events.append(chunk)
                continue
            yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"
    # There is not "content" field in the last delta message, so exclude_none to exclude field "content".
    for finish_chunk in finish_stream_events:
        yield f"data: {finish_chunk.json(exclude_none=True, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"

async def get_gen_params(
    model_name: str,
    worker_addr: str,
    messages: Union[str, List[Dict[str, str]]],
    *,
    temperature: float,
    top_p: float,
    max_tokens: Optional[int],
    echo: Optional[bool],
    stream: Optional[bool],
    stop: Optional[Union[str, List[str]]],
) -> Dict[str, Any]:
    conv = await get_conv(model_name, worker_addr)
    conv = Conversation(
        name=conv["name"],
        system_template=conv["system_template"],
        system_message=conv["system_message"],
        roles=conv["roles"],
        messages=list(conv["messages"]),  # prevent in-place modification
        offset=conv["offset"],
        sep_style=SeparatorStyle(conv["sep_style"]),
        sep=conv["sep"],
        sep2=conv["sep2"],
        stop_str=conv["stop_str"],
        stop_token_ids=conv["stop_token_ids"],
    )

    if isinstance(messages, str):
        prompt = messages
    else:
        for message in messages:
            msg_role = message["role"]
            if msg_role == "system":
                conv.set_system_message(message["content"])
            elif msg_role == "user":
                conv.append_message(conv.roles[0], message["content"])
            elif msg_role == "assistant":
                conv.append_message(conv.roles[1], message["content"])
            else:
                raise ValueError(f"Unknown role: {msg_role}")

        # Add a blank message for the assistant.
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

    if max_tokens is None:
        max_tokens = 512
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_tokens,
        "echo": echo,
        "stream": stream,
    }

    if not stop:
        gen_params.update(
            {"stop": conv.stop_str, "stop_token_ids": conv.stop_token_ids}
        )
    else:
        gen_params.update({"stop": stop})

    logger.debug(f"==== request ====\n{gen_params}")
    return gen_params

conv_template_map = {}


async def get_conv(model_name: str, worker_addr: str):
    conv_template = conv_template_map.get((worker_addr, model_name))
    if conv_template is None:
        conv_template = await fetch_remote(
            worker_addr + "/worker_get_conv_template", {"model": model_name}, "conv"
        )
        conv_template_map[(worker_addr, model_name)] = conv_template
    return conv_template


@app.post("/")
async def create_chat_completion(request: CustomChatCompletionRequest):
    authenticated = False

    if request.key == 'M7ZQL9ELMSDXXE86': authenticated = True

    if authenticated == False:
        response = {'error': 'no valid API key'}
        http_code = 401

    else:

        # url = 'http://0.0.0.0'
        # payload = open("request.json")
        # headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
        # response = requests.post(url, data=payload, headers=headers)

        # check_model, check_requests, get_worker_address, get_gen_params, check_length, chat_completion_stream_generator, StreamingResponse, create_error_response, ErrorCode, UsageInfo
        """Creates a completion for the chat message"""
        error_check_ret = await check_model(request)
        if error_check_ret is not None:
            return error_check_ret
        error_check_ret = check_requests(request)
        if error_check_ret is not None:
            return error_check_ret

        worker_addr = await get_worker_address(request.model)

        gen_params = await get_gen_params(
            request.model,
            worker_addr,
            request.messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            echo=False,
            stream=request.stream,
            stop=request.stop,
        )
        error_check_ret = await check_length(
            request,
            gen_params["prompt"],
            gen_params["max_new_tokens"],
            worker_addr,
        )
        if error_check_ret is not None:
            return error_check_ret

        if request.stream:
            generator = chat_completion_stream_generator(
                request.model, gen_params, request.n, worker_addr
            )
            return StreamingResponse(generator, media_type="text/event-stream")

        choices = []
        chat_completions = []
        for i in range(request.n):
            content = asyncio.create_task(generate_completion(gen_params, worker_addr))
            chat_completions.append(content)
        try:
            all_tasks = await asyncio.gather(*chat_completions)
        except Exception as e:
            return create_error_response(ErrorCode.INTERNAL_ERROR, str(e))
        usage = UsageInfo()
        for i, content in enumerate(all_tasks):
            if content["error_code"] != 0:
                return create_error_response(content["error_code"], content["text"])
            choices.append(
                ChatCompletionResponseChoice(
                    index=i,
                    message=ChatMessage(role="assistant", content=content["text"]),
                    finish_reason=content.get("finish_reason", "stop"),
                )
            )
            if "usage" in content:
                task_usage = UsageInfo.parse_obj(content["usage"])
                for usage_key, usage_value in task_usage.dict().items():
                    setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)
        response = ChatCompletionResponse(model=request.model, choices=choices, usage=usage)
        http_code = 200
    return response, http_code



# def generate_payload(model_name: str, messages: List[Dict[str, str]],
#                      *, temperature: float, max_tokens: int, stop: Union[str, None]):
#     is_chatglm = "chatglm" in model_name.lower()
#     # TODO(suquark): The template is currently a reference. Here we have to make a copy.
#     # We use create a template factory to avoid this.
#     conv = get_default_conv_template(model_name).copy()
#
#     # TODO(suquark): Conv.messages should be a list. But it is a tuple now.
#     #  We should change it to a list.
#     conv.messages = list(conv.messages)
#
#     for message in messages:
#         msg_role = message["role"]
#         if msg_role == "system":
#             conv.system = message["content"]
#         elif msg_role == "user":
#             conv.append_message(conv.roles[0], message["content"])
#         elif msg_role == "assistant":
#             conv.append_message(conv.roles[1], message["content"])
#         else:
#             raise ValueError(f"Unknown role: {msg_role}")
#
#     # Add a blank message for the assistant.
#     conv.append_message(conv.roles[1], None)
#
#     if is_chatglm:
#         prompt = conv.messages[conv.offset:]
#     else:
#         prompt = conv.get_prompt()
#     skip_echo_len = compute_skip_echo_len(model_name, conv, prompt)
#
#     if stop is None:
#         stop = conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2
#
#     # TODO(suquark): We should get the default `max_new_tokens`` from the model.
#     if max_tokens is None:
#         max_tokens = 512
#
#     payload = {
#         "model": model_name,
#         "prompt": prompt,
#         "temperature": temperature,
#         "max_new_tokens": max_tokens,
#         "stop": stop,
#     }
#
#     logger.debug(f"==== request ====\n{payload}")
#     return payload, skip_echo_len
#
#
# async def chat_completion(model_name: str, payload: Dict[str, Any], skip_echo_len: int):
#     controller_url = "http://vicuna_controller:5287"
#     async with httpx.AsyncClient() as client:
#         ret = await client.post(controller_url + "/get_worker_address", json={"model": model_name})
#         worker_addr = ret.json()["address"]
#         # No available worker
#         if worker_addr == "":
#             raise ValueError(f"No available worker for {model_name}")
#
#         logger.debug(f"model_name: {model_name}, worker_addr: {worker_addr}")
#
#         output = ""
#         delimiter = b"\0"
#         async with client.stream("POST", worker_addr + "/worker_generate_stream",
#                                  headers=headers, json=payload, timeout=20) as response:
#             content = await response.aread()
#
#         for chunk in content.split(delimiter):
#             if not chunk:
#                 continue
#             data = json.loads(chunk.decode())
#             if data["error_code"] == 0:
#                 output = data["text"][skip_echo_len:].strip()
#
#         return output


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=5285, reload=True)
