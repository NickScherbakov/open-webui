import logging
import json
import aiohttp
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from open_webui.utils.auth import get_admin_user, get_verified_user

log = logging.getLogger(__name__)

router = APIRouter()

class YandexGPTConfigForm(BaseModel):
    ENABLE_YANDEXGPT_API: Optional[bool] = None
    YANDEXGPT_API_KEY: str
    YANDEXGPT_FOLDER_ID: str
    YANDEXGPT_API_BASE_URL: str
    YANDEXGPT_MODELS: List[str]

class YandexGPTAdapter:
    def __init__(self, api_key: str, folder_id: str, base_url: str):
        self.api_key = api_key
        self.folder_id = folder_id
        self.base_url = base_url.rstrip('/')

    def convert_openai_to_yandex(self, openai_request: Dict[str, Any]) -> Dict[str, Any]:
        model = openai_request.get("model", "yandexgpt")
        return {
            "modelUri": f"gpt://{self.folder_id}/{model}/latest",
            "completionOptions": {
                "stream": openai_request.get("stream", False),
                "temperature": openai_request.get("temperature", 0.6),
                "maxTokens": str(openai_request.get("max_tokens", 2000))
            },
            "messages": self._convert_messages(openai_request.get("messages", []))
        }

    def _convert_messages(self, openai_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        yandex_messages = []
        for msg in openai_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            yandex_role = "assistant" if role == "assistant" else "system" if role == "system" else "user"
            yandex_messages.append({
                "role": yandex_role,
                "text": content
            })
        return yandex_messages

    def convert_yandex_to_openai(self, yandex_response: Dict[str, Any], model: str = "yandexgpt") -> Dict[str, Any]:
        result = yandex_response.get("result", {})
        alternatives = result.get("alternatives", [])
        if not alternatives:
            raise ValueError("No alternatives in YandexGPT response")
        alternative = alternatives[0]
        message = alternative.get("message", {})
        text = message.get("text", "")
        return {
            "id": f"yandex-{hash(text)}",
            "object": "chat.completion",
            "created": int(result.get("usage", {}).get("completionTokens", 0)),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": int(result.get("usage", {}).get("inputTextTokens", 0)),
                "completion_tokens": int(result.get("usage", {}).get("completionTokens", 0)),
                "total_tokens": int(result.get("usage", {}).get("totalTokens", 0))
            }
        }

@router.get("/config")
async def get_yandexgpt_config(request: Request, user=Depends(get_admin_user)):
    return {
        "ENABLE_YANDEXGPT_API": request.app.state.config.ENABLE_YANDEXGPT_API,
        "YANDEXGPT_API_KEY": request.app.state.config.YANDEXGPT_API_KEY,
        "YANDEXGPT_FOLDER_ID": request.app.state.config.YANDEXGPT_FOLDER_ID,
        "YANDEXGPT_API_BASE_URL": request.app.state.config.YANDEXGPT_API_BASE_URL,
        "YANDEXGPT_MODELS": request.app.state.config.YANDEXGPT_MODELS,
    }

@router.post("/config/update")
async def update_yandexgpt_config(
    request: Request, form_data: YandexGPTConfigForm, user=Depends(get_admin_user)
):
    if form_data.ENABLE_YANDEXGPT_API is not None:
        request.app.state.config.ENABLE_YANDEXGPT_API = form_data.ENABLE_YANDEXGPT_API
    request.app.state.config.YANDEXGPT_API_KEY = form_data.YANDEXGPT_API_KEY
    request.app.state.config.YANDEXGPT_FOLDER_ID = form_data.YANDEXGPT_FOLDER_ID
    request.app.state.config.YANDEXGPT_API_BASE_URL = form_data.YANDEXGPT_API_BASE_URL
    request.app.state.config.YANDEXGPT_MODELS = form_data.YANDEXGPT_MODELS
    return {"status": "success"}

@router.get("/models")
async def get_yandexgpt_models(request: Request, user=Depends(get_verified_user)):
    if not request.app.state.config.ENABLE_YANDEXGPT_API:
        raise HTTPException(status_code=400, detail="YandexGPT API is disabled")
    api_key = request.app.state.config.YANDEXGPT_API_KEY
    folder_id = request.app.state.config.YANDEXGPT_FOLDER_ID
    if not api_key or not folder_id:
        raise HTTPException(status_code=400, detail="YandexGPT API key or folder ID not configured")
    models = request.app.state.config.YANDEXGPT_MODELS
    return {
        "object": "list",
        "data": [
            {
                "id": model,
                "object": "model",
                "owned_by": "yandex",
                "permission": [],
            }
            for model in models
        ],
    }

@router.post("/chat/completions")
async def yandexgpt_chat_completions(
    request: Request,
    form_data: Dict[str, Any],
    user=Depends(get_verified_user)
):
    if not request.app.state.config.ENABLE_YANDEXGPT_API:
        raise HTTPException(status_code=400, detail="YandexGPT API is disabled")
    api_key = request.app.state.config.YANDEXGPT_API_KEY
    folder_id = request.app.state.config.YANDEXGPT_FOLDER_ID
    base_url = request.app.state.config.YANDEXGPT_API_BASE_URL
    if not api_key or not folder_id:
        raise HTTPException(status_code=400, detail="YandexGPT API key or folder ID not configured")
    try:
        adapter = YandexGPTAdapter(api_key, folder_id, base_url)
        yandex_request = adapter.convert_openai_to_yandex(form_data)
        headers = {
            "Authorization": f"Api-Key {api_key}",
            "Content-Type": "application/json"
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/foundationModels/v1/completion",
                headers=headers,
                json=yandex_request
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    log.error(f"YandexGPT API error: {response.status} - {error_text}")
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"YandexGPT API error: {error_text}"
                    )
                yandex_response = await response.json()
                openai_response = adapter.convert_yandex_to_openai(
                    yandex_response,
                    form_data.get("model", "yandexgpt")
                )
                return openai_response
    except Exception as e:
        log.error(f"Error in YandexGPT chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))
