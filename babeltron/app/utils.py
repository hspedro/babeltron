import hashlib
import importlib
import json
import os
import pkgutil
from pathlib import Path
from typing import Any

import orjson
from fastapi import FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi_cache import Coder


def get_model_path() -> str:
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        return env_path

    # Get the directory two levels up from utils.py
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent

    possible_paths = [
        Path("/models"),
        project_root / "models",
        Path("./models"),
    ]

    for path in possible_paths:
        if path.exists() and any(path.glob("**/config.json")):
            return str(path)

    return "./models"


def include_routers(app: FastAPI):
    routers_package = "babeltron.app.routers"
    routers_path = os.path.join(os.path.dirname(__file__), "routers")

    for _, module_name, _ in pkgutil.iter_modules([routers_path]):
        module = importlib.import_module(f"{routers_package}.{module_name}")
        if hasattr(module, "router"):
            app.include_router(module.router)


def cache_key_builder(
    func,
    namespace: str = "",
    request: Request = None,
    response: Response = None,
    *args,
    **kwargs,
) -> str:
    if request is None:
        return ""

    body_data = {}
    if hasattr(request, "state") and hasattr(request.state, "body"):
        try:
            body_data = json.loads(request.state.body)
        except (json.JSONDecodeError, AttributeError):
            pass

    src_lang = body_data.get("src_lang", "")
    dst_lang = body_data.get("dst_lang", "")
    text = body_data.get("text", "")

    text_md5 = hashlib.md5(text.encode()).hexdigest() if text else ""

    return ":".join(
        [
            namespace,
            src_lang,
            dst_lang,
            text_md5,
        ]
    )


class ORJsonCoder(Coder):
    @classmethod
    def encode(cls, value: Any) -> bytes:
        return orjson.dumps(
            value,
            default=jsonable_encoder,
            option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY,
        )

    @classmethod
    def decode(cls, value: bytes) -> Any:
        return orjson.loads(value)
