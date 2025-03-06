import importlib
import os
import pkgutil
from pathlib import Path

from fastapi import FastAPI


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
