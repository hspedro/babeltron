import importlib
import os
import pkgutil

from fastapi import FastAPI


def include_routers(app: FastAPI):
    routers_package = "babeltron.app.routers"
    routers_path = os.path.join(os.path.dirname(__file__), "routers")

    for _, module_name, _ in pkgutil.iter_modules([routers_path]):
        module = importlib.import_module(f"{routers_package}.{module_name}")
        if hasattr(module, "router"):
            app.include_router(module.router)
