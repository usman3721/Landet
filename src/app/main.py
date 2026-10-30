import logging

from fastapi import FastAPI

from src.app.api import routes_auth, routes_predict
from src.app.core.config import settings
from src.app.core.exceptions import register_exception_handlers
from src.app.middleware.logging_middleware import LoggingMiddleware

logging.basicConfig(level=logging.INFO)

app = FastAPI(title=settings.PROJECT_NAME)


@app.get("/health")
def health():
    return {"status": "ok"}


# link middleware
app.add_middleware(LoggingMiddleware)

# link endpoints
app.include_router(routes_auth.router, tags=["Auth"])
app.include_router(routes_predict.router, tags=["Prediction"])

# add exception handler
register_exception_handlers(app)
