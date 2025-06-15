"""SmartRAG main application entry point."""

import uvicorn

from src.api.main import app
from src.core.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.worker_count,
        log_level=settings.log_level.lower(),
        access_log=True,
        use_colors=True,
    )