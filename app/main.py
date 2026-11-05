
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.endpoints.documents import router as documents_router
from app.api.v1.endpoints.agent import router as agent_router
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Enterprise Agentic RAG application...")
    yield
    print("Shutting down Enterprise Agentic RAG application...")


app = FastAPI(
    title=settings.APP_NAME,
    description="Enterprise Agentic RAG API",
    version="1.0.0",
    debug=settings.DEBUG,
    lifespan=lifespan,
)


# Document API routes
app.include_router(
    documents_router,
    prefix=settings.API_V1_PREFIX,
)


# Agent API routes
app.include_router(
    agent_router,
    prefix=f"{settings.API_V1_PREFIX}/agent",
)


@app.get("/")
async def root():
    return {
        "message": "Enterprise Agentic RAG API is running",
        "environment": settings.APP_ENV,
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "application": settings.APP_NAME,
    }
