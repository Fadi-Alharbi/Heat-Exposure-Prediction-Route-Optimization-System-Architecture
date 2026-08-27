"""
FastAPI Application Entry Point
────────────────────────────────
Main application setup with middleware and router inclusion.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

app = FastAPI(
    title="Heat Exposure Prediction & Route Optimization API",
    description=(
        "نظام ذكي للتنبؤ بالتعرض الحراري وتحسين المسارات الخارجية\n\n"
        "An intelligent system for predicting heat exposure and optimizing "
        "outdoor routes in hot environments."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
from src.api.routes import router  # noqa: E402
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "name": "Heat Exposure Prediction & Route Optimization System",
        "version": "0.1.0",
        "docs": "/docs",
    }
