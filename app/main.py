from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.routes import router
from app.services.index_manager import initialize_index, start_auto_refresh


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ✅ Fast, non-blocking startup
    try:
        initialize_index()
    except Exception as e:
        print(f"Index init failed: {e}")

    # ✅ Start refresh in background INSIDE the function
    start_auto_refresh(900)

    yield  # 👈 app is READY here

    print("App shutting down")


app = FastAPI(
    title="Job Recommendation API",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/health")
def health():
    return {"status": "ok"}