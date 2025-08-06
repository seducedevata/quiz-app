from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import router as api_router

app = FastAPI(
    title="Knowledge App Backend",
    description="API for generating quizzes and managing knowledge content.",
    version="0.1.0",
)

# Configure CORS
origins = [
    "http://localhost:5173",  # React app development server
    "http://127.0.0.1:5173",
    # Add other origins as needed for deployment
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Knowledge App Backend!"}