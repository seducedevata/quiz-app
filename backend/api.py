from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

from knowledge_app.core.mcq_manager import get_mcq_manager

router = APIRouter()

class QuizParams(BaseModel):
    topic: str = Field(..., description="The topic for the quiz")
    difficulty: str = Field("medium", description="Difficulty level (e.g., 'easy', 'medium', 'hard', 'expert')")
    num_questions: int = Field(1, description="Number of questions to generate")
    mode: str = Field("auto", description="Generation mode ('offline', 'online', 'auto')")
    submode: str = Field("mixed", description="Question type submode (e.g., 'mixed', 'multiple_choice', 'true_false')")
    game_mode: str = Field("casual", description="Game mode (e.g., 'casual', 'timed')")
    timer: str = Field("30s", description="Timer setting for timed modes")
    enable_streaming: bool = Field(False, description="Whether to enable token streaming (not directly used by this endpoint)")

@router.post("/generate-quiz")
async def generate_quiz(params: QuizParams) -> List[Dict[str, Any]]:
    try:
        mcq_manager = get_mcq_manager()
        quiz_questions = await mcq_manager.generate_quiz_async(params.dict())
        if not quiz_questions:
            raise HTTPException(status_code=500, detail="Failed to generate quiz questions.")
        return quiz_questions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quiz generation failed: {str(e)}")
