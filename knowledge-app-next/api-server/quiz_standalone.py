#!/usr/bin/env python3
"""
Standalone Quiz Generation API for Express Server Integration
This script can be called directly by the Node.js Express server.
"""

import sys
import json
import os
import importlib.util

# Add the knowledge_app source path to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def generate_quiz_standalone(topic, difficulty, num_questions, mode, submode, question_type, enable_token_streaming, deepseek_model, custom_prompt):
    """Standalone quiz generation function"""
    try:
        # Import the knowledge_app modules
        from knowledge_app.core.mcq_manager import MCQManager
        from knowledge_app.core.unified_inference_manager import UnifiedInferenceManager
        
        # Initialize managers
        mcq_manager = MCQManager()
        
        # Generate quiz
        questions = mcq_manager.generate_quiz(
            topic=topic,
            difficulty=difficulty,
            num_questions=int(num_questions),
            mode=mode,
            submode=submode,
            question_type=question_type,
            enable_token_streaming=enable_token_streaming,
            deepseek_model=deepseek_model,
            custom_prompt=custom_prompt
        )
        
        return {"status": "success", "questions": questions}
    except ImportError as e:
        error_msg = f"Import error - check Python environment: {str(e)}"
        print(f"ERROR: {error_msg}", file=sys.stderr)
        return {"status": "error", "error": error_msg}
    except Exception as e:
        error_msg = f"Quiz generation failed: {str(e)}"
        print(f"ERROR: {error_msg}", file=sys.stderr)
        return {"status": "error", "error": error_msg}

def generate_mock_quiz(topic, difficulty, num_questions):
    """Generate mock quiz for testing when backend is not available"""
    questions = []
    for i in range(int(num_questions)):
        question = {
            "id": f"mock_{i}",
            "question": f"What is a sample {difficulty} question about {topic}? (Question {i+1})",
            "options": [
                f"Option A for {topic}",
                f"Option B for {topic}",
                f"Option C for {topic}",
                f"Option D for {topic}"
            ],
            "correctAnswerId": f"mock_{i}_option_0",
            "explanation": f"This is a mock explanation for the {topic} question.",
            "topic": topic,
            "difficulty": difficulty,
            "timestamp": "2025-08-07T12:00:00Z"
        }
        questions.append(question)
    
    return {"status": "success", "questions": questions, "mock": True}

if __name__ == "__main__":
    if len(sys.argv) != 10:
        print(json.dumps({
            "status": "error", 
            "error": "Invalid arguments. Expected: topic difficulty num_questions mode submode question_type enable_streaming deepseek_model custom_prompt"
        }))
        sys.exit(1)
    
    topic = sys.argv[1]
    difficulty = sys.argv[2]
    num_questions = sys.argv[3]
    mode = sys.argv[4]
    submode = sys.argv[5]
    question_type = sys.argv[6]
    enable_token_streaming = sys.argv[7].lower() == 'true'
    deepseek_model = sys.argv[8]
    custom_prompt = sys.argv[9]
    
    # Try real quiz generation first, fall back to mock if it fails
    result = generate_quiz_standalone(topic, difficulty, num_questions, mode, submode, question_type, enable_token_streaming, deepseek_model, custom_prompt)
    
    # If real generation failed, use mock data as fallback
    if result["status"] == "error":
        print(f"Real quiz generation failed: {result['error']}", file=sys.stderr)
        print("Falling back to mock quiz generation...", file=sys.stderr)
        result = generate_mock_quiz(topic, difficulty, num_questions)
    
    print(json.dumps(result))
