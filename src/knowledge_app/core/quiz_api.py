import sys
import json
from .mcq_manager import MCQManager
from .unified_inference_manager import UnifiedInferenceManager

def generate_quiz_api(topic, difficulty, num_questions, mode, submode, question_type, enable_token_streaming, deepseek_model, custom_prompt):
    """API endpoint for quiz generation"""
    try:
        # Initialize managers
        mcq_manager = MCQManager()
        unified_inference_manager = UnifiedInferenceManager()
        
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
        
        return json.dumps(questions)
    except Exception as e:
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    topic = sys.argv[1]
    difficulty = sys.argv[2]
    num_questions = sys.argv[3]
    mode = sys.argv[4]
    submode = sys.argv[5]
    question_type = sys.argv[6]
    enable_token_streaming = sys.argv[7].lower() == 'true'
    deepseek_model = sys.argv[8]
    custom_prompt = sys.argv[9]
    
    result = generate_quiz_api(topic, difficulty, num_questions, mode, submode, question_type, enable_token_streaming, deepseek_model, custom_prompt)
    print(result)
