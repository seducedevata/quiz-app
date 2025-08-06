import sys
import json
import time
from .mcq_manager import MCQManager
from .unified_inference_manager import UnifiedInferenceManager

def stream_quiz_generation(params):
    """Stream quiz generation with real-time token output"""
    try:
        print("🚀 Starting streaming quiz generation...")
        print(f"📋 Parameters: {params}")
        
        # Initialize managers
        print("🔧 Initializing MCQ Manager...")
        mcq_manager = MCQManager()
        
        print("🔧 Initializing Inference Manager...")
        inference_manager = UnifiedInferenceManager()
        
        # Extract parameters
        topic = params.get('topic', 'General Knowledge')
        difficulty = params.get('difficulty', 'medium')
        num_questions = int(params.get('numQuestions', 5)) # Changed from num_questions to numQuestions
        mode = params.get('mode', 'standard')
        submode = params.get('gameMode', 'multiple_choice') # Changed from submode to gameMode
        question_type = params.get('questionType', 'general') # New parameter
        enable_token_streaming = params.get('enableTokenStreaming', False) # New parameter
        deepseek_model = params.get('deepSeekModel', '') # New parameter
        custom_prompt = params.get('customPrompt', '') # New parameter
        
        print(f"🎯 Generating {num_questions} {difficulty} questions about {topic}")
        
        # Simulate streaming generation steps
        steps = [
            "Analyzing topic and difficulty...",
            "Consulting knowledge base...",
            "Crafting questions...",
            "Generating answer options...",
            "Creating explanations...",
            "Finalizing quiz structure..."
        ]

        for step in steps:
            print(f"STREAM_TOKEN:{step}") # Prefix with STREAM_TOKEN to differentiate
            time.sleep(0.5)
        
        # Generate actual questions
        print("STREAM_TOKEN:Finalizing quiz generation...")
        questions = mcq_manager.generate_quiz(
            topic=topic,
            difficulty=difficulty,
            num_questions=num_questions,
            mode=mode,
            submode=submode,
            question_type=question_type,
            enable_token_streaming=enable_token_streaming,
            deepseek_model=deepseek_model,
            custom_prompt=custom_prompt
        )
        
        print("STREAM_COMPLETE:") # Signal completion of streaming tokens
        print(json.dumps(questions)) # Print final JSON
        
    except Exception as e:
        print(f"STREAM_ERROR:Error in streaming generation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        params = json.loads(sys.argv[1])
        stream_quiz_generation(params)
    else:
        print("STREAM_ERROR:No parameters provided")
        sys.exit(1)