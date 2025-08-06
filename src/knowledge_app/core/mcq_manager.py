import json
import random

class MCQManager:
    def __init__(self):
        self.unified_inference_manager = UnifiedInferenceManager()

    def generate_quiz(self, topic, difficulty, num_questions, mode, submode, question_type, enable_token_streaming, deepseek_model, custom_prompt):
        print(f"Generating quiz: Topic={topic}, Difficulty={difficulty}, Num_questions={num_questions}, Mode={mode}, Submode={submode}, QuestionType={question_type}, Streaming={enable_token_streaming}, DeepSeekModel={deepseek_model}, CustomPrompt={custom_prompt}")

        if enable_token_streaming:
            # For streaming, we assume num_questions is 1 as it's usually one question at a time
            # The token_callback would be handled by the API server directly.
            # Here, we just call the streaming method and return a placeholder.
            # The actual streaming data will be sent via WebSocket.
            # The quiz_api.py will need to be updated to handle the streaming response.
            print("Calling streaming generation...")
            # This will initiate the streaming process in the background
            # The actual questions will be streamed via WebSocket
            # We return a dummy response for the initial API call
            return [{
                "id": "streaming-placeholder",
                "question": "Generating question via streaming...",
                "options": [],
                "correctAnswerId": "",
                "explanation": "",
                "topic": topic,
                "difficulty": difficulty,
                "timestamp": ""
            }]
        else:
            # For non-streaming, generate the specified number of questions
            questions = []
            for _ in range(num_questions):
                question = self.unified_inference_manager.generate_mcq_sync(
                    topic=topic,
                    difficulty=difficulty,
                    question_type=question_type,
                    context=custom_prompt, # Use custom_prompt as context for now
                    adapter_name=deepseek_model if difficulty == "expert" else None # Pass deepseek_model as adapter_name for expert mode
                )
                if question:
                    questions.append(question)
            return questions

class UnifiedInferenceManager:
    def __init__(self):
        # Placeholder for unified inference logic
        pass

    def get_model_status(self, model_name):
        # Simulate model status
        return {"status": "online", "latency": "50ms"}

    def perform_inference(self, prompt, model_name, config):
        # Simulate inference
        return "This is a simulated response from " + model_name
# Global instance for singleton pattern
_mcq_manager_instance = None

def get_mcq_manager():
    """Get the global MCQ manager instance"""
    global _mcq_manager_instance
    if _mcq_manager_instance is None:
        _mcq_manager_instance = MCQManager()
    return _mcq_manager_instance

# Global instance for unified inference manager
_unified_inference_manager_instance = None

def get_unified_inference_manager():
    """Get the global unified inference manager instance"""
    global _unified_inference_manager_instance
    if _unified_inference_manager_instance is None:
        _unified_inference_manager_instance = UnifiedInferenceManager()
    return _unified_inference_manager_instance