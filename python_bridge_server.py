"""
Python Bridge Server for React Native Windows App
Provides HTTP API for communication between React Native and Python backend
"""

import asyncio
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
import threading
import time

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from aiohttp import web, WSMsgType
from aiohttp.web import Request, Response, WebSocketResponse
import aiohttp_cors

# Import the Knowledge App components
try:
    from src.knowledge_app.core.mcq_manager import MCQManager
    from src.knowledge_app.core.unified_inference_manager import get_unified_inference_manager
    from src.knowledge_app.core.question_history_storage import QuestionHistoryStorage
    KNOWLEDGE_APP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Knowledge App components not available: {e}")
    KNOWLEDGE_APP_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PythonBridgeServer:
    """HTTP server that bridges React Native calls to Python backend"""
    
    def __init__(self, port: int = 8765):
        self.port = port
        self.app = web.Application()
        self.mcq_manager: Optional[MCQManager] = None
        self.question_storage: Optional[QuestionHistoryStorage] = None
        self.websockets = set()
        
        # Setup routes
        self.setup_routes()
        
        # Initialize components
        self.initialize_components()
    
    def setup_routes(self):
        """Setup HTTP routes for the bridge API"""
        # Health check
        self.app.router.add_get('/health', self.health_check)
        
        # Main API endpoint
        self.app.router.add_post('/api/call', self.handle_python_call)
        
        # WebSocket for real-time events
        self.app.router.add_get('/ws', self.websocket_handler)
        
        # CORS setup for development
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        # Add CORS to all routes
        for route in list(self.app.router.routes()):
            cors.add(route)
    
    def initialize_components(self):
        """Initialize Python backend components"""
        if not KNOWLEDGE_APP_AVAILABLE:
            logger.warning("Knowledge App components not available - using mock implementations")
            return
        
        try:
            # Initialize MCQ Manager
            self.mcq_manager = MCQManager()
            logger.info("MCQ Manager initialized successfully")
            
            # Initialize Question Storage
            self.question_storage = QuestionHistoryStorage()
            logger.info("Question Storage initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            logger.error(traceback.format_exc())
    
    async def health_check(self, request: Request) -> Response:
        """Health check endpoint"""
        return web.json_response({
            "status": "healthy",
            "components": {
                "mcq_manager": self.mcq_manager is not None,
                "question_storage": self.question_storage is not None,
                "knowledge_app": KNOWLEDGE_APP_AVAILABLE
            }
        })
    
    async def handle_python_call(self, request: Request) -> Response:
        """Handle Python method calls from React Native"""
        try:
            data = await request.json()
            method_name = data.get('method')
            args_json = data.get('args', '[]')
            request_id = data.get('id', 0)
            
            logger.info(f"Received call: {method_name} with args: {args_json}")
            
            # Parse arguments
            try:
                args = json.loads(args_json) if isinstance(args_json, str) else args_json
                if not isinstance(args, list):
                    args = [args]
            except json.JSONDecodeError:
                args = []
            
            # Route the method call
            result = await self.route_method_call(method_name, args)
            
            return web.json_response({
                "status": "success",
                "data": result,
                "id": request_id
            })
            
        except Exception as e:
            logger.error(f"Error handling Python call: {e}")
            logger.error(traceback.format_exc())
            return web.json_response({
                "status": "error",
                "message": str(e),
                "id": data.get('id', 0) if 'data' in locals() else 0
            })
    
    async def route_method_call(self, method_name: str, args: list) -> Any:
        """Route method calls to appropriate handlers"""
        
        # Quiz generation methods
        if method_name == 'generate_mcq_quiz':
            return await self.generate_mcq_quiz(args[0] if args else {})
        elif method_name == 'submit_mcq_answer':
            return await self.submit_mcq_answer(args[0], args[1], args[2])
        elif method_name == 'get_mcq_quiz_result':
            return await self.get_mcq_quiz_result(args[0])
        
        # Question history methods
        elif method_name == 'get_question_history':
            return await self.get_question_history(args[0] if args else None)
        elif method_name == 'get_history_stats':
            return await self.get_history_stats()
        elif method_name == 'delete_question_from_history':
            return await self.delete_question_from_history(args[0])
        elif method_name == 'edit_question_in_history':
            return await self.edit_question_in_history(args[0])
        
        # Training methods
        elif method_name == 'start_training':
            return await self.start_training(args[0] if args else {})
        elif method_name == 'stop_training':
            return await self.stop_training()
        elif method_name == 'upload_document_for_training':
            return await self.upload_document_for_training(args[0], args[1])
        
        # Settings methods
        elif method_name == 'get_app_settings':
            return await self.get_app_settings()
        elif method_name == 'save_app_settings':
            return await self.save_app_settings(args[0])
        elif method_name == 'validate_api_key':
            return await self.validate_api_key(args[0], args[1])
        
        else:
            raise ValueError(f"Unknown method: {method_name}")
    
    async def generate_mcq_quiz(self, config: Dict[str, Any]) -> str:
        """Generate MCQ quiz"""
        if not self.mcq_manager:
            # Mock implementation
            return "mock_quiz_id_123"
        
        try:
            # Convert React Native config to Python format
            quiz_params = {
                'topic': config.get('topic', 'General Knowledge'),
                'difficulty': config.get('difficulty', 'medium'),
                'submode': config.get('questionType', 'mixed'),
                'mode': config.get('mode', 'auto'),
                'num_questions': config.get('numQuestions', 1),
                'game_mode': config.get('gameMode', 'casual'),
                'timer': config.get('timer', '30s')
            }
            
            # Generate quiz asynchronously
            result = await self.mcq_manager.generate_quiz_async(quiz_params)
            
            if result:
                # Store the quiz and return ID
                quiz_id = f"quiz_{int(time.time())}"
                # In a real implementation, store the quiz result
                return quiz_id
            else:
                raise Exception("Quiz generation failed")
                
        except Exception as e:
            logger.error(f"Quiz generation error: {e}")
            raise
    
    async def submit_mcq_answer(self, quiz_id: str, question_id: str, answer: str) -> bool:
        """Submit quiz answer"""
        # Mock implementation - in real app, validate answer
        logger.info(f"Answer submitted for quiz {quiz_id}, question {question_id}: {answer}")
        return True
    
    async def get_mcq_quiz_result(self, quiz_id: str) -> Dict[str, Any]:
        """Get quiz results"""
        # Mock implementation
        return {
            "quizId": quiz_id,
            "score": 85,
            "totalQuestions": 5,
            "correctAnswers": 4,
            "timeSpent": 120,
            "questions": []
        }
    
    async def get_question_history(self, filter_params: Optional[Dict[str, Any]]) -> list:
        """Get question history"""
        if not self.question_storage:
            # Mock data
            return [
                {
                    "id": "1",
                    "question": "What is the capital of France?",
                    "options": ["London", "Berlin", "Paris", "Madrid"],
                    "correct": 2,
                    "topic": "Geography",
                    "difficulty": "Easy",
                    "timestamp": "2025-01-31T10:00:00"
                }
            ]
        
        try:
            # Get questions from storage
            questions = self.question_storage.get_recent_questions(limit=100)
            return questions
        except Exception as e:
            logger.error(f"Error getting question history: {e}")
            return []
    
    async def get_history_stats(self) -> Dict[str, Any]:
        """Get question history statistics"""
        if not self.question_storage:
            # Mock stats
            return {
                "total_questions": 10,
                "by_topic": {"Geography": 3, "Math": 4, "Science": 3},
                "by_difficulty": {"Easy": 4, "Medium": 4, "Hard": 2}
            }
        
        try:
            stats = self.question_storage.get_statistics()
            return stats
        except Exception as e:
            logger.error(f"Error getting history stats: {e}")
            return {"total_questions": 0, "by_topic": {}, "by_difficulty": {}}
    
    async def delete_question_from_history(self, question_id: str) -> bool:
        """Delete question from history"""
        if not self.question_storage:
            return True  # Mock success
        
        try:
            # Delete question
            success = self.question_storage.delete_question(question_id)
            return success
        except Exception as e:
            logger.error(f"Error deleting question: {e}")
            return False
    
    async def edit_question_in_history(self, question: Dict[str, Any]) -> bool:
        """Edit question in history"""
        if not self.question_storage:
            return True  # Mock success
        
        try:
            # Update question
            success = self.question_storage.update_question(question)
            return success
        except Exception as e:
            logger.error(f"Error editing question: {e}")
            return False
    
    async def start_training(self, config: Dict[str, Any]) -> bool:
        """Start model training"""
        logger.info(f"Starting training with config: {config}")
        
        # Emit progress events
        await self.emit_event("onTrainingProgress", {
            "progress": 0,
            "status": "Starting training...",
            "stage": "initialization"
        })
        
        # Mock training process
        asyncio.create_task(self.mock_training_process())
        
        return True
    
    async def mock_training_process(self):
        """Mock training process with progress updates"""
        stages = [
            ("initialization", "Initializing training environment..."),
            ("data_loading", "Loading training data..."),
            ("model_setup", "Setting up model architecture..."),
            ("training", "Training in progress..."),
            ("validation", "Validating model performance..."),
            ("completion", "Training completed successfully!")
        ]
        
        for i, (stage, message) in enumerate(stages):
            await asyncio.sleep(2)  # Simulate work
            progress = int((i + 1) / len(stages) * 100)
            
            await self.emit_event("onTrainingProgress", {
                "progress": progress,
                "status": message,
                "stage": stage
            })
    
    async def stop_training(self) -> bool:
        """Stop model training"""
        logger.info("Stopping training...")
        return True
    
    async def upload_document_for_training(self, filename: str, content: str) -> bool:
        """Upload document for training"""
        logger.info(f"Uploading document: {filename}")
        
        try:
            # Save document to training directory
            training_dir = Path("data/training")
            training_dir.mkdir(parents=True, exist_ok=True)
            
            # Decode base64 content if needed
            import base64
            try:
                file_content = base64.b64decode(content)
            except:
                file_content = content.encode('utf-8')
            
            # Save file
            file_path = training_dir / filename
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            logger.info(f"Document saved: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading document: {e}")
            return False
    
    async def get_app_settings(self) -> Dict[str, Any]:
        """Get application settings"""
        # Mock settings
        return {
            "theme": "light",
            "apiKeys": {},
            "preferences": {
                "defaultQuizConfig": {
                    "difficulty": "medium",
                    "questionType": "mixed",
                    "numQuestions": 5
                }
            }
        }
    
    async def save_app_settings(self, settings: Dict[str, Any]) -> bool:
        """Save application settings"""
        logger.info(f"Saving settings: {settings}")
        return True
    
    async def validate_api_key(self, provider: str, api_key: str) -> bool:
        """Validate API key"""
        logger.info(f"Validating API key for provider: {provider}")
        # Mock validation
        return len(api_key) > 10
    
    async def emit_event(self, event_name: str, data: Any):
        """Emit event to all connected WebSocket clients"""
        if not self.websockets:
            return
        
        message = json.dumps({
            "event": event_name,
            "data": data
        })
        
        # Send to all connected clients
        disconnected = set()
        for ws in self.websockets:
            try:
                await ws.send_str(message)
            except:
                disconnected.add(ws)
        
        # Remove disconnected clients
        self.websockets -= disconnected
    
    async def websocket_handler(self, request: Request) -> WebSocketResponse:
        """Handle WebSocket connections for real-time events"""
        ws = WebSocketResponse()
        await ws.prepare(request)
        
        self.websockets.add(ws)
        logger.info("WebSocket client connected")
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    # Handle incoming messages if needed
                    pass
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f'WebSocket error: {ws.exception()}')
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            self.websockets.discard(ws)
            logger.info("WebSocket client disconnected")
        
        return ws
    
    async def start_server(self):
        """Start the bridge server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, 'localhost', self.port)
        await site.start()
        
        logger.info(f"Python Bridge Server started on http://localhost:{self.port}")
        
        # Keep the server running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Server shutting down...")
        finally:
            await runner.cleanup()

def start_bridge_server(port: int = 8765):
    """Start the bridge server (entry point for C++)"""
    server = PythonBridgeServer(port)
    
    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        logger.info("Bridge server stopped")
    except Exception as e:
        logger.error(f"Bridge server error: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Python Bridge Server")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    args = parser.parse_args()
    
    start_bridge_server(args.port)