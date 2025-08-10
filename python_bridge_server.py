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
        elif method_name == 'startTraining':
            return await self.start_training_frontend(args[0] if args else {})
        elif method_name == 'stop_training':
            return await self.stop_training()
        elif method_name == 'upload_document_for_training':
            return await self.upload_document_for_training(args[0], args[1])
        elif method_name == 'uploadFile':
            return await self.upload_file(args[0], args[1])
        elif method_name == 'deleteUploadedFile':
            return await self.delete_uploaded_file(args[0])
        elif method_name == 'getUploadedFiles':
            return await self.get_uploaded_files()
        elif method_name == 'getTrainingConfiguration':
            return await self.get_training_configuration()
        elif method_name == 'saveTrainingConfiguration':
            return await self.save_training_configuration(args[0] if args else {})
        
        # Settings methods
        elif method_name == 'get_app_settings':
            return await self.get_app_settings()
        elif method_name == 'save_app_settings':
            return await self.save_app_settings(args[0])
        elif method_name == 'validate_api_key':
            return await self.validate_api_key(args[0], args[1])
        elif method_name == 'getUserSettings':
            return await self.get_user_settings()
        elif method_name == 'saveUserSettings':
            return await self.save_user_settings(args[0])
        elif method_name == 'checkProviderStatus':
            return await self.check_provider_status(args[0])
        
        # Enhanced question history methods
        elif method_name == 'getQuestionHistory':
            return await self.get_question_history_paginated(args[0] if len(args) > 0 else 0, args[1] if len(args) > 1 else 50)
        elif method_name == 'searchQuestions':
            return await self.search_questions(args[0])
        elif method_name == 'filterQuestionsByTopic':
            return await self.filter_questions_by_topic(args[0])
        elif method_name == 'filterQuestionsByDifficulty':
            return await self.filter_questions_by_difficulty(args[0])
        
        # AI Model methods
        elif method_name == 'getDeepSeekStatus':
            return await self.get_deepseek_status()
        
        # Logging methods
        elif method_name == 'logClientEvent':
            return await self.log_client_event(args[0] if args else {})
        
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

    async def get_uploaded_files(self) -> str:
        """Get list of uploaded training files as JSON string"""
        try:
            files = await self.get_uploaded_files_list()
            return json.dumps(files)
        except Exception as e:
            logger.error(f"Error getting uploaded files: {e}")
            return json.dumps([])

    async def save_training_configuration(self, config: Dict[str, Any]) -> bool:
        """Save training configuration"""
        try:
            training_dir = Path("data/training")
            training_dir.mkdir(parents=True, exist_ok=True)
            
            config_path = training_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Training configuration saved: {config}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving training configuration: {e}")
            return False

    async def upload_file(self, filename: str, file_data: list) -> str:
        """Upload file from frontend (expects array of bytes)"""
        try:
            training_dir = Path("data/training")
            training_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert array of bytes to bytes
            file_bytes = bytes(file_data)
            
            # Save file
            file_path = training_dir / filename
            with open(file_path, 'wb') as f:
                f.write(file_bytes)
            
            logger.info(f"File uploaded successfully: {filename} ({len(file_bytes)} bytes)")
            return json.dumps({"success": True, "message": f"File {filename} uploaded successfully"})
            
        except Exception as e:
            logger.error(f"Error uploading file {filename}: {e}")
            return json.dumps({"success": False, "message": f"Error uploading file: {str(e)}"})

    async def delete_uploaded_file(self, filename: str) -> str:
        """Delete uploaded training file"""
        try:
            training_dir = Path("data/training")
            file_path = training_dir / filename
            
            if file_path.exists():
                file_path.unlink()
                logger.info(f"File deleted successfully: {filename}")
                return json.dumps({"success": True, "message": f"File {filename} deleted successfully"})
            else:
                return json.dumps({"success": False, "message": f"File {filename} not found"})
            
        except Exception as e:
            logger.error(f"Error deleting file {filename}: {e}")
            return json.dumps({"success": False, "message": f"Error deleting file: {str(e)}"})

    async def start_training_frontend(self, config_json: str) -> str:
        """Start training from frontend (expects JSON string)"""
        try:
            config = json.loads(config_json) if isinstance(config_json, str) else config_json
            
            logger.info(f"Starting training with config: {config}")
            
            # Save training config
            await self.save_training_configuration(config)
            
            # Emit progress events
            await self.emit_event("onTrainingProgress", {
                "progress": 0,
                "status": "Starting training...",
                "stage": "initialization"
            })
            
            # Start training process
            asyncio.create_task(self.mock_training_process())
            
            return json.dumps({"success": True, "message": "Training started successfully"})
            
        except Exception as e:
            logger.error(f"Error starting training: {e}")
            return json.dumps({"success": False, "message": f"Error starting training: {str(e)}"})

    async def get_uploaded_files(self) -> str:
        """Get list of uploaded training files as JSON string"""
        try:
            files = await self.get_uploaded_files_list()
            return json.dumps(files)
        except Exception as e:
            logger.error(f"Error getting uploaded files: {e}")
            return json.dumps([])

    async def get_uploaded_files_list(self) -> list:
        """Get list of uploaded training files"""
        try:
            training_dir = Path("data/training")
            if not training_dir.exists():
                return []
            
            files = []
            for file_path in training_dir.glob("*"):
                if file_path.is_file() and file_path.name != "config.json":
                    files.append({
                        "name": file_path.name,
                        "size": file_path.stat().st_size,
                        "modified": file_path.stat().st_mtime,
                        "path": str(file_path)
                    })
            
            return files
            
        except Exception as e:
            logger.error(f"Error getting uploaded files: {e}")
            return []

    async def get_training_configuration(self) -> str:
        """Get current training configuration as JSON string"""
        try:
            config = await self.get_training_config_dict()
            return json.dumps(config)
        except Exception as e:
            logger.error(f"Error getting training configuration: {e}")
            return json.dumps({
                "modelType": "text-generation",
                "epochs": 3,
                "batchSize": 8,
                "learningRate": 0.001
            })

    async def get_training_config_dict(self) -> Dict[str, Any]:
        """Get current training configuration"""
        try:
            # Try to load from config file
            config_path = Path("data/training/config.json")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return json.loads(f.read())
            
            # Return default configuration
            return {
                "modelType": "text-generation",
                "epochs": 3,
                "batchSize": 8,
                "learningRate": 0.001,
                "maxLength": 512,
                "warmupSteps": 100,
                "saveSteps": 500,
                "evaluationSteps": 1000,
                "outputDir": "data/models",
                "logLevel": "info"
            }
            
        except Exception as e:
            logger.error(f"Error getting training configuration: {e}")
            return {
                "modelType": "text-generation",
                "epochs": 3,
                "batchSize": 8,
                "learningRate": 0.001
            }
    
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
    
    async def log_client_event(self, event_data: Dict[str, Any]) -> bool:
        """Log client-side events for debugging and analytics"""
        try:
            # Extract event information
            event_type = event_data.get('type', 'unknown')
            event_level = event_data.get('level', 'info')
            event_message = event_data.get('message', '')
            event_details = event_data.get('details', {})
            timestamp = event_data.get('timestamp', time.time())
            
            # Log the event with appropriate level
            log_message = f"CLIENT_EVENT [{event_type}]: {event_message}"
            if event_details:
                log_message += f" | Details: {event_details}"
            
            if event_level.lower() == 'error':
                logger.error(log_message)
            elif event_level.lower() == 'warning':
                logger.warning(log_message)
            elif event_level.lower() == 'debug':
                logger.debug(log_message)
            else:
                logger.info(log_message)
            
            # Optionally store in a database or file for analytics
            # For now, just return success
            return True
            
        except Exception as e:
            logger.error(f"Error logging client event: {e}")
            return False

    async def get_user_settings(self) -> str:
        """Get user settings as JSON string"""
        settings = await self.get_app_settings()
        return json.dumps(settings)

    async def save_user_settings(self, settings_json: str) -> bool:
        """Save user settings from JSON string"""
        try:
            settings = json.loads(settings_json)
            return await self.save_app_settings(settings)
        except Exception as e:
            logger.error(f"Error saving user settings: {e}")
            return False

    async def check_provider_status(self, provider: str) -> str:
        """Check AI provider status"""
        # Mock implementation - check if provider is available
        mock_statuses = {
            'openai': {'available': True, 'status': 'operational'},
            'anthropic': {'available': True, 'status': 'operational'},
            'gemini': {'available': True, 'status': 'operational'},
            'groq': {'available': True, 'status': 'operational'},
            'openrouter': {'available': True, 'status': 'operational'},
            'deepseek': {'available': True, 'status': 'operational'},
            'tavily': {'available': True, 'status': 'operational'}
        }
        
        status = mock_statuses.get(provider, {'available': False, 'status': 'unknown'})
        return json.dumps(status)

    async def get_question_history_paginated(self, offset: int, limit: int) -> str:
        """Get paginated question history"""
        try:
            if not self.question_storage:
                # Mock paginated data
                mock_questions = [
                    {
                        "id": f"q_{i}",
                        "question": f"Sample question {i}?",
                        "options": ["Option A", "Option B", "Option C", "Option D"],
                        "correct": i % 4,
                        "topic": "General",
                        "difficulty": ["Easy", "Medium", "Hard"][i % 3],
                        "timestamp": f"2025-01-{(i % 28) + 1:02d}T10:00:00"
                    }
                    for i in range(offset, min(offset + limit, 100))
                ]
                return json.dumps(mock_questions)
            
            questions = self.question_storage.get_recent_questions(limit=limit, offset=offset)
            return json.dumps(questions)
        except Exception as e:
            logger.error(f"Error getting paginated question history: {e}")
            return json.dumps([])

    async def search_questions(self, search_term: str) -> str:
        """Search questions by term"""
        try:
            # Mock search results
            mock_results = [
                {
                    "id": "search_1",
                    "question": f"Question containing '{search_term}'?",
                    "options": ["A", "B", "C", "D"],
                    "correct": 0,
                    "topic": "Search Results",
                    "difficulty": "Medium",
                    "timestamp": "2025-01-31T10:00:00"
                }
            ]
            return json.dumps(mock_results)
        except Exception as e:
            logger.error(f"Error searching questions: {e}")
            return json.dumps([])

    async def filter_questions_by_topic(self, topic: str) -> str:
        """Filter questions by topic"""
        try:
            # Mock filtered results
            mock_results = [
                {
                    "id": f"topic_{topic}_1",
                    "question": f"Question about {topic}?",
                    "options": ["A", "B", "C", "D"],
                    "correct": 0,
                    "topic": topic,
                    "difficulty": "Medium",
                    "timestamp": "2025-01-31T10:00:00"
                }
            ]
            return json.dumps(mock_results)
        except Exception as e:
            logger.error(f"Error filtering questions by topic: {e}")
            return json.dumps([])

    async def filter_questions_by_difficulty(self, difficulty: str) -> str:
        """Filter questions by difficulty"""
        try:
            # Mock filtered results
            mock_results = [
                {
                    "id": f"diff_{difficulty}_1",
                    "question": f"{difficulty} difficulty question?",
                    "options": ["A", "B", "C", "D"],
                    "correct": 0,
                    "topic": "General",
                    "difficulty": difficulty,
                    "timestamp": "2025-01-31T10:00:00"
                }
            ]
            return json.dumps(mock_results)
        except Exception as e:
            logger.error(f"Error filtering questions by difficulty: {e}")
            return json.dumps([])

    async def get_deepseek_status(self) -> Dict[str, Any]:
        """Get DeepSeek AI model status"""
        try:
            # Mock DeepSeek status
            return {
                "available": True,
                "model": "deepseek-chat",
                "status": "ready",
                "version": "v1.0",
                "capabilities": ["text-generation", "code-completion"],
                "rate_limit": {
                    "requests_per_minute": 60,
                    "tokens_per_minute": 10000
                }
            }
        except Exception as e:
            logger.error(f"Error getting DeepSeek status: {e}")
            return {"available": False, "status": "error", "message": str(e)}
    
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
