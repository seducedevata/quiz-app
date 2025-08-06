import win32pipe
import win32file
import pywintypes
import json
import threading
import logging
import sys
import os

# Add the src directory to the sys.path to allow imports from knowledge_app
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from knowledge_app.core.mcq_manager import get_mcq_manager
from knowledge_app.core.unified_inference_manager import get_unified_inference_manager
from knowledge_app.core.question_history_storage import get_question_history_storage

# Configure logging for the IPC server
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ipc_server.log')
    ]
)

logger = logging.getLogger(__name__)

PIPE_NAME = r"\\.\pipe\KnowledgeAppPipe"
BUFFER_SIZE = 65536 # 64KB buffer

def handle_client(pipe_handle):
    logger.info("Client connected to named pipe.")
    try:
        while True:
            # Read request from client
            # The message format is expected to be JSON: {"method": "method_name", "args": [...], "kwargs": {...}}
            hr, data = win32file.ReadFile(pipe_handle, BUFFER_SIZE)
            if not data:
                break # Client disconnected

            request_str = data.decode('utf-8').strip('\x00') # Remove null terminators
            logger.debug(f"Received request: {request_str}")

            try:
                request = json.loads(request_str)
                method_name = request.get("method")
                args = request.get("args", [])
                kwargs = request.get("kwargs", {})

                response_data = {"status": "error", "message": "Unknown method"}

                if method_name == "generate_mcq_quiz":
                    mcq_manager = get_mcq_manager()
                    quiz_config = args[0] if args else {}
                    result = mcq_manager.generate_quiz(quiz_config)
                    response_data = {"status": "success", "data": result}
                elif method_name == "submit_mcq_answer":
                    mcq_manager = get_mcq_manager()
                    quiz_id, question_id, answer = args
                    is_correct = mcq_manager.submit_answer(quiz_id, question_id, answer)
                    response_data = {"status": "success", "data": is_correct}
                elif method_name == "get_mcq_quiz_result":
                    mcq_manager = get_mcq_manager()
                    quiz_id = args[0]
                    result = mcq_manager.get_quiz_result(quiz_id)
                    response_data = {"status": "success", "data": result}
                elif method_name == "load_ai_model":
                    unified_inference_manager = get_unified_inference_manager()
                    config = args[0]
                    success = unified_inference_manager.load_model(config)
                    response_data = {"status": "success", "data": success}
                elif method_name == "run_inference":
                    unified_inference_manager = get_unified_inference_manager()
                    model_name, input_data = args
                    result = unified_inference_manager.run_inference(model_name, input_data)
                    response_data = {"status": "success", "data": result}
                elif method_name == "validate_model_compatibility":
                    unified_inference_manager = get_unified_inference_manager()
                    compatibility = unified_inference_manager.validate_model_compatibility()
                    response_data = {"status": "success", "data": compatibility}
                elif method_name == "get_deepseek_status":
                    unified_inference_manager = get_unified_inference_manager()
                    status = unified_inference_manager.get_deepseek_status()
                    response_data = {"status": "success", "data": status}
                elif method_name == "configure_deepseek":
                    unified_inference_manager = get_unified_inference_manager()
                    config = args[0]
                    success = unified_inference_manager.configure_deepseek(config)
                    response_data = {"status": "success", "data": success}
                elif method_name == "run_batch_two_model_pipeline":
                    unified_inference_manager = get_unified_inference_manager()
                    data = args[0]
                    result = unified_inference_manager.run_batch_two_model_pipeline(data)
                    response_data = {"status": "success", "data": result}
                elif method_name == "start_training":
                    unified_inference_manager = get_unified_inference_manager()
                    config = args[0]
                    success = unified_inference_manager.start_training(config)
                    response_data = {"status": "success", "data": success}
                elif method_name == "stop_training":
                    unified_inference_manager = get_unified_inference_manager()
                    success = unified_inference_manager.stop_training()
                    response_data = {"status": "success", "data": success}
                elif method_name == "upload_document_for_training":
                    unified_inference_manager = get_unified_inference_manager()
                    file_name, content = args
                    success = unified_inference_manager.upload_document_for_training(file_name, content)
                    response_data = {"status": "success", "data": success}
                elif method_name == "get_question_history":
                    history_storage = get_question_history_storage()
                    filter_params = args[0] if args else {}
                    history = history_storage.get_history(filter_params)
                    response_data = {"status": "success", "data": history}
                elif method_name == "get_history_stats":
                    history_storage = get_question_history_storage()
                    stats = history_storage.get_stats()
                    response_data = {"status": "success", "data": stats}
                elif method_name == "delete_question_from_history":
                    history_storage = get_question_history_storage()
                    question_id = args[0]
                    success = history_storage.delete_question(question_id)
                    response_data = {"status": "success", "data": success}
                elif method_name == "edit_question_in_history":
                    history_storage = get_question_history_storage()
                    question_data = args[0]
                    success = history_storage.edit_question(question_data)
                    response_data = {"status": "success", "data": success}
                elif method_name == "validate_api_key":
                    # Assuming a function exists to validate API keys
                    api_key = args[0]
                    # Placeholder for actual API key validation logic
                    is_valid = True # Replace with actual validation
                    response_data = {"status": "success", "data": is_valid}
                elif method_name == "health_check":
                    response_data = {"status": "success", "data": "ok"}
                # Add more method handlers here for other functionalities
                else:
                    logger.warning(f"Method not implemented: {method_name}")

            except json.JSONDecodeError:
                response_data = {"status": "error", "message": "Invalid JSON request"}
                logger.error(f"Invalid JSON received: {request_str}")
            except Exception as e:
                response_data = {"status": "error", "message": str(e)}
                logger.exception(f"Error processing request for method {method_name}")

            response_str = json.dumps(response_data)
            win32file.WriteFile(pipe_handle, response_str.encode('utf-8'))

    except pywintypes.error as e:
        if e.args[0] == 109: # ERROR_BROKEN_PIPE
            logger.info("Client disconnected from named pipe.")
        else:
            logger.error(f"Named pipe error: {e}")
    except Exception as e:
        logger.exception("Unhandled exception in client handler.")
    finally:
        win32file.DisconnectNamedPipe(pipe_handle)
        win32file.CloseHandle(pipe_handle)

def start_ipc_server():
    logger.info(f"Starting IPC server on {PIPE_NAME}")
    while True:
        try:
            pipe_handle = win32pipe.CreateNamedPipe(
                PIPE_NAME,
                win32pipe.PIPE_ACCESS_DUPLEX, # duplex for read/write
                win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT, # message type, message read mode, blocking
                win32pipe.PIPE_UNLIMITED_INSTANCES, # max instances
                BUFFER_SIZE, # output buffer size
                BUFFER_SIZE, # input buffer size
                0, # default timeout
                None
            )
            win32pipe.ConnectNamedPipe(pipe_handle, None)
            thread = threading.Thread(target=handle_client, args=(pipe_handle,))
            thread.daemon = True # Allow main program to exit even if threads are running
            thread.start()
        except pywintypes.error as e:
            if e.args[0] == 2: # ERROR_FILE_NOT_FOUND, pipe might be in a bad state
                logger.warning(f"Named pipe {PIPE_NAME} not found, retrying...")
            else:
                logger.exception(f"Error creating named pipe: {e}")
            # Add a small delay before retrying to prevent busy-looping
            import time
            time.sleep(1)
        except Exception as e:
            logger.exception("Unhandled exception in server loop.")

if __name__ == "__main__":
    start_ipc_server()
