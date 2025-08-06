"""
[STREAM] Streaming Token Inference System
Real-time token streaming for quiz generation with beautiful visualizations
"""

import asyncio
import json
import re
import time
import uuid
from typing import Dict, Any, Optional, Callable, AsyncGenerator
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class TokenStreamSession:
    """Manages a single token streaming session"""
    
    def __init__(self, session_id: str, topic: str, difficulty: str, question_type: str):
        self.session_id = session_id
        self.topic = topic
        self.difficulty = difficulty
        self.question_type = question_type
        self.start_time = time.time()
        self.tokens_streamed = 0
        self.is_active = True
        self.raw_response = None  # Store the raw DeepSeek response
        self.final_question = None  # Store the parsed question object
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "topic": self.topic,
            "difficulty": self.difficulty,
            "question_type": self.question_type,
            "start_time": self.start_time,
            "tokens_streamed": self.tokens_streamed,
            "is_active": self.is_active,
            "duration": time.time() - self.start_time if not self.is_active else None
        }

class StreamingInferenceEngine:
    """
    [STREAM] Real-time token streaming engine for quiz generation
    
    Provides beautiful token-by-token visualization of AI thinking process
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.active_sessions: Dict[str, TokenStreamSession] = {}
        self.token_callback: Optional[Callable] = None
        
        # Streaming configuration
        self.stream_delay = self.config.get('stream_delay', 0.05)  # 50ms between tokens
        self.chunk_size = self.config.get('chunk_size', 3)  # Tokens per chunk
        self.enable_thinking_simulation = self.config.get('enable_thinking_simulation', True)
        
        logger.info("[STREAM] StreamingInferenceEngine initialized")

    async def _stream_deepseek_generation(self, session: TokenStreamSession):
        """Stream real DeepSeek generation with thinking tokens"""
        try:
            import requests
            import json

            # Create prompt for DeepSeek
            prompt = f"""Create a challenging {session.difficulty} level question about {session.topic}.

Generate a JSON response with this format:
{{
    "question": "Your question here",
    "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
    "correct_answer": "A) option1",
    "explanation": "Explanation here",
    "topic": "{session.topic}",
    "difficulty": "{session.difficulty}"
}}

Only return valid JSON, no other text."""

            # Stream from DeepSeek via Ollama
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "deepseek-r1:14b",
                    "prompt": prompt,
                    "stream": True,  # Enable streaming
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 2000
                    }
                },
                stream=True,
                timeout=120
            )

            if response.status_code == 200:
                full_response = ""
                thinking_mode = False

                for line in response.iter_lines():
                    if not session.is_active:
                        break

                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            chunk = data.get('response', '')

                            if chunk:
                                full_response += chunk

                                # Detect thinking mode
                                if '<think>' in chunk:
                                    thinking_mode = True
                                elif '</think>' in chunk:
                                    thinking_mode = False

                                # Stream the chunk - clean content only
                                if thinking_mode:
                                    # Thinking tokens - clean and stream
                                    clean_chunk = self._clean_token_for_display(chunk.replace('<think>', '').replace('</think>', ''))
                                    if clean_chunk.strip() and self.token_callback:
                                        self.token_callback(session.session_id, clean_chunk.strip())
                                else:
                                    # Regular output tokens - clean before streaming
                                    clean_chunk = self._clean_token_for_display(chunk)
                                    if clean_chunk.strip() and self.token_callback:
                                        self.token_callback(session.session_id, clean_chunk.strip())

                                session.tokens_streamed += 1
                                await asyncio.sleep(0.05)  # Small delay for smooth streaming

                            if data.get('done', False):
                                break

                        except json.JSONDecodeError:
                            continue

                # Store raw response for parsing
                session.raw_response = full_response
                logger.info(f"[OK] DeepSeek streaming completed: {len(full_response)} chars")

            else:
                raise Exception(f"DeepSeek request failed: {response.status_code}")

        except Exception as e:
            logger.error(f"[ERROR] DeepSeek streaming error: {e}")
            raise

    def _clean_token_for_display(self, token: str) -> str:
        """Clean token for display by removing emojis and unwanted elements"""
        import re

        # Remove common emojis used in DeepSeek responses
        emoji_patterns = [
            r'[BRAIN]', r'[THINK]', r'[THINK]', r'[DOC]', r'[TARGET]', r'[FAST]', r'[OK]', r'[ERROR]', r'[WARNING]',
            r'[RELOAD]', r'[HOT]', r'[STREAM]', r'[START]', r'[INFO]', r'[PALETTE]', r'[SEARCH]', r'[STATS]', r'[TENT]',
            r'[MASKS]', r'[PALETTE]', r'[TARGET]', r'[DICE]', r'[GAME]', r'[GUITAR]', r'[PIANO]', r'[TRUMPET]', r'[VIOLIN]'
        ]

        cleaned = token
        for pattern in emoji_patterns:
            cleaned = re.sub(pattern, '', cleaned)

        # Remove XML-like tags that might appear
        cleaned = re.sub(r'<[^>]*>', '', cleaned)

        # Clean up extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        return cleaned

    def set_token_callback(self, callback: Callable[[str, str], None]):
        """Set callback function for token streaming"""
        self.token_callback = callback
        logger.info("[OK] Token callback registered")
    
    async def stream_question_generation(
        self, 
        topic: str, 
        difficulty: str = "medium", 
        question_type: str = "mixed"
    ) -> str:
        """
        [STREAM] Stream question generation with real-time token visualization
        
        Returns:
            session_id: Unique identifier for this streaming session
        """
        session_id = str(uuid.uuid4())
        session = TokenStreamSession(session_id, topic, difficulty, question_type)
        self.active_sessions[session_id] = session
        
        logger.info(f"[STREAM] Starting token stream for {topic} ({difficulty}) - Session: {session_id}")
        
        try:
            # [START] CRITICAL FIX: Start streaming and await completion
            # This ensures the streaming pipeline runs to completion
            logger.info(f"[OK] Starting streaming task for session: {session_id}")

            # Start the streaming process - this will run the full pipeline
            await self._stream_tokens_async(session)

            return session_id

        except Exception as e:
            logger.error(f"[ERROR] Failed to start token stream: {e}")
            session.is_active = False
            if self.token_callback:
                self.token_callback(session_id, f"ERROR: {str(e)}")
            raise
    
    async def _stream_tokens_async(self, session: TokenStreamSession):
        """Internal method to handle token streaming"""
        try:
            logger.info(f"[STREAM] Starting streaming pipeline for session: {session.session_id}")

            # Phase 1: Thinking simulation
            if self.enable_thinking_simulation:
                logger.info(f"[BRAIN] Phase 1: Starting thinking simulation for session: {session.session_id}")
                await self._stream_thinking_phase(session)
                logger.info(f"[OK] Phase 1: Thinking simulation completed for session: {session.session_id}")

            # Phase 2: Question generation
            logger.info(f"[DOC] Phase 2: Starting question generation for session: {session.session_id}")
            await self._stream_question_generation_phase(session)
            logger.info(f"[OK] Phase 2: Question generation completed for session: {session.session_id}")

            # Phase 3: Completion
            logger.info(f"[FINISH] Phase 3: Starting completion for session: {session.session_id}")
            await self._complete_streaming_session(session)
            logger.info(f"[OK] Phase 3: Streaming completed for session: {session.session_id}")

        except Exception as e:
            logger.error(f"[ERROR] Token streaming error for session {session.session_id}: {e}")
            session.is_active = False
            if self.token_callback:
                self.token_callback(session.session_id, f"ERROR: {str(e)}")
    
    async def _stream_thinking_phase(self, session: TokenStreamSession):
        """Stream the AI thinking process"""

        thinking_tokens = [
            "Analyzing", "topic:", f'"{session.topic}"',
            "Considering", "difficulty:", f'"{session.difficulty}"',
            "Planning", "question", "structure",
            "Focusing", "on", f'"{session.question_type}"', "format",
            "Generating", "content"
        ]

        logger.info(f"Streaming {len(thinking_tokens)} thinking tokens")

        for i, token in enumerate(thinking_tokens):
            if not session.is_active:
                logger.warning(f"Session {session.session_id} became inactive at token {i}")
                break

            if self.token_callback:
                logger.info(f"[STREAM] Emitting thinking token {i+1}/{len(thinking_tokens)}: {token}")
                self.token_callback(session.session_id, token)
            else:
                logger.warning("[WARNING] No token callback registered!")

            session.tokens_streamed += 1
            await asyncio.sleep(self.stream_delay)

        logger.info(f"[OK] Thinking phase completed: {session.tokens_streamed} tokens streamed")
    
    async def _stream_question_generation_phase(self, session: TokenStreamSession):
        """Stream the actual question generation using DeepSeek"""
        try:
            # [HOT] REAL DEEPSEEK STREAMING: Connect to actual DeepSeek generation
            await self._stream_deepseek_generation(session)
        except Exception as e:
            logger.error(f"[ERROR] DeepSeek streaming failed: {e}")
            logger.info("[RELOAD] Falling back to simulation streaming...")
            # Fallback to simulation if DeepSeek fails
            question_parts = await self._generate_question_parts(session)
            logger.info(f"[STREAM] Generated {len(question_parts)} question parts for streaming")

            for part in question_parts:
                if not session.is_active:
                    break

                # Stream tokens in chunks for smooth visualization
                tokens = part.split()
                for i in range(0, len(tokens), self.chunk_size):
                    chunk = " ".join(tokens[i:i + self.chunk_size])

                    if self.token_callback:
                        logger.info(f"[STREAM] Emitting question token: {chunk}")
                        self.token_callback(session.session_id, chunk)
                    else:
                        logger.warning("[WARNING] No token callback registered for question!")

                    session.tokens_streamed += len(tokens[i:i + self.chunk_size])
                    await asyncio.sleep(self.stream_delay)
    
    async def _generate_question_parts(self, session: TokenStreamSession) -> list:
        """Generate realistic question parts for streaming"""
        # This would normally call the actual AI model
        # For now, simulate with topic-specific content
        
        if "physics" in session.topic.lower():
            return [
                "What is the relationship between",
                "force and acceleration according to",
                "Newton's second law of motion?",
                "A) F = ma",
                "B) F = mv",
                "C) F = m/a", 
                "D) F = a/m",
                "The correct answer is A) F = ma.",
                "This fundamental equation shows that force",
                "equals mass times acceleration."
            ]
        elif "chemistry" in session.topic.lower() or "atoms" in session.topic.lower():
            return [
                "In molecular orbital theory,",
                "how do electrons behave differently",
                "compared to VSEPR theory?",
                "A) Electrons are localized to specific atoms",
                "B) Electrons are delocalized across the molecule", 
                "C) Electrons follow classical physics",
                "D) Electrons have no wave properties",
                "The correct answer is B.",
                "MO theory treats electrons as delocalized",
                "across the entire molecular framework."
            ]
        else:
            return [
                f"What is a key concept in {session.topic}",
                "that demonstrates advanced understanding?",
                "A) Basic definition",
                "B) Complex interaction",
                "C) Advanced application",
                "D) Theoretical framework",
                "The correct answer depends on context.",
                "Advanced topics require deep analysis",
                "of underlying principles."
            ]

    def _clean_deepseek_response(self, response_text: str) -> str:
        """Clean DeepSeek response by removing <think> tags and extracting JSON"""
        if not response_text:
            return ""

        logger.info(f"[CLEAN] Cleaning DeepSeek response: {len(response_text)} characters")

        # Remove <think> tags and their content
        cleaned_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        logger.info(f"[CLEAN] After removing <think> tags: {len(cleaned_text)} characters")

        # Find JSON block (between ```json and ```)
        json_match = re.search(r'```json\s*(.*?)\s*```', cleaned_text, re.DOTALL)
        if json_match:
            json_content = json_match.group(1).strip()
            logger.info(f"[OK] Found JSON block: {len(json_content)} characters")
            return json_content

        # Fallback: Look for JSON-like content (starts with { and ends with })
        json_fallback = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
        if json_fallback:
            json_content = json_fallback.group(0).strip()
            logger.info(f"[OK] Found JSON fallback: {len(json_content)} characters")
            return json_content

        # Last resort: Return cleaned text and hope it's valid JSON
        logger.warning(f"[WARNING] No JSON block found, returning cleaned text")
        return cleaned_text.strip()

    async def _complete_streaming_session(self, session: TokenStreamSession):
        """Complete the streaming session"""
        session.is_active = False

        # [CONFIG] CRITICAL FIX: Parse actual DeepSeek response instead of using placeholder
        final_question = None

        # Try to parse the actual DeepSeek response (stored in session.raw_response)
        if hasattr(session, 'raw_response') and session.raw_response:
            try:
                # The DeepSeek response should be JSON, but may contain <think> tags
                if isinstance(session.raw_response, str):
                    logger.info(f"[SEARCH] Processing raw DeepSeek response: {len(session.raw_response)} characters")
                    logger.info(f"Raw response preview: {session.raw_response[:200]}...")

                    # [CONFIG] CRITICAL FIX: Clean the response before parsing
                    cleaned_response = self._clean_deepseek_response(session.raw_response)
                    logger.info(f"[CLEAN] Cleaned response length: {len(cleaned_response)} characters")

                    # Parse JSON response from DeepSeek
                    parsed_response = json.loads(cleaned_response)

                    # Validate that it has the required fields
                    if (isinstance(parsed_response, dict) and
                        'question' in parsed_response and
                        'options' in parsed_response):

                        final_question = {
                            "question": parsed_response.get('question', ''),
                            "options": parsed_response.get('options', []),
                            "correct_answer": parsed_response.get('correct_answer', 'A'),
                            "explanation": parsed_response.get('explanation', ''),
                            "metadata": {
                                "session_id": session.session_id,
                                "tokens_streamed": session.tokens_streamed,
                                "generation_time": time.time() - session.start_time,
                                "difficulty": session.difficulty,
                                "question_type": session.question_type
                            }
                        }
                        logger.info(f"[OK] Successfully parsed DeepSeek response: {final_question['question'][:100]}...")
                    else:
                        logger.warning(f"[WARNING] DeepSeek response missing required fields: {list(parsed_response.keys()) if isinstance(parsed_response, dict) else 'not a dict'}")

            except json.JSONDecodeError as e:
                logger.warning(f"[WARNING] Failed to parse DeepSeek response as JSON: {e}")
                logger.warning(f"Raw response: {session.raw_response[:200]}...")
            except Exception as e:
                logger.warning(f"[WARNING] Error processing DeepSeek response: {e}")

        # CRITICAL: No fallback questions - fail if parsing failed
        if not final_question:
            logger.error(f"[CRITICAL] Question generation failed for {session.topic} - No fallback questions allowed")
            # Mark session as failed instead of using placeholder
            session.final_question = None
            session.is_active = False
            
            # Notify completion with failure
            if self.token_callback:
                self.token_callback(session.session_id, "STREAM_FAILED")
            
            logger.error(f"[FAILED] Token stream failed - Session: {session.session_id}, Topic: {session.topic}")
            return

        # Store the final question (either parsed or placeholder)
        session.final_question = final_question
        
        # Notify completion
        if self.token_callback:
            self.token_callback(session.session_id, "STREAM_COMPLETE")
        
        logger.info(f"[OK] Token stream completed - Session: {session.session_id}, Tokens: {session.tokens_streamed}")
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a streaming session"""
        session = self.active_sessions.get(session_id)
        return session.to_dict() if session else None
    
    def stop_session(self, session_id: str) -> bool:
        """Stop an active streaming session"""
        session = self.active_sessions.get(session_id)
        if session:
            session.is_active = False
            logger.info(f"[STOP] Stopped token stream - Session: {session_id}")
            return True
        return False
    
    def cleanup_completed_sessions(self):
        """Clean up completed streaming sessions"""
        completed = [sid for sid, session in self.active_sessions.items() if not session.is_active]
        for session_id in completed:
            del self.active_sessions[session_id]
        
        if completed:
            logger.info(f"[CLEAN] Cleaned up {len(completed)} completed streaming sessions")

# Global streaming engine instance
_streaming_engine: Optional[StreamingInferenceEngine] = None

def get_streaming_engine(config: Optional[Dict] = None) -> StreamingInferenceEngine:
    """Get or create the global streaming inference engine"""
    global _streaming_engine
    if _streaming_engine is None:
        _streaming_engine = StreamingInferenceEngine(config)
    return _streaming_engine
