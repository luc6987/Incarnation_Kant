import google.generativeai as genai
from google.api_core import exceptions
from loguru import logger
from typing import Optional, List, Generator, Union
from src.config import settings
import time
import random

class GeminiClient:
    """Wrapper for Google Gemini API."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.llm.api_key
        
        # Usage tracking
        self.usage_stats = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "requests": 0
        }
        
        if not self.api_key:
            logger.warning("⚠️ No Google API Key provided. LLM calls will fail.")
            return
            
        genai.configure(api_key=self.api_key)
        
        # Priority: explicit arg > config > default
        self.model_name = settings.llm.model_name
        
        self.model = genai.GenerativeModel(self.model_name)
        logger.info(f"✨ Initialized Gemini Client with model: {self.model_name}")

    def generate_content(
        self, 
        prompt: str, 
        stream: bool = False,
        temperature: Optional[float] = None
    ) -> Union[str, Generator[str, None, None]]:
        """Generate content from the model."""
        if not self.api_key:
            raise ValueError("Google API Key is not set.")

        config = genai.types.GenerationConfig(
            temperature=temperature or settings.llm.temperature,
            max_output_tokens=settings.llm.max_output_tokens,
        )

        # Disable safety settings to prevent truncation of philosophical texts
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]

        try:
            self.usage_stats["requests"] += 1
            
            # Retry logic for Rate Limiting
            max_retries = 3
            current_attempt = 0
            
            while True:
                try:
                    response = self.model.generate_content(
                        prompt,
                        generation_config=config,
                        safety_settings=safety_settings,
                        stream=stream
                    )
                    break # Success
                except exceptions.ResourceExhausted:
                    current_attempt += 1
                    if current_attempt > max_retries:
                        logger.error("❌ Max retries reached for Rate Limit.")
                        raise
                    
                    # Exponential backoff: 2s, 4s, 8s + jitter
                    sleep_time = (2 ** current_attempt) + random.uniform(0, 1)
                    logger.warning(f"⏳ Rate Limit hit. Retrying in {sleep_time:.2f}s... (Attempt {current_attempt}/{max_retries})")
                    time.sleep(sleep_time)
                except Exception:
                    raise # Re-raise other exceptions immediately
            
            # Update usage stats if available (mostly for non-stream, or accessible on stream object)
            try:
                if hasattr(response, "usage_metadata"):
                    self.usage_stats["input_tokens"] += response.usage_metadata.prompt_token_count
                    self.usage_stats["output_tokens"] += response.usage_metadata.candidates_token_count
                    self.usage_stats["total_tokens"] += response.usage_metadata.total_token_count
            except Exception:
                pass # Usage metadata might not be available immediately in stream mode

            if stream:
                return self._stream_response(response)
            else:
                # For non-stream, check if parts exist
                if response.parts:
                    return response.text
                else:
                    logger.warning(f"⚠️ Response finished with reason: {response.prompt_feedback}")
                    return f"[Blocked: {response.prompt_feedback}]"
                
        except Exception as e:
            logger.error(f"❌ Gemini Generation Error: {e}")
            raise

    def _stream_response(self, response) -> Generator[str, None, None]:
        """Yield text chunks from streamed response."""
        try:
            for chunk in response:
                # Check for safety blocking on the chunk level
                if chunk.candidates and chunk.candidates[0].finish_reason:
                     finish_reason = chunk.candidates[0].finish_reason
                     if finish_reason != 0 and finish_reason != 1: # 0=UNKNOWN, 1=STOP
                         logger.warning(f"⚠️ Stream chunk finished with reason: {finish_reason}")
                
                # Try to access text
                try:
                    if chunk.text:
                        yield chunk.text
                except ValueError:
                    # Usually happens if the chunk was blocked
                    candidate = chunk.candidates[0] if chunk.candidates else None
                    reason_code = candidate.finish_reason if candidate else "Unknown"
                    
                    # Log detailed safety ratings if available
                    safety_ratings = candidate.safety_ratings if candidate else "N/A"
                    citation = candidate.citation_metadata if candidate and hasattr(candidate, 'citation_metadata') else None
                    
                    logger.warning(
                        f"⚠️ Blocked chunk detail:\n"
                        f"   - Finish Reason: {reason_code}\n"
                        f"   - Safety Ratings: {safety_ratings}\n"
                        f"   - Citation/Recitation: {citation}"
                    )
                    pass
                    
        except Exception as e:
            logger.error(f"❌ Error during streaming: {e}")
            yield f"[Error: {str(e)}]"
