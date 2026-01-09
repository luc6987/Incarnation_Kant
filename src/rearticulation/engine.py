from typing import List, Dict, Any, Generator, Union
from src.utils.llm_client import GeminiClient
from loguru import logger
from src.config import PROJECT_ROOT, settings
from string import Template

class Rearticulator:
    """Engine for re-articulating Kant's thoughts in modern language."""
    
    def __init__(self):
        self.client = GeminiClient()
        
    def rearticulate(
        self, 
        user_query: str, 
        context_docs: List[Dict[str, Any]], 
        stream: bool = True
    ) -> Union[str, Generator[str, None, None]]:
        """Generate the final response based on retrieved context."""
        
        context_text = "\n\n".join([
            f"--- Fragment {i+1} ---\n{doc['content']}\nSource: {doc['metadata']}" 
            for i, doc in enumerate(context_docs)
        ])
        
        logger.info("🧠 Generating re-articulation...")
        
        try:
            prompt_path = PROJECT_ROOT / "prompts" / "rearticulation.txt"
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_template = Template(f.read())

            prompt = prompt_template.safe_substitute(context_text=context_text, user_query=user_query)
            
            # Use specific temperature from config
            return self.client.generate_content(
                prompt, 
                stream=stream,
                temperature=settings.rearticulation.temperature
            )
        except Exception as e:
            logger.error(f"❌ Re-articulation failed: {e}")
            raise

    def detect_language(self, text: str) -> str:
        """Detect the language of the text using LLM for better accuracy than langdetect."""
        try:
            # Short prompt to identify language
            prompt = (
                "Identify the language of the following text. "
                "Return ONLY the language name in English (e.g. 'English', 'German', 'Chinese'). "
                "Do not return language codes, only the full name.\n\n"
                f"Text:\n{text[:200]}"
            )
            lang = self.client.generate_content(prompt, stream=False)
            # Cleanup in case LLM is chatty
            return lang.strip().split('\n')[0].replace(".", "")
        except Exception as e:
            logger.warning(f"LLM Language detection failed: {e}")
            return "English"  # Fallback

    def translate_text(self, text: str, target_lang: str) -> str:
        """Translate text to target language using the same LLM engine."""
        if not target_lang or target_lang.lower() == 'unknown':
            return text
            
        # If the target language is German, we might just return the text, 
        # but the user might be asking in modern German vs 18th century German.
        # So we will proceed with 'translation' (modernization) if requested.
        
        prompt = (
            f"Translate the following philosophical text into {target_lang}. "
            "Return ONLY the translated text, preserving the philosophical meaning. "
            "Do not add any introductory or concluding remarks.\n\n"
            f"Text:\n{text}"
        )
        
        try:
            return self.client.generate_content(prompt, stream=False)
        except Exception as e:
            logger.error(f"❌ Translation failed: {e}")
            return text
