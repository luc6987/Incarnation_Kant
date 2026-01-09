from src.utils.llm_client import GeminiClient
from loguru import logger
from src.config import PROJECT_ROOT, settings
from string import Template

class HydeGenerator:
    """Hypothetical Document Embeddings (HyDE) generator."""
    
    def __init__(self):
        self.client = GeminiClient()
        
    def generate_hypothesis(self, user_query: str) -> str:
        """Convert user query into an 18th-century German Kantian monologue."""
        logger.info(f"🤔 Generating HyDE for: {user_query[:50]}...")
        
        try:
            prompt_path = PROJECT_ROOT / "prompts" / "hyde.txt"
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_template = Template(f.read())
                
            prompt = prompt_template.safe_substitute(user_query=user_query)
            
            # Use specific temperature from config
            hypothesis = self.client.generate_content(
                prompt, 
                temperature=settings.hyde.temperature
            )
            logger.info(f"🇩🇪 Generated HyDE: {hypothesis[:100]}...")
            return hypothesis.strip()
        except Exception as e:
            logger.error(f"❌ HyDE generation failed: {e}")
            raise
