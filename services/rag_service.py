"""
RAG service integrating embeddings with OpenAI/Gemini generation.
Handles retrieval-augmented generation for question creation and descriptions.
"""

from google import genai
from google.genai import types
from typing import List, Dict, Any, Optional
import logging
import json
import asyncio
from config.settings import settings
from services.embedding_service import embedding_service
from utils.language_detector import language_detector
from utils.metadata_extractor import compact_metadata

logger = logging.getLogger(__name__)


class DirectOpenAILLM:
    """Small OpenAI SDK wrapper matching the generate interface used by RAGService."""

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=api_key, max_retries=0)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def generate(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(4):
            try:
                if attempt > 0:
                    wait = min(2 ** attempt, 8)
                    logger.info("OpenAI rate limited, retrying in %ss (attempt %s)", wait, attempt + 1)
                    await asyncio.sleep(wait)

                request_args = {
                    "model": self.model,
                    "messages": messages,
                }
                if self.model.startswith("gpt-5"):
                    # Current GPT-5-family chat models use max_completion_tokens
                    # and may reject non-default temperature values.
                    request_args["max_completion_tokens"] = self.max_tokens
                else:
                    request_args["temperature"] = self.temperature
                    request_args["max_tokens"] = self.max_tokens
                response = await self.client.chat.completions.create(**request_args)
                return response.choices[0].message.content or ""
            except Exception as e:
                error_text = str(e).lower()
                if "insufficient_quota" in error_text:
                    raise
                if "429" in error_text and attempt < 3:
                    continue
                raise

    async def generate_structured(self, prompt: str, schema: dict, system: Optional[str] = None) -> dict:
        response = await self.generate(
            prompt=f"{prompt}\n\nReturn only JSON matching this schema:\n{json.dumps(schema, ensure_ascii=False)}",
            system=system,
        )
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        try:
            return json.loads(cleaned.strip())
        except json.JSONDecodeError:
            logger.warning("Failed to parse structured OpenAI response: %s", response[:200])
            return {}


class RAGService:
    """
    Retrieval-Augmented Generation service.
    Combines semantic search with Google Gemini for context-aware generation.
    """
    
    def __init__(self):
        # Initialize attributes to avoid potential AttributeError
        self.llm = None
        self.openai_llm = None
        self.gpt4_nano_llm = None
        
        # Determine provider
        self.provider = settings.llm_provider.lower()
        # Always define generation_config because structured generation uses it even when OpenAI is selected.
        self.generation_config = {
            "temperature": settings.gemini_temperature,
            "max_output_tokens": settings.gemini_max_tokens if settings.gemini_max_tokens > 4096 else 8192,
        }
        self.openai_llm = None
        self.gpt4_nano_llm = None
        if getattr(settings, "openai_api_key", None):
            try:
                self.openai_llm = DirectOpenAILLM(
                    api_key=settings.openai_api_key,
                    model=settings.openai_model,
                    temperature=settings.openai_temperature,
                    max_tokens=settings.openai_max_tokens,
                )
                
                # Special instance for gpt-4.1-nano (custom model requested by user)
                self.gpt4_nano_llm = DirectOpenAILLM(
                    api_key=settings.openai_api_key,
                    model="gpt-4.1-nano",
                    temperature=0.7, 
                    max_tokens=settings.openai_max_tokens,
                )
                logger.info(f"OpenAI generation initialized ({settings.openai_model} & gpt-4.1-nano)")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI generation: {e}")

        # Always initialize Gemini client as a fallback — even when OpenAI is the primary provider.
        # This prevents AttributeError in quota-exceeded fallback paths.
        self.gemini_client = None
        self.gemini_model_name = None
        self.gemini_config = None
        if getattr(settings, "google_api_key", None):
            try:
                self.gemini_client = genai.Client(api_key=settings.google_api_key)
                self.gemini_model_name = settings.gemini_model
                self.gemini_config = types.GenerateContentConfig(
                    temperature=self.generation_config.get("temperature"),
                    max_output_tokens=self.generation_config.get("max_output_tokens")
                )
                logger.info(f"Gemini client initialized ({settings.gemini_model})")
            except Exception as e:
                logger.warning(f"Gemini client initialization failed: {e}")

        if self.provider == "openai" and self.openai_llm:
            self.llm = self.openai_llm
            logger.info(f"RAG Service default initialized with OpenAI ({settings.openai_model})")
        else:
            if self.provider == "openai" and not self.openai_llm:
                logger.warning("LLM provider is set to 'openai' but OpenAI is unavailable; falling back to Gemini")
                self.provider = "gemini"
            if not self.gemini_client:
                raise RuntimeError("Neither OpenAI nor Gemini could be initialized. Check API keys.")
            logger.info(f"RAG Service default initialized with Gemini ({settings.gemini_model})")
    
    async def retrieve_context(
        self,
        query: str,
        n_results: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from vector database.
        
        Args:
            query: Search query
            n_results: Number of results
            metadata_filter: Optional metadata filters
            
        Returns:
            List of relevant contexts
        """
        try:
            results = await embedding_service.search(
                query=query,
                n_results=n_results,
                filter_metadata=metadata_filter
            )
            
            logger.info(f"Retrieved {len(results)} context chunks")
            return results
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return []

    async def retrieve_with_metadata(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Metadata-aware hybrid retrieval.
        Combines semantic search results with metadata match scoring.
        Formula: final_score = (semantic_score * 0.7) + (metadata_score * 0.3)
        """
        from utils.query_metadata_extractor import extract_query_metadata
        
        # 1. Extract and merge filters, then stringify all values for consistent ChromaDB querying
        extracted = extract_query_metadata(query)
        filters = {**extracted, **(metadata_filter or {})}
        filters = compact_metadata(filters)

        # 2. Semantic Search (ChromaDB)
        # We fetch more than top_k to allow re-ranking
        hard_filter = {}
        if filters.get("file_id"):
            hard_filter["file_id"] = filters["file_id"]

        semantic_results = await embedding_service.search(
            query=query,
            n_results=top_k * 3,
            filter_metadata=hard_filter or None
        )

        if not semantic_results:
            return []

        # 3. Apply Hybrid Scoring
        scored: List[Dict[str, Any]] = []
        for res in semantic_results:
            semantic_score = res.get('score', 0.0)
            meta = res.get('metadata') or {}
            
            # Calculate metadata match score
            matched = 0
            total = 0
            for key, expected in filters.items():
                total += 1
                # Handle common aliases
                candidates = [meta.get(key)]
                if key == 'grade_level': candidates.append(meta.get('grade'))
                if key == 'semester': candidates.append(meta.get('term'))
                
                if expected in candidates:
                    matched += 1
            
            metadata_score = (matched / total) if total > 0 else 0.0
            
            # Hybrid Formula: 0.7 Semantic + 0.3 Metadata
            final_score = (semantic_score * 0.7) + (metadata_score * 0.3)
            
            if final_score >= min_score:
                scored.append({
                    'text': res.get('text', ''),
                    'metadata': meta,
                    'score': final_score,
                    'semantic_score': semantic_score,
                    'metadata_score': metadata_score,
                })

        # 4. Re-sort and limit
        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored[:top_k]

    async def generate_directly(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        force_provider: Optional[str] = None
    ) -> str:
        """
        Generate response using selected provider directly without context.
        
        Args:
            prompt: User prompt
            system_instruction: Optional system instruction
            force_provider: Optional override to force 'openai', 'gpt-4o-mini', or 'gemini'
            
        Returns:
            Generated response
        """
        try:
            # Build full prompt with system instruction (mostly for Gemini)
            if system_instruction:
                full_prompt = f"System: {system_instruction}\n\nUser: {prompt}"
            else:
                full_prompt = prompt
            
            # Generate response
            provider_to_use = (force_provider or self.provider).lower()
            if provider_to_use == "gpt-4.1-nano" and self.gpt4_nano_llm:
                try:
                    return await self.gpt4_nano_llm.generate(prompt=prompt, system=system_instruction)
                except Exception as e:
                    if self._is_quota_error(e) and settings.allow_gemini_fallback:
                        logger.warning("OpenAI quota exceeded, falling back to Gemini")
                        return await self._fallback_to_gemini(full_prompt)
                    raise
            elif provider_to_use == "openai" and self.openai_llm:
                try:
                    return await self.openai_llm.generate(prompt=prompt, system=system_instruction)
                except Exception as e:
                    if self._is_quota_error(e) and settings.allow_gemini_fallback:
                        logger.warning("OpenAI quota exceeded, falling back to Gemini")
                        return await self._fallback_to_gemini(full_prompt)
                    raise
            else:
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model_name,
                    contents=full_prompt,
                    config=self.gemini_config
                )
                return response.text
            
        except Exception as e:
            logger.error(f"Direct generation failed: {e}")
            raise
    
    async def generate_with_context(
        self,
        prompt: str,
        context: List[Dict[str, Any]],
        system_instruction: Optional[str] = None
    ) -> str:
        """
        Generate response using retrieved context and Gemini.
        
        Args:
            prompt: User prompt
            context: Retrieved context chunks
            system_instruction: Optional system instruction
            
        Returns:
            Generated response
        """
        try:
            # Build context string
            context_str = self._build_context_string(context)
            
            # Build full prompt
            full_prompt = self._build_prompt(prompt, context_str, system_instruction)
            
            # Generate response
            if self.provider == "openai":
                try:
                    return await self.llm.generate(prompt=full_prompt)
                except Exception as e:
                    logger.error(f"OpenAI generation failed: {e}")
                    if self._is_quota_error(e) and settings.allow_gemini_fallback:
                        logger.warning("OpenAI quota exceeded, falling back to Gemini")
                        response = self.gemini_client.models.generate_content(
                            model=self.gemini_model_name,
                            contents=full_prompt,
                            config=self.gemini_config
                        )
                        return response.text
                    raise
            else:
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model_name,
                    contents=full_prompt,
                    config=self.gemini_config
                )
                return response.text
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def _build_context_string(self, context: List[Dict[str, Any]]) -> str:
        """Build formatted context string from retrieved chunks."""
        if not context:
            return ""
        
        context_parts = []
        for idx, item in enumerate(context, 1):
            text = item.get('text', '')
            score = item.get('score', 0)
            context_parts.append(f"[{idx}] (Relevance: {score:.2f})\n{text}")
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(
        self,
        user_prompt: str,
        context: str,
        system_instruction: Optional[str] = None
    ) -> str:
        """Build complete prompt with context and instructions."""
        parts = []
        
        if system_instruction:
            parts.append(f"=== INSTRUCTIONS ===\n{system_instruction}\n")
        
        if context:
            parts.append(f"=== CONTEXT ===\n{context}\n")
        
        parts.append(f"=== USER REQUEST ===\n{user_prompt}")
        
        return "\n".join(parts)

    def _is_quota_error(self, error: Exception) -> bool:
        """Check if error is due to OpenAI quota exhaustion."""
        error_str = str(error).lower()
        return "insufficient_quota" in error_str or "quota" in error_str and "429" in error_str

    def _is_transient_generation_error(self, error: Exception) -> bool:
        """Check if a generation error is likely temporary and worth retrying."""
        error_str = str(error).lower()
        transient_markers = (
            "503",
            "unavailable",
            "high demand",
            "temporarily",
            "try again later",
            "deadline exceeded",
            "timeout",
            "rate limit",
            "resource_exhausted",
        )
        return any(marker in error_str for marker in transient_markers)
    
    async def _fallback_to_gemini(self, prompt: str) -> str:
        """Fallback to Gemini when OpenAI fails."""
        try:
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model_name,
                contents=prompt,
                config=self.gemini_config
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini fallback also failed: {e}")
            raise
    
    async def rag_query(
        self,
        query: str,
        n_context: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        system_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve + generate.
        
        Args:
            query: User query
            n_context: Number of context chunks to retrieve
            metadata_filter: Optional metadata filters
            system_instruction: Optional system instruction
            
        Returns:
            Dictionary with response and metadata
        """
        # Retrieve context with metadata-aware hybrid retrieval
        context = await self.retrieve_with_metadata(
            query=query,
            top_k=n_context,
            metadata_filter=metadata_filter
        )
        
        # Generate response
        response = await self.generate_with_context(
            prompt=query,
            context=context,
            system_instruction=system_instruction
        )
        
        return {
            "response": response,
            "context_used": len(context),
            "sources": [c.get('metadata', {}) for c in context]
        }
    
    async def generate_structured_output(
        self,
        prompt: str,
        context: List[Dict[str, Any]],
        output_schema: Dict[str, Any],
        system_instruction: Optional[str] = None
    ) -> Any:
        """
        Generate structured JSON output using RAG.
        
        Args:
            prompt: User prompt
            context: Retrieved context
            output_schema: Expected output structure
            system_instruction: Optional system instruction
            
        Returns:
            Parsed structured output
        """
        # Add JSON formatting instruction
        json_instruction = (
            f"\n\nYou must respond with ONLY valid JSON matching this schema:\n"
            f"{json.dumps(output_schema, indent=2)}\n"
            f"Do not include any text before or after the JSON."
        )
        
        full_instruction = system_instruction or ""
        full_instruction += json_instruction
        
        # Use higher token limit for question generation
        question_config = self.generation_config.copy()
        question_config["max_output_tokens"] = 8192  # Ensure enough tokens for multiple questions
        
        # Generate
        max_retries = 4
        last_error = None
        
        current_prompt = prompt
        
        for attempt in range(max_retries):
            try:
                response_text = await self.generate_with_context(
                    prompt=current_prompt,
                    context=context,
                    system_instruction=full_instruction
                )
                
                # Parse JSON
                # Remove markdown code blocks if present
                cleaned = response_text.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.startswith("```"):
                    cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                
                cleaned = cleaned.strip()
                parsed = json.loads(cleaned)
                return parsed
                
            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(f"JSON parse attempt {attempt + 1} failed: {e}")
                # For retry, explicitly ask for shorter output or more strictly valid JSON
                current_prompt = f"{prompt}\n\nIMPORTANT: Your previous response was not valid JSON. Please ensure the response is strictly valid JSON and if it was too long, make it shorter/more concise to avoid truncation."
            except Exception as e:
                last_error = e
                logger.error(f"Generation attempt {attempt + 1} failed: {e}")
                if self._is_quota_error(e):
                    raise
                if self._is_transient_generation_error(e) and attempt < max_retries - 1:
                    wait = min(2 ** (attempt + 1), 12)
                    logger.info(
                        "Transient generation error, retrying in %ss (attempt %s/%s)",
                        wait,
                        attempt + 2,
                        max_retries,
                    )
                    await asyncio.sleep(wait)
                    continue
                
        raise ValueError(f"Generated response was not valid JSON after {max_retries} attempts. Last error: {last_error}")


# Global instance
rag_service = RAGService()
