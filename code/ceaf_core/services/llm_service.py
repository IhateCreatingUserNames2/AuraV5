# ceaf_core/services/llm_service.py
import os
import logging
import asyncio
from typing import Optional, Dict, Any
import litellm
from dotenv import load_dotenv

load_dotenv()

from ceaf_core.services.vast_ai_engine import VastAIEngine
from ceaf_core.models import LLMConfig
from ceaf_core.utils.embedding_utils import get_embedding_client

logger = logging.getLogger("LLMService")


class LLMService:
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.inference_mode = os.getenv("INFERENCE_MODE", "openrouter")
        self.vast_engine = None

        if self.inference_mode == "vastai":
            self.vast_engine = VastAIEngine(timeout=self.config.timeout_seconds)

        self.embedding_client = get_embedding_client()

    async def ainvoke(
            self,
            model: str,
            prompt: str,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            vector_data: Optional[Dict[str, Any]] = None
    ) -> str:
        eff_temp = temperature if temperature is not None else self.config.default_temperature
        eff_tokens = max_tokens if max_tokens is not None else self.config.max_tokens_output

        # REDIRECIONAMENTO LOCAL (Obrigatório)
        if self.inference_mode == "vastai" and self.vast_engine:
            return await self.vast_engine.generate(
                prompt=prompt,
                max_tokens=eff_tokens,
                temperature=eff_temp,
                vector_data=vector_data
            )

        # OPENROUTER FALLBACK
        try:
            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=eff_temp,
                max_tokens=eff_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return f"[Error: {e}]"

    async def ainvoke_with_logprobs(self, model: str, prompt: str, temperature: float = 0.6):
        """
        Versão com suporte a logprobs, mas interceptada para modo LOCAL.
        """
        # CORREÇÃO: Bloquear OpenRouter aqui também
        if self.inference_mode == "vastai" and self.vast_engine:
            # O Soul Engine local não suporta logprobs reais,
            # então retornamos a resposta normal dentro de um objeto compatível.
            text = await self.vast_engine.generate(prompt=prompt, temperature=temperature)

            # Criamos um "Mock" do objeto do LiteLLM para o AgencyModule não quebrar
            class MockResponse:
                def __init__(self, content):
                    self.choices = [type('obj', (object,), {
                        'message': type('obj', (object,), {'content': content}),
                        'logprobs': None  # Indica ao avaliador para usar a heurística
                    })]

            return MockResponse(text)

        try:
            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=self.config.max_tokens_output,
                logprobs=True,
                top_logprobs=1
            )
            return response
        except Exception as e:
            logger.error(f"Erro ao buscar logprobs: {e}")
            return None