# ceaf_core/services/vast_ai_engine.py
import asyncio
import os
import aiohttp
import logging
import ssl
import json
from typing import Optional, Dict, Any, List

logger = logging.getLogger("VastAIEngine")


class VastAIEngine:
    def __init__(self, timeout=240): # Aumentado para 60s
        # FORÇA 127.0.0.1 para evitar erro de resolução de nome do Windows
        self.base_url = os.getenv("VASTAI_ENDPOINT", "http://199.68.217.31:54213").rstrip("/")
        self.generate_url = f"{self.base_url}/generate_with_soul"
        self.embed_url = f"{self.base_url}/embed"
        self.timeout = timeout

        self.ssl_ctx = ssl.create_default_context()
        self.ssl_ctx.check_hostname = False
        self.ssl_ctx.verify_mode = ssl.CERT_NONE


    async def get_embedding(self, text: str) -> list:
        payload = {"text": text, "layer_idx": -1}
        max_retries = 3
        for attempt in range(max_retries):
            try:
                timeout_cfg = aiohttp.ClientTimeout(total=30)
                async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
                    async with session.post(self.embed_url, json=payload) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            return data["vector"]
                        return []
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(1) # Espera o Windows liberar a porta
                    continue
                logger.error(f"🔗 Falha persistente na Engine: {e}")

                return []

    async def ensure_vector_exists(self, concept_name: str) -> bool:
        """Verifica se a nuvem tem o vetor. Se não tiver, faz upload do PC."""
        try:
            async with aiohttp.ClientSession() as session:
                # 1. Checa se o conceito existe na nuvem
                async with session.get(f"{self.base_url}/concepts") as resp:
                    concepts = (await resp.json()).get("concepts", [])

                if concept_name in concepts:
                    return True  # Já está lá

                # 2. Se não está lá, procura na pasta local do seu PC
                local_path = os.path.join("vectors", f"{concept_name}.npy")
                if not os.path.exists(local_path):
                    logger.error(f"❌ Vetor {concept_name} não existe nem na nuvem nem no PC!")
                    return False

                # 3. Faz o upload (Restauração em tempo real)
                logger.warning(f"💊 Restaurando medicamento '{concept_name}' para a Vast.ai...")
                data = aiohttp.FormData()
                data.add_field('file', open(local_path, 'rb'), filename=f"{concept_name}.npy")

                async with session.post(f"{self.base_url}/vectors/upload", data=data) as up_resp:
                    return up_resp.status == 200

        except Exception as e:
            logger.error(f"Falha na sincronização de vetores: {e}")
            return False

    async def generate(self, prompt: str, **kwargs) -> str:
        # 1. Extrai os dados do vetor (se houver)
        vector_data = kwargs.get("vector_data")
        vector_name = vector_data.get("concept") if vector_data else None

        # 2. SE houver um pedido de hormônio (vetor), garante que ele existe na nuvem antes de prosseguir
        if vector_name:
            # Esta função vai checar a Vast.ai e fazer o upload do seu PC se necessário
            await self.ensure_vector_exists(vector_name)

        # 3. Prepara o payload para a Soul Engine
        payload = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 2000),
            "temperature": kwargs.get("temperature", 0.6),
            "vector_name": vector_name,
            "intensity": vector_data.get("intensity", 0.0) if vector_data else 0.0,
            "layer_idx": vector_data.get("layer_idx", 16) if vector_data else 16
        }

        # 4. Executa a requisição de geração
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.generate_url, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["text"]

                    # Se der erro 400 (Vetor não encontrado), algo falhou no upload
                    error_detail = await resp.text()
                    logger.error(f"❌ Erro na Engine: {resp.status} - {error_detail}")
                    return f"[Error: {resp.status}]"

        except Exception as e:
            logger.error(f"🔗 Falha de Conexão na Engine (Generate): {e}")
            return "[Connection Error]"

    async def _send_request(self, payload: Dict[str, Any], allow_fallback: bool = True) -> str:
        try:
            timeout_settings = aiohttp.ClientTimeout(total=self.timeout)

            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=self.ssl_ctx),
                                             timeout=timeout_settings) as session:
                async with session.post(self.generate_url, json=payload) as resp:

                    if resp.status != 200:
                        text = await resp.text()

                        # TRATAMENTO ESPECÍFICO PARA ERRO 400 (Conceito não encontrado)
                        if resp.status == 400 and "Conceito" in text and not allow_fallback:
                            # Retorna marcador especial para o método generate fazer o retry
                            return f"[ERROR: Concept Missing] {text}"

                        logger.error(f"❌ Vast.AI HTTP {resp.status}: {text}")
                        return f"[ERROR: Vast.AI returned {resp.status}]"

                    result = await resp.json()
                    full_text = result.get("text", "").strip()

                    # Basic Cleanup (Remove prompt echo if present)
                    sent_prompt = payload.get("prompt", "").strip()
                    if full_text.startswith(sent_prompt):
                        full_text = full_text[len(sent_prompt):].strip()

                    return full_text

        except asyncio.TimeoutError:
            logger.error("❌ Vast.AI Timeout.")
            return "[ERROR: Vast.AI Timeout]"
        except Exception as e:
            logger.error(f"❌ Vast.AI Connection Error: {e}")
            return f"[ERROR: Connection Failed: {str(e)}]"