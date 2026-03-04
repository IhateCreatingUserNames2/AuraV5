# ceaf_core/modules/vector_lab.py

import logging
import asyncio
import numpy as np
import os
from typing import List, Dict, Tuple, Optional
import json
import httpx
from dotenv import load_dotenv

load_dotenv()

# Dependências internas
from ceaf_core.services.llm_service import LLMService
from ceaf_core.utils.common_utils import extract_json_from_text

SOUL_ENGINE_URL = os.getenv("VASTAI_ENDPOINT", "http://199.68.217.31:54213").rstrip("/")

# Importa o SoulScanner (que deve estar acessível no path)
# Nota: Em produção, o SoulScanner deveria ser um cliente API para o soul_engine,
# mas para rodar no worker local com GPU, importamos direto.
try:
    from soul_scanner import SoulScanner

    SCANNER_AVAILABLE = True
except ImportError:
    SCANNER_AVAILABLE = False

logger = logging.getLogger("VectorLab")


class VectorLab:
    """
    O Laboratório de Vetores Autônomo da Aura.
    Gera dados, cria testes, escaneia o cérebro (modelo) e sintetiza novos vetores de comportamento.
    """

    def __init__(self, llm_service: LLMService, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str = "cuda"):
        self.llm = llm_service
        self.model_id = model_id
        self.device = device
        self.scanner = None

        if SCANNER_AVAILABLE:
            # Inicialização preguiçosa para não ocupar VRAM se não for usar
            pass
        else:
            logger.warning("SoulScanner não encontrado. Otimização de vetores desabilitada.")

    def _ensure_scanner(self):
        """Inicializa o scanner sob demanda."""
        if self.scanner is None and SCANNER_AVAILABLE:
            logger.info(f"VectorLab: Carregando modelo {self.model_id} para escaneamento...")
            try:
                self.scanner = SoulScanner(model_id=self.model_id, device=self.device)
            except Exception as e:
                logger.error(f"VectorLab: Falha ao carregar SoulScanner: {e}")
                raise e

    async def _generate_training_data(self, concept_name: str) -> Tuple[List[str], List[str]]:
        """
        V3: Extração robusta para lidar com listas, objetos ou formatos aninhados.
        """
        logger.info(f"🧪 VectorLab: Gerando dataset sintético para '{concept_name}'...")

        prompt = f"""
           Gere um dataset JSON para o conceito: "{concept_name}".
           Retorne APENAS um objeto JSON com as chaves "positive" e "negative", 
           cada uma contendo uma lista de 10 frases.
           Exemplo: {{"positive": ["...", "..."], "negative": ["...", "..."]}}
           """

        try:
            response = await self.llm.ainvoke(self.llm.config.smart_model, prompt, temperature=0.7)
            data = extract_json_from_text(response)

            if not data:
                raise ValueError("LLM não retornou JSON válido.")

            pos, neg = [], []

            # Cenário A: O LLM retornou um dicionário correto
            if isinstance(data, dict):
                pos = data.get("positive", [])
                neg = data.get("negative", [])

            # Cenário B: O LLM retornou uma lista direta [...]
            elif isinstance(data, list):
                logger.warning("VectorLab: LLM retornou lista. Tentando split 50/50...")
                mid = len(data) // 2
                pos = [str(i) for i in data[:mid]]
                neg = [str(i) for i in data[mid:]]

            # Validação Final
            if not pos or not neg:
                logger.error(f"VectorLab: Falha ao segmentar dados. Tipo recebido: {type(data)}")
                return [], []

            return [str(x) for x in pos], [str(x) for x in neg]

        except Exception as e:
            logger.error(f"VectorLab: Erro crítico ao gerar dados: {e}", exc_info=True)
            return [], []

    async def _generate_dynamic_arena(self, concept_name: str, positive_samples: List[str]) -> Tuple[
        List[str], List[str]]:
        """
        Cria um teste unitário (Arena) específico para o conceito.
        """
        logger.info(f"🧪 VectorLab: Criando Arena Dinâmica para '{concept_name}'...")

        samples_preview = "\n".join(positive_samples[:3])

        prompt = f"""
        O conceito é: "{concept_name}".
        Exemplos do comportamento desejado:
        {samples_preview}

        Sua tarefa é criar um "Campo de Batalha" (Arena) para validar se a IA aprendeu.
        1. Gere 5 prompts de usuário (inputs) que desafiariam a IA a usar esse traço.
        2. Gere 5 palavras-chave ou frases curtas que, se aparecerem na resposta, indicam SUCESSO.

        Responda APENAS com um JSON:
        {{
            "battlefield_questions": ["pergunta 1", ...],
            "victory_keywords": ["keyword 1", ...]
        }}
        """

        try:
            response = await self.llm.ainvoke(self.llm.config.smart_model, prompt, temperature=0.7)
            data = extract_json_from_text(response)
            return data.get("battlefield_questions", []), data.get("victory_keywords", [])
        except Exception as e:
            logger.error(f"VectorLab: Falha na Arena Generator: {e}")
            return [], []

    async def run_optimization_cycle(self, concept_name: str, target_layer: int = 16) -> str:
        """
        Workflow Automático:
        1. Gera dados (CPU/API Externa)
        2. Envia para Soul Engine (GPU Remota)
        3. Valida resultado
        """
        # 1. Geração de Dados
        pos, neg = await self._generate_training_data(concept_name)

        if not pos:
            return f"Falha: Não foi possível gerar dados para {concept_name}"

        logger.info(f"📊 Dados gerados: {len(pos)} pares. Enviando para Soul Engine em {SOUL_ENGINE_URL}...")

        # 2. Payload para a API de Calibração
        payload = {
            "concept_name": concept_name,
            "positive_samples": pos,
            "negative_samples": neg,
            "layer_idx": target_layer
        }

        # 3. Requisição HTTP para a GPU
        try:
            async with httpx.AsyncClient(timeout=420.0) as client:
                # Chama o novo endpoint /calibrate que criamos no soul_engine
                resp = await client.post(f"{SOUL_ENGINE_URL}/calibrate", json=payload)

                if resp.status_code == 200:
                    data = resp.json()

                    # --- NOVO: BACKUP AUTOMÁTICO PARA O PC ---
                    try:
                        async with httpx.AsyncClient(timeout=60.0) as client:
                            # Baixa o arquivo recém criado
                            download_url = f"{SOUL_ENGINE_URL}/vectors/{concept_name}"
                            v_resp = await client.get(download_url)

                            if v_resp.status_code == 200:
                                local_path = os.path.join("vectors", f"{concept_name}.npy")
                                os.makedirs("vectors", exist_ok=True)
                                with open(local_path, "wb") as f:
                                    f.write(v_resp.content)
                                logger.critical(f"💾 BACKUP LOCAL REALIZADO: {local_path}")
                    except Exception as backup_err:
                        logger.error(f"Falha no backup local do vetor: {backup_err}")

                    return f"✅ SUCESSO: Vetor '{concept_name}' cristalizado."

                else:
                    error_msg = f"Erro {resp.status_code} no Soul Engine: {resp.text}"
                    logger.error(error_msg)
                    return error_msg

        except Exception as e:
            logger.error(f"VectorLab: Erro de conexão com Soul Engine: {e}")
            return f"Falha de conexão: {e}"