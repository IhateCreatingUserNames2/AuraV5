# ceaf_core/utils/embedding_utils.py

import asyncio
import logging
import os
import numpy as np
from typing import List, Optional, Union, Dict, Any
import litellm

# --- CORREÇÃO 1: CARREGAR ENV IMEDIATAMENTE ---
from dotenv import load_dotenv

load_dotenv()

# Import the Vast Client (que também serve para Local Soul Engine)
from ceaf_core.services.vast_ai_engine import VastAIEngine

# Tenta importar SentenceTransformer apenas se não estivermos forçando o modo remoto
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    SentenceTransformer = None  # type: ignore

logger = logging.getLogger(__name__)

# --- Configuração Global ---
# Define quem é a autoridade sobre a geometria vetorial
INFERENCE_MODE = os.getenv("INFERENCE_MODE", "local")
GEOMETRIC_DIMENSION = int(os.getenv("GEOMETRIC_DIMENSION"))
DEFAULT_MODEL_NAME = os.getenv("MODEL_ID")

# --- Configurações Legadas (Fallback Local) ---
DEFAULT_EMBEDDING_PROVIDER = os.getenv("CEAF_EMBEDDING_PROVIDER", "sentence_transformers")
DEFAULT_EMBEDDING_MODEL_FOR_CLIENT = os.getenv("CEAF_DEFAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

EMBEDDING_MODELS_CONFIG: Dict[str, str] = {
    "default_query": DEFAULT_EMBEDDING_MODEL_FOR_CLIENT,
    "DEFAULT_FALLBACK": DEFAULT_EMBEDDING_MODEL_FOR_CLIENT
}


class EmbeddingClient:
    """
    Cliente unificado para geração de embeddings.
    Atua como um roteador:
    - Se INFERENCE_MODE='vastai': Delega para o Qwen remoto (Geometria Real).
    - Se INFERENCE_MODE='local': Usa SentenceTransformers/LiteLLM na CPU local.
    """

    def __init__(
            self,
            provider: Optional[str] = None,
            default_model_name: Optional[str] = None,
            litellm_kwargs: Optional[Dict[str, Any]] = None
    ):
        self.inference_mode = INFERENCE_MODE
        self.geometric_engine = None
        self._st_model_cache: Dict[str, SentenceTransformer] = {}

        logger.info(f"🔌 EmbeddingClient Init | Mode: {self.inference_mode.upper()} | Dim: {GEOMETRIC_DIMENSION}")

        # 1. Configuração para Modo VAST.AI / SOUL ENGINE LOCAL (Geometria Unificada)
        if self.inference_mode == "vastai":
            logger.info(
                f"🌌 MODO GEOMÉTRICO ATIVO: Conectando ao Soul Engine em {os.getenv('VASTAI_ENDPOINT')}...")
            try:
                self.geometric_engine = VastAIEngine()
            except Exception as e:
                logger.critical(f"❌ Falha ao inicializar Engine Remota: {e}")

        # 2. Configuração para Modo LOCAL (Legado / Fallback)
        else:
            self.provider = provider or "litellm"
            self.default_model_name = default_model_name or DEFAULT_MODEL_NAME
            self.litellm_kwargs = litellm_kwargs or {}

            logger.info(f"🏠 MODO LOCAL ISOLADO: Usando {self.provider} com {self.default_model_name}.")

            if self.provider == "sentence_transformers" and not SENTENCE_TRANSFORMER_AVAILABLE:
                logger.error("SentenceTransformer selecionado mas biblioteca não instalada.")

    def _resolve_model_name(self, context_type: Optional[str]) -> str:
        """Apenas para modo local: resolve o nome do modelo MiniLM."""
        if context_type and context_type in EMBEDDING_MODELS_CONFIG:
            return EMBEDDING_MODELS_CONFIG[context_type]
        return EMBEDDING_MODELS_CONFIG.get("DEFAULT_FALLBACK", self.default_model_name)

    def _get_st_model(self, model_name_str: str) -> SentenceTransformer:
        """Carrega modelo localmente (apenas modo local fallback)."""
        if not SENTENCE_TRANSFORMER_AVAILABLE:
            raise RuntimeError("biblioteca sentence-transformers não instalada.")

        if model_name_str not in self._st_model_cache:
            logger.info(f"Carregando modelo local: {model_name_str}")
            try:
                trust_code = any(kc in model_name_str for kc in ["nomic-ai/", "jinaai/"])
                self._st_model_cache[model_name_str] = SentenceTransformer(
                    model_name_str, device='cpu', trust_remote_code=trust_code
                )
            except Exception as e:
                logger.error(f"Erro ao carregar modelo local: {e}")
                # Fallback para default
                if self.default_model_name:
                    self._st_model_cache[model_name_str] = SentenceTransformer(self.default_model_name)

        return self._st_model_cache[model_name_str]

    async def encode(self, text: Union[str, List[str]], **kwargs):
        if isinstance(text, list):
            return await self.get_embeddings(text)
        return await self.get_embedding(text)

    async def get_embedding(self, text: str, context_type: Optional[str] = None, **kwargs) -> List[float]:
        """
        Obtém o embedding do texto.
        ROTA CRÍTICA: Decide entre Soul Engine (Qwen/2048d) e Local (MiniLM/384d).
        """
        # 1. Tratamento de input vazio ou inválido
        if not text or not isinstance(text, str):
            logger.warning("Recebido texto vazio para embedding. Gerando vetor nulo.")
            # Retorna zeros na dimensão esperada pelo Qdrant para não crashar
            dim = GEOMETRIC_DIMENSION if self.inference_mode == "vastai" else 384
            return [0.0] * dim

        # --- ROTA 1: SOUL ENGINE (Modo vastai no .env) ---
        if self.inference_mode == "vastai":
            if self.geometric_engine:
                try:
                    # Chama o soul_engine_local.py (localhost:8093)
                    vector = await self.geometric_engine.get_embedding(text)

                    # Validação Crítica de Dimensão
                    if vector and len(vector) == GEOMETRIC_DIMENSION:
                        return vector

                    # Se o vetor veio com tamanho errado (ex: 384 em vez de 2048)
                    if vector:
                        logger.error(
                            f"🚨 DIMENSÃO INCORRETA: Soul Engine retornou {len(vector)}d, mas o Qdrant espera {GEOMETRIC_DIMENSION}d.")

                except Exception as e:
                    logger.error(f"🔗 Falha de conexão com Soul Engine: {e}")

            # --- BLINDAGEM DE SEGURANÇA ---
            # Se chegamos aqui em modo 'vastai', a Engine falhou ou deu erro de tamanho.
            # RETORNAMOS ZEROS NA DIMENSÃO CORRETA (2048) em vez de cair no MiniLM(384).
            # Isso evita o erro 400 "Vector dimension error" no Qdrant.
            logger.critical(
                f"⚠️ Falha na geometria unificada. Gerando vetor nulo de {GEOMETRIC_DIMENSION}d para estabilidade.")
            return [0.0] * GEOMETRIC_DIMENSION

        # --- ROTA 2: LOCAL FALLBACK (MiniLM) ---
        # Esta rota só é executada se INFERENCE_MODE for 'local' no seu .env
        actual_model_name = self._resolve_model_name(context_type)

        if self.provider == "sentence_transformers":
            try:
                st_model = self._get_st_model(actual_model_name)
                loop = asyncio.get_running_loop()
                embedding_array = await loop.run_in_executor(None, st_model.encode, text)
                return embedding_array.tolist()
            except Exception as e:
                logger.error(f"Erro SentenceTransformers local: {e}")

        elif self.provider == "litellm":
            try:
                final_kwargs = {**self.litellm_kwargs, **kwargs}
                response = await litellm.aembedding(
                    model=actual_model_name, input=[text], **final_kwargs
                )
                if response.data:
                    return response.data[0].embedding
            except Exception as e:
                logger.error(f"Erro LiteLLM local: {e}")

        # Fallback final absoluto para modo local (384d)
        return [0.0] * 384

    async def get_embeddings(self, texts: List[str], context_type: Optional[str] = None, **kwargs) -> List[List[float]]:
        """Processamento em lote."""
        if not texts: return []

        if self.inference_mode == "vastai":
            # Faz chamadas paralelas para o endpoint
            tasks = [self.get_embedding(t) for t in texts]
            return await asyncio.gather(*tasks)

        # Modo Local Batch
        actual_model_name = self._resolve_model_name(context_type)
        if self.provider == "sentence_transformers":
            st_model = self._get_st_model(actual_model_name)
            loop = asyncio.get_running_loop()
            embeddings_array = await loop.run_in_executor(None, st_model.encode, texts)
            return [emb.tolist() for emb in embeddings_array]

        # Fallback Litellm Batch... (código original)
        return [[0.0] * 384 for _ in texts]


# --- Similarity Utilities ---
def cosine_similarity_np(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray): return 0.0
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0: return 0.0
    return float(np.dot(vec1, vec2) / (norm_vec1 * norm_vec2))


def compute_adaptive_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    if not embedding1 or not embedding2: return 0.0
    a = np.array(embedding1, dtype=np.float32)
    b = np.array(embedding2, dtype=np.float32)
    if a.shape != b.shape:
        min_dim = min(a.shape[0], b.shape[0])
        a = a[:min_dim]
        b = b[:min_dim]
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0: return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


_embedding_client_instance: Optional[EmbeddingClient] = None


def get_embedding_client(force_reinitialize=False) -> EmbeddingClient:
    global _embedding_client_instance
    if _embedding_client_instance is None or force_reinitialize:
        _embedding_client_instance = EmbeddingClient()
    return _embedding_client_instance