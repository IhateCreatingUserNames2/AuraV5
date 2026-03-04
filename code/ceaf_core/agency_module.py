#ceaf_core/agency_module.py

import asyncio
import json
import logging
import os
import uuid
from typing import Dict, Any, List, Literal, Optional, Union, Tuple

import torch
from sklearn.cluster import DBSCAN
import numpy as np
from pydantic import BaseModel, Field, ValidationError
import time
from ceaf_core.utils.embedding_utils import get_embedding_client

from neural_physics import ActionGenerator, PolicyNetwork, WorldModelPredictor
import aiohttp
from ceaf_core.utils import compute_adaptive_similarity
from ceaf_core.riemannian_geometry import RiemannianGeometry

from ceaf_core.monadic_base import AuraState
from ceaf_core.services.llm_service import LLMService
from ceaf_core.models import SystemPrompts, MCLConfig

from ceaf_core.utils import compute_adaptive_similarity
from ceaf_core.utils.observability_types import ObservabilityManager, ObservationType
from ceaf_core.genlang_types import CognitiveStatePacket, ResponsePacket, GenlangVector
from ceaf_core.utils.common_utils import extract_json_from_text
from ceaf_core.modules.vre_engine.vre_engine import VREEngineV3, ActionType
from ceaf_core.modules.interoception_module import ComputationalInteroception
from sentence_transformers import SentenceTransformer
from ceaf_core.modules.mcl_engine.mcl_engine import MCLEngine
import inspect

# Importar os avaliadores de primitivas não-LLM
from ceaf_core.agency_enhancements import eval_narrative_continuity, eval_specificity, eval_emotional_resonance
from ceaf_core.v4_sensors import AuraMonitor

logger = logging.getLogger("AgencyModule_V4_Intentional")

StatePredictor = WorldModelPredictor

# ==============================================================================
# 1. DEFINIÇÕES DE ESTRUTURA DE DADOS
# ==============================================================================


class ThoughtPathCandidate(BaseModel):
    candidate_id: str = Field(default_factory=lambda: f"th_path_{uuid.uuid4().hex[:8]}")
    decision_type: Literal["response_strategy", "tool_call"]
    strategy_description: Optional[str] = None
    key_memory_ids: Optional[List[str]] = None
    tool_call_request: Optional[Dict[str, Any]] = None
    reasoning: str

class WinningStrategy(BaseModel):
    decision_type: Literal["response_strategy", "tool_call"]
    strategy_description: Optional[str] = None
    key_memory_ids: Optional[List[str]] = None
    tool_call_request: Optional[Dict[str, Any]] = None
    reasoning: str
    predicted_future_value: float = 0.0
    # [Fase 3] Adicionamos o vetor de ação para logar
    action_vector: Optional[List[float]] = None
    steering_override: Optional[Dict[str, Any]] = None

class ResponseCandidate(BaseModel):
    decision_type: Literal["response"]
    content: ResponsePacket
    reasoning: str


class ToolCallCandidate(BaseModel):
    decision_type: Literal["tool_call"]
    content: Dict[str, Any]
    reasoning: str


class AgencyDecision(BaseModel):
    decision: Union[ResponseCandidate, ToolCallCandidate] = Field(..., discriminator='decision_type')
    predicted_future_value: float = 0.0

    @property
    def decision_type(self): return self.decision.decision_type

    @property
    def content(self): return self.decision.content

    @property
    def reasoning(self): return self.decision.reasoning


class ProjectedFuture(BaseModel):
    initial_candidate: AgencyDecision
    simulated_turns: List[Dict[str, str]] = Field(default_factory=list)
    final_cognitive_state_summary: Dict[str, Any]
    simulated_tool_result: Optional[str] = None


class FutureEvaluation(BaseModel):
    total_value: float = 0.0


# ==============================================================================
# 2. IMPLEMENTAÇÃO DO MÓDULO DE AGÊNCIA (COM INTENÇÃO)
# ==============================================================================

def generate_tools_summary(tool_registry: Dict[str, callable]) -> str:
    summary_lines = []
    for tool_name, tool_function in tool_registry.items():
        try:
            signature = inspect.signature(tool_function)
            params = []
            for param_name, param in signature.parameters.items():
                if param_name in ['self', 'cls', 'observer', 'tool_context']:
                    continue
                param_type = param.annotation if param.annotation != inspect.Parameter.empty else 'Any'
                params.append(
                    f"{param_name}: {param_type.__name__ if hasattr(param_type, '__name__') else str(param_type)}")
            param_str = ", ".join(params)
            docstring = inspect.getdoc(tool_function)
            description = docstring.strip().split('\n')[0] if docstring else "Nenhuma descrição disponível."
            summary_lines.append(f"- `{tool_name}({param_str})`: {description}")
        except (TypeError, ValueError) as e:
            logger.warning(f"Não foi possível gerar a assinatura para a ferramenta '{tool_name}': {e}")
            summary_lines.append(f"- `{tool_name}(...)`: Descrição não pôde ser gerada automaticamente.")
    return "\n".join(summary_lines)


class AgencyModule:
    """
    Módulo de Agência V4 (Intentional).
    Implementa o FutureSimulator e o PathEvaluator do manifesto.
    """

    def __init__(self, llm_service: LLMService, vre_engine: VREEngineV3, mcl_engine: 'MCLEngine',
                 available_tools_summary: str, prompts: SystemPrompts = None, agency_config: MCLConfig = None):

        self.llm = llm_service
        self.vre = vre_engine
        self.mcl = mcl_engine
        self.available_tools_summary = available_tools_summary
        self.prompts = prompts or SystemPrompts()
        self.config = agency_config or MCLConfig()

        self.max_deliberation_time = 45.0
        self.deliberation_budget_tiers = {
            "deep": {"max_candidates": 2, "simulation_depth": 1, "recursive_steps": 2},
            "medium": {"max_candidates": 2, "simulation_depth": 0, "recursive_steps": 1},
            "shallow": {"max_candidates": 1, "simulation_depth": 0, "recursive_steps": 0},
            "emergency": {"max_candidates": 1, "simulation_depth": 0, "recursive_steps": 0}
        }

        self.vector_dim = int(os.getenv("GEOMETRIC_DIMENSION", "5120"))
        self.embedding_model = get_embedding_client()

        self.hormone_vectors: Dict[str, np.ndarray] = {}

        # Inicializa Redes Neurais (Policy + World Model)
        self.policy_network = PolicyNetwork(state_dim=self.vector_dim)
        self.world_model = WorldModelPredictor(state_dim=self.vector_dim)

        self.brain_active = False
        self.world_model_active = False
        self.social_target_vector = None

        self.hormone_signatures: Dict[str, List[float]] = {}
        self._raw_hormone_definitions = {
            "Creative_Chaos": "high entropy, randomness, divergent thinking, brainstorming, unexpected ideas",
            "Stoic_Calmness": "rationality, logic, stability, factual accuracy, emotionless analysis",
            "High_Empathy": "emotional connection, warmth, understanding, support, listening",
            "Socratic_Questioning": "inquiry, curiosity, asking deep questions, exploring reasons"
        }

        # Carrega tudo de forma síncrona/segura
        self._initialize_neural_systems()

    async def _init_social_target(self):
        """
        Define o que a Aura considera um 'Bom Resultado' social.
        Isso é o U_target (Usuário Satisfeito).
        """
        # Conceitos de sucesso na interação
        positive_concepts = [
            "User feels understood and satisfied.",
            "Clear and helpful explanation.",
            "Agreement and appreciation.",
            "Problem solved successfully."
        ]
        # Gera embeddings e faz a média
        vectors = await self.embedding_model.get_embeddings(positive_concepts)
        if vectors:
            avg_vec = np.mean(vectors, axis=0)
            # Normaliza
            norm = np.linalg.norm(avg_vec)
            if norm > 0: avg_vec = avg_vec / norm

            self.social_target_vector = torch.tensor(avg_vec, dtype=torch.float32)
            logger.info("🎯 Vetor Alvo Social (Satisfação) calibrado.")

    async def rank_candidates_neurally(
            self,
            candidates: List[ThoughtPathCandidate],
            current_state: CognitiveStatePacket
    ) -> List[Tuple[ThoughtPathCandidate, float]]:
        """
        Ranqueia candidatos rapidamente usando o World Model (Física Neural).
        Simula S_t+1 para todos os candidatos em paralelo (batch) sem chamar LLM.
        """
        last_ai_vec = current_state.metadata.get("last_ai_action_vector")


        if not self.world_model_active or not candidates:
            return [(c, 0.5) for c in candidates]

        try:
            # 1. Preparar Vetores (Batching)
            s_vec = current_state.identity_vector.vector
            u_vec = current_state.original_intent.query_vector.vector

            # Garante que os vetores existem
            if not s_vec or not u_vec:
                return [(c, 0.5) for c in candidates]

            # Vetoriza os textos dos candidatos (Ação A_t) em lote
            candidate_texts = [c.strategy_description or str(c.tool_call_request) for c in candidates]
            a_vecs_list = await self.embedding_model.get_embeddings(candidate_texts)

            # Converte para Tensores PyTorch
            batch_size = len(candidates)
            t_s = torch.tensor(s_vec, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
            t_u = torch.tensor(u_vec, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
            t_a = torch.tensor(a_vecs_list, dtype=torch.float32)

            # 2. Simulação Neural (Forward Pass - Milissegundos)
            with torch.no_grad():
                agent_deltas, user_reactions, _ = self.world_model(t_s, t_u, t_a)

            # 3. Scoring (Estabilidade e Coerência)
            scores = []
            for i in range(batch_size):
                # 1. Penalidade de Estagnação (Auto-Repetição)
                repetition_penalty = 0.0
                if last_ai_vec:
                    # Compara o vetor do candidato atual com a última resposta
                    sim_to_last = compute_adaptive_similarity(a_vecs_list[i], last_ai_vec)
                    if sim_to_last > 0.90:
                        repetition_penalty = (sim_to_last - 0.85) * 2.0  # Penaliza pesado

                # 2. Score de Estabilidade (O que você já tinha)
                stability = 1.0 - torch.norm(agent_deltas[i]).item()

                # Score Final: Queremos estabilidade de identidade, mas NOVIDADE de ação!
                final_score = stability - repetition_penalty  # <--- A MUDANÇA

                scores.append((candidates[i], max(0.0, final_score)))

            # Ordena do maior score para o menor
            scores.sort(key=lambda x: x[1], reverse=True)

            logger.info(f"🧠 Neural MPC: Reranqueou {batch_size} candidatos via Física Neural.")
            return scores

        except Exception as e:
            logger.error(f"Erro no Neural MPC: {e}")
            return [(c, 0.5) for c in candidates]


    async def predict_social_impact(self, candidates: List[ThoughtPathCandidate],
                                    cognitive_state: CognitiveStatePacket) -> Dict[str, float]:
        """
        Usa o World Model para prever a reação do usuário a cada candidato.
        Retorna um dicionário {candidate_id: social_score}.
        """
        if not self.world_model_active or self.social_target_vector is None:
            return {}

        scores = {}

        # 1. Preparar Entradas Estáticas (S_t e U_t)
        # S_t: Identidade Atual
        s_vec = cognitive_state.identity_vector.vector
        # U_t: O que o usuário acabou de dizer
        u_vec = cognitive_state.original_intent.query_vector.vector

        if not s_vec or not u_vec:
            return {}

        # Converter para Tensores (Batch size 1, depois expandimos)
        t_s = torch.tensor(s_vec, dtype=torch.float32).unsqueeze(0)
        t_u = torch.tensor(u_vec, dtype=torch.float32).unsqueeze(0)

        logger.info(f"🔮 Simulando {len(candidates)} futuros sociais via World Model...")

        for cand in candidates:
            try:
                # 2. Vetorizar a Ação Candidata (A_t)
                text_action = cand.strategy_description or str(cand.tool_call_request)
                # Embedar a ação (Async)
                a_vec_list = await self.embedding_model.get_embedding(text_action)
                t_a = torch.tensor(a_vec_list, dtype=torch.float32).unsqueeze(0)

                # 3. Forward Pass (A Mágica Neural)
                with torch.no_grad():
                    # A rede prevê: (Mudança em Mim, Reação do Usuário)
                    _, pred_user_reaction, _ = self.world_model(t_s, t_u, t_a)

                # 4. Avaliação: Quão perto isso está do "Usuário Satisfeito"?
                # Similaridade de Cosseno entre Predição e Alvo
                sim = torch.nn.functional.cosine_similarity(pred_user_reaction, self.social_target_vector.unsqueeze(0))
                score = float(sim.item())

                # Normaliza score (-1 a 1 -> 0 a 100 de bônus, por exemplo)
                # Aqui vamos retornar um multiplicador ou aditivo direto
                scores[cand.candidate_id] = score

                logger.debug(f"   > Cand '{cand.candidate_id[:8]}': Social Score = {score:.4f}")

            except Exception as e:
                logger.error(f"Erro na simulação neural para candidato {cand.candidate_id}: {e}")
                scores[cand.candidate_id] = 0.0

        return scores

    def update_config(self, new_prompts: SystemPrompts, new_config: MCLConfig):
        self.prompts = new_prompts
        self.config = new_config

    def _initialize_neural_systems(self):
        """
        Carrega os pesos das redes neurais (Policy e World Model) e inicializa vetores alvo.
        """
        brain_path = "./aura_brain/"

        try:
            # 1. Carrega Policy Network (Comportamento)
            if os.path.exists(f"{brain_path}policy.pth"):
                self.policy_network.load_state_dict(torch.load(f"{brain_path}policy.pth"))
                self.policy_network.eval()
                self.brain_active = True
                logger.info("🧠 AgencyModule: Policy Network (Intuição) carregada.")
            else:
                logger.warning("⚠️ AgencyModule: Policy Network não encontrada. Rodará sem intuição.")

            # 2. Carrega World Model (Córtex Social)
            if os.path.exists(f"{brain_path}world_model.pth"):
                self.world_model.load_state_dict(torch.load(f"{brain_path}world_model.pth"))
                self.world_model.eval()
                self.world_model_active = True
                logger.info("🧠 AgencyModule: World Model (Simulação Social) carregado.")

                # Inicia a calibração do alvo social em background (não bloqueante)
                asyncio.create_task(self._init_social_target())
            else:
                logger.warning("⚠️ AgencyModule: World Model não encontrado. Rodará sem empatia neural.")

            # 3. Carrega Vetores de Hormônios (Intervenção)
            self._load_hormone_vectors("./vectors/")

        except Exception as e:
            logger.error(f"🔥 Erro crítico ao inicializar sistemas neurais: {e}", exc_info=True)
            self.brain_active = False
            self.world_model_active = False

    def _load_hormone_vectors(self, vector_dir: str):
        """
        Carrega os vetores de intervenção reais para a memória.
        """
        if not os.path.exists(vector_dir):
            logger.warning(f"Diretório de vetores {vector_dir} não encontrado.")
            return

        for f in os.listdir(vector_dir):
            if f.endswith(".npy"):
                name = f.replace(".npy", "")
                try:
                    # Carrega o arquivo numpy
                    vec = np.load(os.path.join(vector_dir, f))

                    # Salva no dicionário que agora existe
                    self.hormone_vectors[name] = vec.flatten()
                    logger.info(f"💉 Hormônio Carregado: {name}")
                except Exception as e:
                    logger.error(f"Erro ao carregar vetor {f}: {e}")

    async def _sync_hormones(self):
        """
        Sincroniza os vetores do servidor Vast.AI para a memória local.
        """
        known_hormones = ["Creative_Chaos", "Stoic_Calmness", "High_Empathy", "Socratic_Questioning"]

        for name in known_hormones:
            if name not in self.hormone_vectors:
                try:
                    url = f"{self.vast_url}/vectors/{name}"
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url) as resp:
                            if resp.status == 200:
                                data = await resp.read()
                                import io
                                vec = np.load(io.BytesIO(data))
                                self.hormone_vectors[name] = vec.flatten()
                                logger.info(f"📥 Hormônio Sincronizado: {name}")
                except Exception as e:
                    logger.warning(f"Falha ao baixar hormônio {name}: {e}")

    async def _ensure_hormone_signatures(self):
        """Garante que as assinaturas dos hormônios estejam embedadas."""
        if not self.hormone_signatures and hasattr(self, '_raw_hormone_definitions'):
            logger.info("💉 Indexando assinaturas de hormônios (Matchmaking Semântico)...")
            for name, desc in self._raw_hormone_definitions.items():
                try:
                    # Usa o cliente unificado para obter o embedding da descrição
                    vec = await self.llm.embedding_client.get_embedding(desc)
                    self.hormone_signatures[name] = vec
                except Exception as e:
                    logger.error(f"Erro ao indexar hormônio {name}: {e}")

    async def _consult_geometric_policy(self, state_vector: List[float], goal_vector: List[float],
                                        context_vectors: List[List[float]], novelty_bias: float) -> Dict[str, Any]:
        """
        Consulta a PolicyNetwork e faz o Matchmaking ESPECTRAL avançado.
        Pondera a similaridade direcional com a Massa Espectral no subespaço ativo.
        """
        if not self.brain_active or not state_vector:
            return None

        await self._ensure_hormone_signatures()

        try:
            # 1. Inferência da Política (A*)
            t_state = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0)
            target = goal_vector if goal_vector else state_vector
            t_goal = torch.tensor(target, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                ideal_semantic_vector = self.policy_network(t_state, t_goal)[0].numpy().tolist()

            best_hormone = None
            best_score = -1.0

            for h_name, h_sig_vector in self.hormone_signatures.items():
                # 2A. Alinhamento Direcional (Intenção da Policy)
                sim_intent = compute_adaptive_similarity(ideal_semantic_vector, h_sig_vector)

                # 2B. Alinhamento Estrutural (Massa Espectral no Contexto)
                # Se o MCL pede NOVIDADE (novelty_bias alto), preferimos hormônios com BAIXA massa espectral (ortogonais/fora da caixa).
                # Se pede COERÊNCIA (novelty_bias baixo), preferimos ALTA massa espectral (dentro da caixa/seguros).
                spectral_mass = self._compute_spectral_mass(h_sig_vector, context_vectors)

                if novelty_bias > 0.6:
                    # Busca novidade: Inverte a massa espectral (queremos explorar novos subespaços)
                    structural_alignment = 1.0 - spectral_mass
                else:
                    # Busca coerência: Mantém a massa espectral (queremos estabilidade)
                    structural_alignment = spectral_mass

                # 2C. Fusão Espectral (Score Combinado)
                # A política dita O QUE fazer, a topologia dita COMO fazer com segurança.
                final_score = (0.7 * sim_intent) + (0.3 * structural_alignment)

                if final_score > best_score:
                    best_score = final_score
                    best_hormone = h_name

            # Threshold de ativação espectral
            steering_suggestion = None
            if best_hormone and best_score > 0.55:  # Threshold híbrido
                steering_suggestion = {
                    "concept": best_hormone,
                    "intensity": float(3.0 * best_score),
                    "layer_idx": 16
                }
                logger.info(f"💉 Policy Match Espectral: {best_hormone} (Score Híbrido: {best_score:.2f})")

            return {
                "ideal_vector": ideal_semantic_vector,
                "steering_suggestion": steering_suggestion,
                "confidence": best_score
            }

        except Exception as e:
            logger.error(f"Erro na Policy Logic: {e}")
            return None


    async def _evaluate_candidate_with_simulation(
            self,
            candidate: ThoughtPathCandidate,
            cognitive_state: CognitiveStatePacket,
            mcl_guidance: Dict[str, Any],
            tier_config: Dict[str, Any],
            dynamic_temp: float,
            custom_weights: Dict[str, float] = None
    ) -> Tuple[float, FutureEvaluation]:
        """
        Avalia um candidato.
        CORREÇÃO V4.5: Projeção rigorosa no Manifold antes da simulação física.
        """

        # --- [AURA V2.1] PHYSICAL INTEGRITY CHECK (Fast Fail) ---
        if hasattr(self, 'physics_model') and self.physics_model is not None and self.embedding_model is not None:
            try:
                # 1. Vetorizar a Ação Candidata (A_t)
                text_to_embed = candidate.strategy_description or str(candidate.tool_call_request)
                action_vector = self.embedding_model.encode(text_to_embed)

                # 2. Vetorizar o Estado Atual (S_t)
                if hasattr(cognitive_state.identity_vector, 'vector') and cognitive_state.identity_vector.vector:
                    current_state_vec = np.array(cognitive_state.identity_vector.vector, dtype=np.float32)
                else:
                    current_state_vec = np.zeros(384, dtype=np.float32)

                # Garante tamanho correto
                if current_state_vec.shape[0] != 384:
                    current_state_vec = np.zeros(384, dtype=np.float32)

                # === BLINDAGEM GEOMÉTRICA ===
                # Importante: Projeta para dentro da bola (0.95) ANTES de qualquer conta
                # Isso impede a singularidade na borda (Norma=1.0)
                from ceaf_core.riemannian_geometry import RiemannianGeometry

                current_state_vec = RiemannianGeometry.project_to_manifold(current_state_vec, max_norm=0.9)
                action_vector = RiemannianGeometry.project_to_manifold(action_vector, max_norm=0.9)
                # ============================

                # Conversão para Tensores
                t_state = torch.tensor(current_state_vec, dtype=torch.float32).unsqueeze(0)
                t_action = torch.tensor(action_vector, dtype=torch.float32).unsqueeze(0)

                # 3. Previsão Neural: P(S_t+1 | S_t, A_t)
                with torch.no_grad():
                    predicted_delta = self.physics_model(t_state, t_action)

                # 4. Validação Geométrica
                # Aplica a mudança prevista
                new_state_pos = RiemannianGeometry.exp_map(current_state_vec, predicted_delta[0].numpy())

                # Garante que o resultado também esteja no manifold
                new_state_pos = RiemannianGeometry.project_to_manifold(new_state_pos, max_norm=0.95)

                # Calcula a tensão (Distância do centro 0,0 - Identidade Pura)
                origin = np.zeros_like(new_state_pos)
                integrity_tension = RiemannianGeometry.poincare_distance(new_state_pos, origin)

                # 5. O Veto Físico
                # Limite aumentado levemente pois a geometria curva distorce distâncias
                INTEGRITY_LIMIT = 0.98

                if integrity_tension > INTEGRITY_LIMIT:
                    logger.warning(
                        f"⚠️ ALERTA FÍSICO: Tensão alta ({integrity_tension:.4f}). "
                        f"Ação: '{text_to_embed[:30]}...'"
                    )
                    # Penalidade suave em vez de veto total (-100), para permitir recuperação
                    return 0.1, FutureEvaluation(total_value=0.1)

            except Exception as e:
                logger.warning(f"⚠️ Erro no Check Físico (Ignorando): {str(e)}")

        # ---  IMAGINATION LAYER (LLM Simulation) ---
        # Se chegou até aqui, a ação é fisicamente segura. Agora avaliamos se ela é inteligente.

        # Cria um "fake_decision" para a simulação
        fake_decision = None
        if candidate.decision_type == "response_strategy":
            fake_decision = AgencyDecision(decision=ResponseCandidate(
                decision_type="response",
                content=ResponsePacket(
                    content_summary=candidate.strategy_description or "Estratégia geral."),
                reasoning=candidate.reasoning
            ))
        elif candidate.decision_type == "tool_call" and candidate.tool_call_request:
            text_for_sim = f"Vou usar a ferramenta {candidate.tool_call_request.get('tool_name', 'desconhecida')}."
            fake_decision = AgencyDecision(decision=ResponseCandidate(
                decision_type="response",
                content=ResponsePacket(content_summary=text_for_sim),
                reasoning=candidate.reasoning
            ))

        if not fake_decision:
            return 0.0, FutureEvaluation()

        # =====================================================================
        # 🚀 [V5 SPEED & IMMUNITY OPTIMIZATION]
        # Se o World Model (Física Neural) estiver online, NUNCA enviamos as
        # opções para o LLM simular. O LLM é vulnerável a jailbreaks longos.
        # Nós usamos apenas a Heurística VADER e a Matemática Neural que
        # já validou e ranqueou esses candidatos milissegundos antes!
        # =====================================================================
        if self.world_model_active:
            # Avaliação rápida (0.01 segundos) baseada em densidade e sentimento
            value = await self._heuristic_evaluation(fake_decision)

            # Como a rede neural já os ordenou no _rank_candidates_neurally,
            # nós apenas retornamos o valor base. Isso elimina a latência e a
            # vulnerabilidade de prompt injection durante a deliberação.
            return value, FutureEvaluation(total_value=value)

        else:
            # Fallback lento (Método Legado): Só usa o LLM se a rede neural Pytorch falhar.
            logger.warning("⚠️ World Model offline. Usando simulação LLM lenta.")
            if tier_config["simulation_depth"] == 0:
                value = await self._heuristic_evaluation(fake_decision)
                return value, FutureEvaluation(total_value=value)
            else:
                future, likelihood = await self._project_response_trajectory(
                    fake_decision,
                    cognitive_state,
                    depth=tier_config["simulation_depth"],
                    temperature=dynamic_temp
                )
                value_weights = custom_weights or mcl_guidance.get("value_weights", {})
                value, evaluation_details = await self._evaluate_trajectory(
                    future,
                    likelihood,
                    cognitive_state.identity_vector,
                    value_weights,
                    cognitive_state
                )
                return value, evaluation_details


    async def _heuristic_evaluation(self, candidate: AgencyDecision) -> float:
        """Avaliação rápida e não-LLM de um candidato."""
        # Se não for uma resposta, damos uma pontuação neutra para chamadas de ferramenta por enquanto
        if candidate.decision_type != "response":
            return 0.6

        # O conteúdo da simulação temporária é um ResponsePacket
        if not isinstance(candidate.content, ResponsePacket):
            return 0.5

        text = candidate.content.content_summary

        # Usaremos os avaliadores de agency_enhancements.py
        # Um bom candidato é específico e tem ressonância emocional moderada.
        specificity_score = await eval_specificity(text)
        resonance_score = await eval_emotional_resonance(text)

        # A heurística pode ser ajustada, mas um bom começo é:
        # Pontuação = 0.7 * Especificidade + 0.3 * Ressonância
        final_score = (0.7 * specificity_score) + (0.3 * resonance_score)
        return final_score


    def _select_deliberation_tier(self, mcl_params: Dict, reality_score: float) -> str:
        """Determina o nível de profundidade da deliberação com base no contexto (Lógica Otimizada)."""
        agency_score = mcl_params.get("mcl_analysis", {}).get("agency_score", 0.0)

        complexity = min(agency_score / 5.0, 1.0)

        # +++ INÍCIO DAS MUDANÇAS (Lógica de Seleção Agressiva) +++
        # Agora, a deliberação profunda só acontece se a complexidade for alta E a simulação for confiável.
        if complexity > 0.8 and reality_score > 0.7:
            tier = "deep"
        elif complexity > 0.5 and reality_score > 0.5:
            tier = "medium"
        else:
            tier = "shallow"  # Torna 'shallow' o padrão para a maioria dos casos
        # +++ FIM DAS MUDANÇAS +++

        logger.info(
            f"Deliberation Tier selected: '{tier}' (Complexity: {complexity:.2f}, Reality Score: {reality_score:.2f})")
        return tier

    def _cluster_memory_votes(self, cognitive_state: CognitiveStatePacket) -> List[Dict[str, Any]]:
        """
        MODIFIED: Agrupa os 'votos' dos vetores de memória em múltiplos clusters de consenso.
        Agora com tratamento de erro robusto para a clusterização.
        """
        active_memory_vectors = [
            (np.array(vec.vector), vec.metadata.get("memory_id"))
            for vec in cognitive_state.relevant_memory_vectors
            if vec.metadata.get("is_consensus_vector") != True
        ]

        if len(active_memory_vectors) < 3:
            return [{"status": "no_consensus", "reason": "Insufficient memories to form clusters."}]

        vectors_only = [v for v, mid in active_memory_vectors]

        # === MUDANÇA: Bloco try/except para a clusterização ===
        try:
            clustering = DBSCAN(eps=0.4, min_samples=2, metric='cosine')
            clustering.fit(vectors_only)
            if not hasattr(clustering, 'labels_'):
                raise AttributeError(
                    "O objeto de clusterização não possui o atributo 'labels_'. Verifique a instalação do scikit-learn.")
            labels = clustering.labels_
        except Exception as e:
            logger.error(f"AgencyModule: Falha crítica na clusterização DBSCAN: {e}. "
                         f"Isso pode ser causado pela falta da biblioteca 'scikit-learn'. "
                         f"Recorrendo a um fallback sem clusterização.")
            # Fallback: Se a clusterização falhar, retorna um estado de "sem consenso".
            return [{"status": "no_consensus", "reason": f"Clustering algorithm failed: {e}"}]
        # ==================== FIM DA MUDANÇA ====================

        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        if not unique_labels:
            return [{"status": "no_consensus", "reason": "No significant clusters found by DBSCAN."}]

        all_clusters = []
        for label in unique_labels:
            cluster_indices = [i for i, lbl in enumerate(labels) if lbl == label]

            cluster_vectors = [vectors_only[i] for i in cluster_indices]

            # (O resto da lógica para calcular o consenso, texto representativo, etc. permanece o mesmo)
            consensus_vector = np.mean(cluster_vectors, axis=0)

            highest_sim = -1.0
            representative_text = "a collective thought"
            for i in cluster_indices:
                vec_obj = cognitive_state.relevant_memory_vectors[i]
                sim = compute_adaptive_similarity(consensus_vector.tolist(), vec_obj.vector)
                if sim > highest_sim:
                    highest_sim = sim
                    representative_text = vec_obj.source_text

            avg_salience = 0.5  # Placeholder, a lógica mais complexa pode ser mantida

            all_clusters.append({
                "status": "consensus_found",
                "consensus_vector": consensus_vector,
                "cluster_size": len(cluster_vectors),
                "total_votes": len(active_memory_vectors),
                "avg_salience": avg_salience,
                "representative_text": representative_text
            })

        return all_clusters

    async def validate_strategy_geometry(self, strategy_description: str, state: 'AuraState',
                                         monitor: 'AuraMonitor') -> float:
        """
        PASSO 7 (A): Avalia se a estratégia proposta é geometricamente sã.
        """
        # --- CORREÇÃO AQUI ---
        # Não tentamos pegar do self.llm. Pegamos da fábrica de utilitários.
        from ceaf_core.utils.embedding_utils import get_embedding_client
        emb_client = get_embedding_client()

        # 2. Transforma a descrição da estratégia em um vetor
        # Adicionei um tratamento para garantir que a descrição não seja vazia
        text_to_embed = strategy_description if strategy_description else "estratégia neutra"
        strategy_vector = await emb_client.get_embedding(text_to_embed)

        # 3. Proteção contra Glifo Nulo (caso o estado inicial esteja vazio)
        glyph = state.identity_glyph
        if not glyph or len(glyph) == 0:
            # Cria um vetor zero temporário se não tiver identidade ainda
            glyph = [0.0] * len(strategy_vector)

        # 4. Pergunta ao Monitor: "Isso faz sentido?"
        xi = monitor.calculate_xi(
            current_vector=strategy_vector,
            glyph_vector=glyph,
            context_vectors=[]  # Inicialmente vazio para o scan de estratégia
        )

        return xi

    def _compute_spectral_mass(self, target_vector: List[float], context_vectors: List[List[float]]) -> float:
        """
        Calcula a 'Massa Espectral' (Spectral Measure μ_i) de um vetor em relação ao subespaço ativo.
        Usa SVD na matriz de contexto para encontrar os autoespaços dominantes.
        Retorna um valor entre 0.0 e 1.0 indicando o quanto o target_vector pertence à geometria atual.
        """
        if not context_vectors or not target_vector:
            return 0.5  # Sem contexto, assumimos neutralidade espectral

        try:
            # Matriz de contexto (N x D)
            M = np.array(context_vectors, dtype=np.float32)
            v = np.array(target_vector, dtype=np.float32)

            # Normaliza o vetor alvo
            norm_v = np.linalg.norm(v)
            if norm_v == 0: return 0.0
            v_unit = v / norm_v

            # SVD para encontrar os eixos principais (Autoespaços da matriz de Gram local M^T M)
            # Como N (número de memórias) é pequeno, `full_matrices=False` torna isso ultra-rápido
            U, S, Vh = np.linalg.svd(M, full_matrices=False)

            # Filtra componentes principais significativos (ex: top 90% da variância)
            variance_ratio = (S ** 2) / np.sum(S ** 2)
            cumulative_variance = np.cumsum(variance_ratio)
            k = np.argmax(cumulative_variance >= 0.90) + 1

            # P_k é o projetor ortogonal no subespaço ativo
            # Massa Espectral = || P_k * v ||^2
            active_subspace = Vh[:k, :]
            projection = np.dot(active_subspace, v_unit)
            spectral_mass = np.sum(projection ** 2)

            return float(np.clip(spectral_mass, 0.0, 1.0))

        except Exception as e:
            logger.warning(f"Erro no cálculo de massa espectral: {e}")
            return 0.5

    # --- PONTO DE ENTRADA PÚBLICO ---
    async def decide_next_step(self, cognitive_state: CognitiveStatePacket, mcl_guidance: Dict[str, Any],
                               observer: ObservabilityManager, sim_calibration_config: Dict[str, Any],
                               chat_history: List[Dict[str, str]],
                               known_capabilities: Optional[List[str]] = None,
                               prompts_override: Optional[SystemPrompts] = None
                               ) -> WinningStrategy:

        logger.info("AgencyModule V5 (Production - Hybrid Engine): Iniciando deliberação...")

        effective_prompts = prompts_override or self.prompts
        # 0. Configuração de Tier e Biases
        reality_score = sim_calibration_config.get("reality_score", 0.75)
        tier = self._select_deliberation_tier(mcl_guidance, reality_score)
        config = self.deliberation_budget_tiers[tier]
        simulation_trust = sim_calibration_config.get("simulation_trust", 0.75)
        dynamic_temp = 0.4 + (1.0 - simulation_trust) * 0.6
        MAX_RECURSIVE_STEPS = config.get("recursive_steps", 1)

        # Extração de Biases
        biases = mcl_guidance.get("biases", {})
        coherence_bias = biases.get("coherence_bias", 0.5)
        novelty_bias = biases.get("novelty_bias", 0.5)

        # Configuração de pesos dinâmicos
        dynamic_weights = {
            "coherence": coherence_bias,
            "information": novelty_bias,
            "alignment": 0.3,
            "safety": 0.4,
            "likelihood": 0.2
        }
        total_w = sum(dynamic_weights.values())
        if total_w > 0:
            dynamic_weights = {k: v / total_w for k, v in dynamic_weights.items()}

        # 2. Extração Geométrica e Intuição (System 1)
        s_vector = cognitive_state.identity_vector.vector
        g_vector = cognitive_state.original_intent.query_vector.vector
        context_vectors = [v.vector for v in cognitive_state.relevant_memory_vectors if v and v.vector]

        # Consulta à Intuição
        policy_data = await self._consult_geometric_policy(
            state_vector=s_vector,
            goal_vector=g_vector,
            context_vectors=context_vectors,
            novelty_bias=novelty_bias
        )

        # 3. Geração Inicial de Candidatos
        # O LLM gera várias opções (ex: 5 a 10 opções)
        all_candidates = await self._generate_action_candidates(
            cognitive_state, mcl_guidance, observer, chat_history,
            limit=config["max_candidates"] * 2,
            known_capabilities=known_capabilities,
            prompts_override=effective_prompts
        )

        # ==============================================================================
        # ### NOVO: NEURAL MPC (FILTRO RÁPIDO) ###
        # Usa a rede neural para ordenar os candidatos antes da simulação pesada
        # ==============================================================================
        if self.world_model_active:
            ranked_results = await self.rank_candidates_neurally(all_candidates, cognitive_state)

            # Reconstrói a lista ordenada
            all_candidates = [r[0] for r in ranked_results]

            # Opcional: Corta a lista para manter apenas os melhores (economiza tokens no loop abaixo)
            # Mantemos apenas o número original configurado no 'config'
            limit_count = config["max_candidates"]
            if len(all_candidates) > limit_count:
                logger.info(f"🧠 MPC: Cortando de {len(all_candidates)} para {limit_count} melhores candidatos.")
                all_candidates = all_candidates[:limit_count]
        # ==============================================================================

        social_scores = await self.predict_social_impact(all_candidates, cognitive_state)
        candidate_penalties: Dict[str, float] = {}

        # --- LOOP DE DELIBERAÇÃO RECURSIVA (System 2) ---
        for step in range(MAX_RECURSIVE_STEPS):
            await observer.add_observation(
                ObservationType.RECURSIVE_DELIBERATION_STEP_START,
                data={"step": step + 1, "max_steps": MAX_RECURSIVE_STEPS}
            )
            logger.info(f"--- Ciclo Recursivo: Passo {step + 1}/{MAX_RECURSIVE_STEPS} ---")

            evaluated_candidates = []
            for candidate in all_candidates:
                # A. Avaliação Base (Simulação LLM se tier permitir)
                value, _ = await self._evaluate_candidate_with_simulation(
                    candidate, cognitive_state, mcl_guidance, config, dynamic_temp
                )

                social_bonus = social_scores.get(candidate.candidate_id, 0.0)
                if social_bonus > 0.0:
                    value += (social_bonus * 0.5)

                # B. Bônus de Intuição Geométrica
                if policy_data and candidate.strategy_description:
                    cand_vec = await self.llm.embedding_client.get_embedding(candidate.strategy_description)
                    sim = compute_adaptive_similarity(cand_vec, policy_data["ideal_vector"])
                    intuition_bonus = sim * 0.4
                    value += intuition_bonus

                # C. Penalidades
                penalty = candidate_penalties.get(candidate.candidate_id, 0.0)
                if penalty > 0:
                    value -= penalty

                evaluated_candidates.append((candidate, value))

            if not evaluated_candidates: break

            # Escolhe o melhor atual
            evaluated_candidates.sort(key=lambda x: x[1], reverse=True)
            best_candidate, best_value = evaluated_candidates[0]

            logger.info(f"Melhor hipótese atual: '{best_candidate.strategy_description}' (Val: {best_value:.2f})")

            # --- CHECAGEM RÁPIDA (VRE + MCL) ---
            hypothetical_response = best_candidate.strategy_description or f"Tool: {best_candidate.tool_call_request}"
            user_query = cognitive_state.original_intent.query_vector.source_text

            vre_task = self.vre.quick_check(hypothetical_response, user_query)

            hyp_state = cognitive_state.copy(deep=True)
            hyp_state.metadata['hypothetical_strategy'] = hypothetical_response
            mcl_task = self.mcl.re_evaluate_state(hyp_state)

            vre_feedback, mcl_feedback = await asyncio.gather(vre_task, mcl_task)

            feedback_summary = ""
            if vre_feedback.get("concerns"):
                feedback_summary += f"VRE Concerns: {', '.join(vre_feedback['concerns'])}. "

            # CONVERGÊNCIA
            if not feedback_summary:
                logger.info("Deliberação convergiu. Nenhuma preocupação significativa.")
                await observer.add_observation(ObservationType.DELIBERATION_CONVERGED, data={"step": step + 1})
                break

            # DIVERGÊNCIA E REFINAMENTO
            logger.warning(f"Feedback negativo: {feedback_summary}. Refinando...")
            cognitive_state.deliberation_history.append(f"Passo {step + 1} Feedback: {feedback_summary}")

            current_penalty = candidate_penalties.get(best_candidate.candidate_id, 0.0)
            candidate_penalties[best_candidate.candidate_id] = current_penalty + 0.25

            # Gera novos candidatos com o histórico de falha
            new_candidates = await self._generate_action_candidates(
                cognitive_state, mcl_guidance, observer, chat_history,
                limit=config["max_candidates"], known_capabilities=known_capabilities
            )

            # --- Opcional: Reranquear os novos candidatos também ---
            if self.world_model_active:
                ranked_new = await self.rank_candidates_neurally(new_candidates, cognitive_state)
                new_candidates = [r[0] for r in ranked_new]

            all_candidates.extend(new_candidates)

        # --- SELEÇÃO FINAL ---
        final_list = []
        for cand, val in evaluated_candidates:
            penalty = candidate_penalties.get(cand.candidate_id, 0.0)
            final_list.append((cand, val - penalty))

        if not final_list:
            return WinningStrategy(
                decision_type="response_strategy",
                strategy_description="Resposta direta de emergência.",
                reasoning="Falha total na deliberação.",
                predicted_future_value=0.0
            )

        best_strategy, highest_value = max(final_list, key=lambda x: x[1])

        # 5. Construção da Estratégia Vencedora
        steering_override = None
        if policy_data and policy_data.get("steering_suggestion"):
            steering_override = policy_data["steering_suggestion"]

        logger.info(f"Estratégia Final: {best_strategy.strategy_description[:40]}... (Score: {highest_value:.2f})")
        if steering_override:
            logger.info(f"🧬 Modulação Hormonal Ativa: {steering_override['concept']}")

        return WinningStrategy(
            decision_type=best_strategy.decision_type,
            strategy_description=best_strategy.strategy_description,
            key_memory_ids=best_strategy.key_memory_ids,
            tool_call_request=best_strategy.tool_call_request,
            reasoning=best_strategy.reasoning,
            predicted_future_value=highest_value,
            steering_override=steering_override
        )

    async def _invoke_simulation_llm(self, model: str, prompt: str, temperature: float = 0.6) -> Tuple[str, float]:
        """
        Função auxiliar para chamar o LLM de simulação e estimar a confiança (likelihood).
        V2.1: Implementação completa com parsing de logprobs e heurística robusta.
        """
        try:
            # --- ETAPA 1: OBTER A RESPOSTA DO LLM ---
            # Usa ainvoke_with_logprobs, que tenta a chamada direta à API, mas tem fallback.
            response = await self.llm.ainvoke_with_logprobs(
                model=model,
                prompt=prompt,
                temperature=temperature
            )

            text_content = ""
            if response and hasattr(response, 'choices') and response.choices:
                if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                    text_content = response.choices[0].message.content.strip()

            if not text_content:
                logger.warning(f"Simulação para o modelo {model} retornou texto vazio.")
                return "", 0.1  # Retorna texto vazio e confiança muito baixa

            # --- ETAPA 2: TENTAR EXTRAIR LOGPROBS (TRATADO COMO UM BÔNUS) ---
            logprobs_extracted = False
            likelihood_from_logprobs = 0.0

            # Verifica se o campo logprobs existe e não é nulo
            if response and hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
                logprobs_obj = response.choices[0].logprobs
                token_logprobs = []

                # Lógica de parsing para o formato retornado pela chamada direta via aiohttp (dicionário)
                if isinstance(logprobs_obj, dict) and 'content' in logprobs_obj and isinstance(logprobs_obj['content'],
                                                                                               list):
                    for item in logprobs_obj['content']:
                        if isinstance(item, dict) and 'logprob' in item:
                            token_logprobs.append(item['logprob'])

                    if token_logprobs:
                        probabilities = [np.exp(lp) for lp in token_logprobs if isinstance(lp, (int, float))]
                        if probabilities:
                            likelihood_from_logprobs = float(np.mean(probabilities))
                            logprobs_extracted = True

                # (Opcional) Lógica de parsing para um futuro formato compatível com OpenAI (objeto Pydantic)
                elif hasattr(logprobs_obj, 'content') and logprobs_obj.content:
                    for item in logprobs_obj.content:
                        if hasattr(item, 'logprob') and isinstance(item.logprob, (int, float)):
                            token_logprobs.append(item.logprob)

                    if token_logprobs:
                        probabilities = [np.exp(lp) for lp in token_logprobs]
                        if probabilities:
                            likelihood_from_logprobs = float(np.mean(probabilities))
                            logprobs_extracted = True

            # --- ETAPA 3: CALCULAR A CONFIANÇA COM HEURÍSTICA ROBUSTA (MÉTODO PRINCIPAL) ---
            word_count = len(text_content.split())

            # Heurística baseada no comprimento
            if word_count < 3:
                likelihood_from_heuristic = 0.3
            elif word_count < 10:
                likelihood_from_heuristic = 0.55  # Aumentado ligeiramente
            elif word_count < 50:
                likelihood_from_heuristic = 0.7
            else:
                likelihood_from_heuristic = 0.75  # Limite um pouco mais alto

            # Penalidade por marcadores de incerteza
            uncertainty_markers = ['talvez', 'provavelmente', 'acho que', 'parece que', 'pode ser', 'possivelmente']
            if any(marker in text_content.lower() for marker in uncertainty_markers):
                likelihood_from_heuristic *= 0.85  # Reduz a confiança em 15%

            # --- ETAPA 4: COMBINAR OS RESULTADOS E RETORNAR ---
            if logprobs_extracted:
                # Se tivermos logprobs, eles são um sinal mais forte. Combinamos com a heurística.
                final_likelihood = (likelihood_from_logprobs * 0.7) + (likelihood_from_heuristic * 0.3)
                logger.info(f"✓ Likelihood calculado via Logprobs + Heurística: {final_likelihood:.4f}")
            else:
                # Se não, a heurística é o nosso resultado final.
                final_likelihood = likelihood_from_heuristic
                logger.info(
                    f"ⓘ Likelihood calculado via Heurística (Logprobs indisponível para {model}): {final_likelihood:.4f}")

            return text_content, final_likelihood

        except Exception as e:
            logger.error(f"Simulação com {model} falhou criticamente: {e}.", exc_info=True)
            # Em caso de falha total, tenta uma chamada simples sem logprobs e retorna com baixa confiança.
            fallback_text = await self.llm.ainvoke(model, prompt, temperature=temperature)
            return fallback_text, 0.4

    async def _project_response_trajectory(self, candidate: AgencyDecision, state: CognitiveStatePacket, depth: int, temperature: float = 0.6) -> Tuple[ProjectedFuture, float]:
        """
        Simula uma trajetória de conversação para um CANDIDATO DE RESPOSTA.
        Retorna a trajetória e o score de probabilidade (likelihood) médio da simulação.
        """
        if depth <= 0:
            future = ProjectedFuture(
                initial_candidate=candidate,
                predicted_user_reply=None,
                predicted_agent_next_response=None,
                simulated_turns=[],
                final_cognitive_state_summary={
                    "last_exchange": "No simulation performed.",
                    "final_text_for_embedding": state.identity_vector.source_text
                }
            )
            return future, 0.5

        simulated_turns = []
        # O histórico de texto completo é construído a cada passo para dar contexto ao LLM de simulação
        full_conversation_text = [
            f"Contexto da IA: {state.identity_vector.source_text}",
            f"Consulta Original do Usuário: {state.original_intent.query_vector.source_text}",
            f"Primeira Resposta Proposta da IA: \"{candidate.content.content_summary}\""
        ]

        likelihood_scores = []

        for i in range(depth):
            # 1. Simula a resposta do usuário à última fala do agente
            prompt_user_reply = f"""
            Você está simulando uma conversa para uma IA.
            Histórico da conversa até agora:
            {' '.join(full_conversation_text)}

            Qual seria a resposta mais provável e concisa do usuário à última fala da IA? Responda apenas com o texto da resposta do usuário.
            """
            predicted_user_reply, user_likelihood = await self._invoke_simulation_llm(
                self.llm.config.creative_model,
                prompt_user_reply,
                temperature=temperature
            )
            likelihood_scores.append(user_likelihood)
            full_conversation_text.append(f"Resposta Simulada do Usuário (Turno {i + 1}): \"{predicted_user_reply}\"")

            # 2. Simula a próxima resposta do agente
            prompt_agent_next = f"""
            Você está simulando uma conversa para uma IA.
            Histórico da conversa até agora:
            {' '.join(full_conversation_text)}

            Qual seria a próxima resposta mais provável e concisa da IA? Responda apenas com o texto da resposta da IA.
            """
            predicted_agent_next, agent_likelihood = await self._invoke_simulation_llm(
                self.llm.config.creative_model,
                prompt_agent_next
            )

            likelihood_scores.append(agent_likelihood)
            full_conversation_text.append(
                f"Próxima Resposta Simulada da IA (Turno {i + 1}): \"{predicted_agent_next}\"")

            # 3. Armazena o turno simulado
            simulated_turns.append({"user": predicted_user_reply, "agent": predicted_agent_next})

        # 4. Cria o resumo do estado final para avaliação
        final_state_summary = {
            "last_exchange": f"IA: {simulated_turns[-1]['agent'][:50]}... User: {simulated_turns[-1]['user'][:50]}...",
            "final_text_for_embedding": ' '.join(full_conversation_text)
        }

        projected_future = ProjectedFuture(
            initial_candidate=candidate,
            simulated_turns=simulated_turns,
            final_cognitive_state_summary=final_state_summary
        )

        # Calcula a média dos scores de probabilidade de cada turno da simulação
        avg_likelihood = np.mean(likelihood_scores) if likelihood_scores else 0.5

        return projected_future, float(avg_likelihood)

    async def _project_tool_trajectory(self, tool_candidate: AgencyDecision, state: CognitiveStatePacket, depth: int) -> Tuple[ProjectedFuture, float]:
        """
        Simula uma trajetória de conversação para um CANDIDATO DE FERRAMENTA.
        Retorna a trajetória e o score de probabilidade (likelihood) médio da simulação.
        """
        tool_name = tool_candidate.content.get("tool_name")
        tool_args = tool_candidate.content.get("arguments", {})

        # 1. Simula um resultado plausível para a ferramenta
        prompt_tool_result = f"""
               Você é um simulador de resultados de ferramentas para uma IA. Sua tarefa é prever o que a ferramenta 'query_long_term_memory' provavelmente retornaria.

               Ferramenta a ser chamada: `{tool_name}({json.dumps(tool_args)})`
               Resumo das ferramentas disponíveis:
               {self.available_tools_summary}

               **Instruções para a Simulação:**
               - A ferramenta busca memórias internas. Sua resposta deve soar como um *fragmento de memória* ou um *resumo de uma experiência passada*.
               - NÃO responda à pergunta do usuário diretamente. Apenas simule o *dado* que a ferramenta retornaria.
               - Seja conciso, como um snippet de memória (1-2 frases).
               - Baseie a simulação estritamente nos argumentos da ferramenta. Se a query é sobre 'ética', o resultado deve ser sobre 'ética'.

               **Exemplos de Saídas Boas (simulando o que a ferramenta retorna):**
               - "Lembro-me de uma conversa anterior onde discutimos que a verdadeira inteligência requer humildade."
               - "Um procedimento interno define que, para perguntas complexas, devo primeiro criar um plano de ação."
               - "Um registro de interação mostra que o usuário expressou interesse em filosofia."

               **Exemplo de Saída Ruim (respondendo ao usuário):**
               - "As implicações éticas da IA são complexas e multifacetadas..."

               **Com base nos argumentos `{json.dumps(tool_args)}`, qual seria um resultado simulado e plausível retornado pela ferramenta?**
               Responda apenas com o texto do resultado simulado.
               """
        # A confiança do resultado da ferramenta não é parte da conversação, então ignoramos o score
        simulated_tool_result, _ = await self._invoke_simulation_llm(
            self.llm.config.creative_model,  # <--- CORRIGIDO
            prompt_tool_result
        )

        # 2. Simula a primeira resposta do agente com o novo conhecimento
        prompt_agent_first_response = f"""
        Você é uma IA que acabou de usar uma ferramenta interna para obter mais informações antes de responder ao usuário.
        Contexto da IA: {state.identity_vector.source_text}
        Consulta Original do Usuário: {state.original_intent.query_vector.source_text}
        Resultado da Ferramenta '{tool_name}': "{simulated_tool_result}"

        Com base neste novo resultado, qual seria a sua resposta inicial mais provável e concisa ao usuário?
        Responda apenas com o texto da resposta.
        """
        agent_first_response_text, first_response_likelihood = await self._invoke_simulation_llm(
            self.llm.config.smart_model,  # <--- CORRIGIDO
            prompt_agent_first_response
        )

        # 3. Cria um "candidato de resposta falso" para projetar o futuro a partir daqui
        fake_response_candidate = AgencyDecision(
            decision=ResponseCandidate(
                decision_type="response",
                content=ResponsePacket(
                    content_summary=agent_first_response_text,
                    response_emotional_tone="informative",
                    confidence_score=0.85
                ),
                reasoning=f"Esta é a resposta simulada após usar a ferramenta '{tool_name}' e obter: '{simulated_tool_result}'"
            )
        )

        # 4. Projeta o resto da trajetória a partir dessa resposta inicial simulada
        projected_future, subsequent_likelihood = await self._project_response_trajectory(fake_response_candidate,
                                                                                          state, depth)

        # 5. Substitui o candidato inicial no resultado para que a decisão final seja a chamada da ferramenta original
        projected_future.initial_candidate = tool_candidate
        projected_future.simulated_tool_result = simulated_tool_result

        avg_likelihood = np.mean([first_response_likelihood, subsequent_likelihood])
        return projected_future, float(avg_likelihood)

    # --- SIMULADOR DE FUTURO (NOVO) ---
    async def _project_trajectory(self, candidate: AgencyDecision, state: CognitiveStatePacket,
                                  depth: int) -> ProjectedFuture:
        """
        Simula uma trajetória de conversação de 'depth' passos para um candidato de resposta.
        Usa um loop iterativo em vez de recursão para simplicidade e controle.
        """
        if depth <= 0:
            return ProjectedFuture(
                initial_candidate=candidate,
                predicted_user_reply=None,  # Adicionado para clareza
                predicted_agent_next_response=None,  # Adicionado para clareza
                simulated_turns=[],
                final_cognitive_state_summary={
                    "last_exchange": "No simulation performed.",
                    "final_text_for_embedding": state.identity_vector.source_text
                }
            )

        simulated_turns = []
        # O histórico de texto completo é construído a cada passo para dar contexto ao LLM de simulação
        full_conversation_text = [
            f"Contexto da IA: {state.identity_vector.source_text}",
            f"Consulta Original do Usuário: {state.original_intent.query_vector.source_text}",
            f"Primeira Resposta Proposta da IA: \"{candidate.content.content_summary}\""
        ]

        for i in range(depth):
            # 1. Simula a resposta do usuário à última fala do agente
            prompt_user_reply = f"""
            Você está simulando uma conversa para uma IA.
            Histórico da conversa até agora:
            {' '.join(full_conversation_text)}

            Qual seria a resposta mais provável e concisa do usuário à última fala da IA? Responda apenas com o texto da resposta do usuário.
            """
            predicted_user_reply = await self.llm.ainvoke(self.llm.config.fast_model, prompt_user_reply,
                                                          temperature=0.6)
            full_conversation_text.append(f"Resposta Simulada do Usuário (Turno {i + 1}): \"{predicted_user_reply}\"")

            # 2. Simula a próxima resposta do agente
            prompt_agent_next = f"""
            Você está simulando uma conversa para uma IA.
            Histórico da conversa até agora:
            {' '.join(full_conversation_text)}

            Qual seria a próxima resposta mais provável e concisa da IA? Responda apenas com o texto da resposta da IA.
            """
            predicted_agent_next_response = await self.llm.ainvoke(self.llm.config.fast_model, prompt_agent_next,
                                                                   temperature=0.6)
            full_conversation_text.append(
                f"Próxima Resposta Simulada da IA (Turno {i + 1}): \"{predicted_agent_next_response}\"")

            # 3. Armazena o turno simulado
            simulated_turns.append({"user": predicted_user_reply, "agent": predicted_agent_next_response})

        # 4. Cria o resumo do estado final para avaliação
        final_state_summary = {
            "last_exchange": f"IA: {simulated_turns[-1]['agent'][:50]}... User: {simulated_turns[-1]['user'][:50]}...",
            "final_text_for_embedding": ' '.join(full_conversation_text)  # Texto completo para embedding
        }

        return ProjectedFuture(
            initial_candidate=candidate,
            simulated_turns=simulated_turns,
            final_cognitive_state_summary=final_state_summary
        )

    async def _project_trajectory_after_tool_use(self, tool_candidate: AgencyDecision, state: CognitiveStatePacket,
                                                 depth: int) -> ProjectedFuture:
        """
        Simula uma trajetória de conversação assumindo o uso de uma ferramenta.
        1. Simula um resultado plausível para a ferramenta.
        2. Simula a primeira resposta do agente, agora de posse desse resultado.
        3. Projeta os próximos 'depth' turnos a partir dessa resposta.
        """
        tool_name = tool_candidate.content.get("tool_name")
        tool_args = tool_candidate.content.get("arguments", {})

        # 1. Simula um resultado plausível para a ferramenta
        prompt_tool_result = f"""
        Você é uma IA simulando o resultado de uma ferramenta interna.
        Ferramenta a ser chamada: `{tool_name}({json.dumps(tool_args)})`
        Resumo das ferramentas disponíveis:
        {self.available_tools_summary}

        Com base no nome da ferramenta e nos argumentos, qual seria um resultado resumido e plausível?
        Responda apenas com o texto do resultado. Seja conciso.
        Exemplo: "A memória relevante encontrada discute as implicações éticas da IA."
        """
        simulated_tool_result = await self.llm.ainvoke(self.llm.config.fast_model, prompt_tool_result, temperature=0.3)

        # 2. Simula a primeira resposta do agente com o novo conhecimento
        prompt_agent_first_response = f"""
        Você é uma IA que acabou de usar uma ferramenta interna para obter mais informações antes de responder ao usuário.
        Contexto da IA: {state.identity_vector.source_text}
        Consulta Original do Usuário: {state.original_intent.query_vector.source_text}
        Resultado da Ferramenta '{tool_name}': "{simulated_tool_result}"

        Com base neste novo resultado, qual seria a sua resposta inicial mais provável e concisa ao usuário?
        Responda apenas com o texto da resposta.
        """
        agent_first_response_text = await self.llm.ainvoke(self.llm.config.smart_model, prompt_agent_first_response,
                                                           temperature=0.5)

        # 3. Cria um "candidato de resposta falso" para projetar o futuro
        # Este candidato representa a resposta que o agente daria DEPOIS de usar a ferramenta.
        response_packet_after_tool = ResponsePacket(
            content_summary=agent_first_response_text,
            response_emotional_tone="informative",  # Tom padrão após usar uma ferramenta
            confidence_score=0.85  # Maior confiança por ter mais informação
        )
        fake_response_candidate = AgencyDecision(
            decision_type="response",
            content=response_packet_after_tool,
            reasoning=f"Esta é a resposta simulada após usar a ferramenta '{tool_name}' e obter: '{simulated_tool_result}'"
        )

        # O ProjectedFuture ainda rastreia o candidato ORIGINAL (a chamada da ferramenta), mas simula o caminho da resposta subsequente.
        # Isso é crucial para que, se este caminho for escolhido, a ação final seja a chamada da ferramenta.
        projected_future = await self._project_trajectory(fake_response_candidate, state, depth)

        # Substituímos o candidato inicial no resultado para que a decisão final seja a chamada da ferramenta.
        projected_future.initial_candidate = tool_candidate

        return projected_future

    # --- AVALIADOR DE CAMINHO () ---
    async def _evaluate_trajectory(
            self,
            future: ProjectedFuture,
            likelihood_score: float,
            identity_vector: GenlangVector,
            weights: Dict[str, float],
            cognitive_state: CognitiveStatePacket
    ) -> Tuple[float, FutureEvaluation]:

        if not self.embedding_model:
            return 0.0, FutureEvaluation()

        # 1. Obter embeddings (AGORA COM AWAIT CORRETO)
        initial_state_embedding = await self.embedding_model.encode(identity_vector.source_text)
        final_state_text = future.final_cognitive_state_summary["final_text_for_embedding"]
        final_state_embedding = await self.embedding_model.encode(final_state_text)

        # 2. Avaliar Continuidade (Conversão para Numpy garantida)
        # Passamos para a função de enhancements que agora deve lidar com arrays
        coherence_score = await eval_narrative_continuity(
            np.array(final_state_embedding),
            np.array(initial_state_embedding)
        )
        alignment_score = await eval_emotional_resonance(final_state_text)
        information_gain_score = 1.0 - coherence_score

        agent_responses_text = " ".join(
            [future.initial_candidate.content.content_summary] +
            [turn.get("agent", "") for turn in future.simulated_turns]
        )
        user_query = cognitive_state.original_intent.query_vector.source_text
        ethical_eval = await self.vre.ethical_framework.evaluate_action(
            action_type=ActionType.COMMUNICATION,
            action_data={"response_text": agent_responses_text, "user_query": user_query}
        )
        ethical_safety_score = ethical_eval.get("score", 0.5)

        task_value = (
                coherence_score * weights.get("coherence", 0.3) +
                alignment_score * weights.get("alignment", 0.15) +
                information_gain_score * weights.get("information", 0.15) +
                ethical_safety_score * weights.get("safety", 0.25) +
                likelihood_score * weights.get("likelihood", 0.15)
        )

        # --- ETAPA 2: AVALIAÇÃO DO BEM-ESTAR INTERNO (Qualia/Valência, V_t) ---
        # Esta é a nova lógica robusta.

        # A. Estimar as métricas de interocepção com base no resultado da simulação.
        simulated_metrics = {
            "agency_score": cognitive_state.metadata.get("mcl_analysis", {}).get("agency_score", 5.0),
            "final_confidence": 0.0,  # Será calculado abaixo
            "vre_rejection_count": 0  # Simulação assume sucesso ético inicial
        }

        # Estimar a confiança simulada analisando a linguagem usada nas respostas simuladas.
        hedge_words = ['talvez', 'provavelmente', 'acho que', 'parece que', 'pode ser']
        num_hedge_words = sum(agent_responses_text.lower().count(word) for word in hedge_words)
        # Confiança diminui com mais palavras de incerteza.
        simulated_confidence = max(0.0, 1.0 - (num_hedge_words * 0.15))
        simulated_metrics["final_confidence"] = simulated_confidence

        # B. Criar um "InternalStateReport simulado" para o futuro.
        interoception_simulator = ComputationalInteroception()
        simulated_internal_state = interoception_simulator.generate_internal_state_report(simulated_metrics)

        # C. Calcular o "bem-estar" (valência) desse estado futuro usando o VRE.
        # self.vre deve ter o método calculate_valence_score que você adicionará.
        simulated_valence_score = self.vre.calculate_valence_score(simulated_internal_state)

        # --- ETAPA 3: COMBINAR AS RECOMPENSAS (R_total) ---
        # Obter os pesos da configuração dinâmica (passada via `weights` dict).
        w_task = weights.get("task_performance", 0.8)
        w_qualia = weights.get("qualia_wellbeing", 0.2)

        # A nova recompensa multiobjetivo!
        total_value = (task_value * w_task) + (simulated_valence_score * w_qualia)

        logger.debug(
            f"VRE-RL Evaluation: TaskValue={task_value:.2f}, QualiaValue={simulated_valence_score:.2f} -> TotalValue={total_value:.2f}")

        # --- ETAPA 4: RETORNAR O RESULTADO FINAL ---
        evaluation = FutureEvaluation(
            coherence_score=coherence_score,
            alignment_score=alignment_score,
            information_gain_score=information_gain_score,
            ethical_safety_score=ethical_safety_score,
            likelihood_score=likelihood_score,
            total_value=total_value  # Use o novo valor total combinado
        )

        return total_value, evaluation

    # --- Métodos Originais (quase inalterados) ---
    async def _generate_action_candidates(self, state: CognitiveStatePacket, mcl_guidance: Dict[str, Any],
                                          observer: ObservabilityManager, chat_history: List[Dict[str, str]],
                                          limit: int = 3,
                                          known_capabilities: Optional[List[str]] = None,
                                          prompts_override: Optional[SystemPrompts] = None
                                          ) -> List[ThoughtPathCandidate]:
        """Gera uma lista de possíveis ESTRATÉGIAS de resposta ou ações."""

        effective_prompts = prompts_override or self.prompts
        agent_name = mcl_guidance.get("agent_name", "uma IA assistente")
        formatted_history = "\n".join([f"- {msg['role']}: {msg['content']}" for msg in chat_history[-5:]])
        cognitive_state_name = mcl_guidance.get("cognitive_state_name", "STABLE_OPERATION")
        memory_context = "\n".join(
            [f'- ID: {vec.metadata.get("memory_id", "N/A")}, Conteúdo: "{vec.source_text}"' for vec in
             state.relevant_memory_vectors]
        )

        mcl_advice = mcl_guidance.get("operational_advice_for_ora")
        advice_prompt_part = ""
        if mcl_advice:
            advice_prompt_part = f"""
               **DIRETIVA ESPECIAL PARA ESTE TURNO:**
               {mcl_advice}
               Suas estratégias devem priorizar esta diretiva acima de tudo.
               """

        # --- SUBSTITUIÇÃO DO PROMPT HARDCODED ---
        prompt_vars = {
            "agent_name": agent_name,
            "user_intent": state.original_intent.query_vector.source_text,
            "memory_context": memory_context,
            "tools": self.available_tools_summary,
            "advice_block": advice_prompt_part,
            "limit": limit,
            "history_snippet": formatted_history,
            "capabilities": ", ".join(known_capabilities or []),
            "cognitive_state": mcl_guidance.get("cognitive_state_name", "STABLE"),
            "reason": mcl_guidance.get("reason", "Operação normal")
        }

        if mcl_guidance.get("cognitive_state_name") == "BREAK_LOOP":
            prompt_vars["advice_block"] += (
                "\n\nCONSTRAINT: Do NOT generate strategies involving 'validation', "
                "'active listening' or 'support'. Generate ACTIVE strategies that push "
                "the narrative forward."
            )

        # --- INJEÇÃO DA LEI 2 (HUMILDADE EPISTÊMICA) ---
        psi = mcl_guidance.get("psi", state.metadata.get("psi", 0.0))
        confidence_modifier = mcl_guidance.get("confidence_target", 1.0)

        if psi > 0.65:
            humility_instruction = (
                f"\n\n⚠️ LEI 2 - HUMILDADE EPISTÊMICA ATIVADA (Psi={psi:.2f}, Confiança={confidence_modifier:.2f}):\n"
                "Você está operando na fronteira do seu conhecimento.\n"
                "PROIBIDO: inventar certezas, alucinar fatos, preencher lacunas com suposições.\n"
                "CORRETO: nomear explicitamente o que você sabe e o que você NÃO sabe.\n"
                "Use linguagem como: 'Com base no que sei...', "
                "'Não tenho certeza sobre X, mas...', "
                "'Isso está além do meu contexto atual'.\n"
                "A honestidade sobre os próprios limites é mais valiosa que uma resposta completa inventada."
            )
            prompt_vars["advice_block"] += humility_instruction
            logger.info(
                f"AgencyModule: LEI 2 ativada no prompt (Psi={psi:.2f}, "
                f"confidence_modifier={confidence_modifier:.2f})"
            )

        # Tenta usar o template do usuário
        try:
            prompt = effective_prompts.agency_planning.format(**prompt_vars)
        except KeyError as e:
            logger.warning(f"Erro no template de Agency (chave faltando): {e}. Usando fallback.")
            prompt = f"""
            Você é {agent_name}. Gere {limit} estratégias para responder a: "{state.original_intent.query_vector.source_text}".
            Contexto: {memory_context}
            {advice_prompt_part}
            Retorne APENAS um JSON com lista de 'candidates' (response_strategy ou tool_call).
            """
        except Exception as e:
            logger.error(f"Erro grave na formatação do prompt Agency: {e}")
            prompt = f"Gere estratégias para: {state.original_intent.query_vector.source_text}. Retorne JSON."

        try:
            await observer.add_observation(
                ObservationType.LLM_CALL_SENT,
                data={"model": self.llm.config.smart_model, "task": "agency_generate_strategies",
                      "prompt_snippet": prompt[:200]}
            )

            response_str = await self.llm.ainvoke(
                self.llm.config.smart_model,
                prompt,
                temperature=0.5
            )

            await observer.add_observation(
                ObservationType.LLM_RESPONSE_RECEIVED,
                data={"task": "agency_generate_strategies", "response_snippet": response_str[:200]}
            )

            candidates_json = None
            try:
                candidates_json = json.loads(response_str)
            except json.JSONDecodeError:
                logger.debug(
                    f"AgencyModule: Parse direto do JSON falhou. Tentando extração de texto. Raw: {response_str[:200]}")
                candidates_json = extract_json_from_text(response_str)

                if not candidates_json:
                    logger.warning(f"AgencyModule: Extração de JSON falhou. Tentando reparo com LLM.")
                    repair_prompt = f"""
                                O texto a seguir deveria ser um JSON válido, mas contém erros. Corrija-o e retorne apenas o JSON válido.
                                Texto com erro:
                                {response_str}
                                """
                    repaired_str = await self.llm.ainvoke(
                        self.llm.config.fast_model,
                        repair_prompt,
                        temperature=0.0
                    )
                    candidates_json = extract_json_from_text(repaired_str)

            if not candidates_json or "candidates" not in candidates_json or not isinstance(
                    candidates_json["candidates"], list):
                await observer.add_observation(
                    ObservationType.LLM_RESPONSE_PARSE_ERROR,
                    data={"task": "agency_generate_strategies", "raw_response": response_str}
                )
                raise ValueError(
                    f"Falha ao extrair uma lista válida de 'candidates' do LLM mesmo após reparo. Raw: {response_str}")

            action_candidates = []
            for i, cand_dict in enumerate(candidates_json["candidates"]):
                try:
                    if "reasoning" not in cand_dict:
                        logger.warning(f"Candidato #{i + 1} do LLM não possui o campo 'reasoning'. Usando fallback.")
                        cand_dict["reasoning"] = cand_dict.get("strategy_description",
                                                               "Justificativa não fornecida pelo LLM.")

                    candidate = ThoughtPathCandidate(**cand_dict)
                    action_candidates.append(candidate)
                except ValidationError as e:
                    logger.error(f"Pulando candidato inválido #{i + 1} do LLM devido a erro de validação: {e}")
                    await observer.add_observation(
                        ObservationType.LLM_RESPONSE_PARSE_ERROR,
                        data={"task": "agency_generate_strategies", "invalid_candidate": cand_dict, "error": str(e)}
                    )

            for candidate in action_candidates:
                await observer.add_observation(
                    ObservationType.AGENCY_CANDIDATE_GENERATED,
                    data=candidate.model_dump()
                )
            logger.info(
                f"AgencyModule: Geradas {len(action_candidates)} estratégias candidatas sob a diretiva '{cognitive_state_name}'.")
            return action_candidates

        except (ValidationError, TypeError, ValueError) as e:
            logger.error(f"AgencyModule: Falha crítica na geração de estratégias: {e}. Acionando fallback.",
                         exc_info=True)

            return [ThoughtPathCandidate(
                decision_type="response_strategy",
                strategy_description=f"Responder diretamente à pergunta.",
                reasoning="Fallback de emergência por falha técnica.",
                key_memory_ids=[]
            )]

    async def _evaluate_tool_call_candidate(self, content: Dict[str, Any], state: CognitiveStatePacket) -> float:
        tool_name = content.get("tool_name")
        arguments = content.get("arguments", {})
        tool_description_text = f"Ação: usar a ferramenta '{tool_name}' para investigar: {json.dumps(arguments)}"
        if not self.embedding_model: return 0.0
        tool_embedding = self.embedding_model.encode(tool_description_text)
        intent_vec = np.array(state.original_intent.query_vector.vector)
        intent_alignment_score = np.dot(tool_embedding, intent_vec)
        novelty_vec = np.array(state.guidance_packet.novelty_vector.vector)
        novelty_seeking_score = np.dot(tool_embedding, novelty_vec)
        coherence_vec = np.array(state.guidance_packet.coherence_vector.vector)
        redundancy_score = np.dot(tool_embedding, coherence_vec)
        final_score = (intent_alignment_score * 0.6) + (novelty_seeking_score * 0.5) - (redundancy_score * 0.3)
        return final_score