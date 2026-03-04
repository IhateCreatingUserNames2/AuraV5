# ceaf_core/activities.py
import logging
import asyncio
import os
import time
from datetime import datetime
from typing import Dict, Any, List

from ceaf_core.modules.memory_blossom import ExplicitMemory, ExplicitMemoryContent, MemorySourceType, MemorySalience
from ceaf_core.services.cognitive_log_service import CognitiveLogService
from ceaf_core.services.user_profiling_service import UserProfilingService
from ceaf_core.identity_manifold import IdentityManifold
from pathlib import Path
import numpy as np
from temporalio import activity
from ceaf_core.utils.embedding_utils import get_embedding_client
from ceaf_core.services.state_manager import StateManager
# Import Core Logic Modules
from ceaf_core.translators.human_to_genlang import HumanToGenlangTranslator
from ceaf_core.rlm_investigator import RLMInvestigator
from ceaf_core.hormonal_metacontroller import HormonalMetacontroller
from ceaf_core.agency_module import AgencyModule
from ceaf_core.translators.genlang_to_human import GenlangToHumanTranslator
from ceaf_core.birag_validator import BiRAGValidator
from ceaf_core.v4_sensors import AuraMonitor
from ceaf_core.services.mbs_memory_service import MBSMemoryService
from ceaf_core.services.llm_service import LLMService
from ceaf_core.modules.ncim_engine.ncim_module import NCIMModule
from ceaf_core.modules.vre_engine.vre_engine import VREEngineV3
from ceaf_core.modules.mcl_engine.mcl_engine import MCLEngine
from ceaf_core.modules.lcam_module import LCAMModule
from ceaf_core.modules.embodiment_module import EmbodimentModule
from ceaf_core.modules.geometric_brain import GeometricBrain
from ceaf_core.modules.ncim_engine.ncim_module import SELF_MODEL_MEMORY_ID
from agent_manager import AgentManager

# Import Types
from ceaf_core.genlang_types import (
    CognitiveStatePacket, IntentPacket, ResponsePacket,
    GenlangVector, GuidancePacket
)
from ceaf_core.monadic_base import AuraState
from ceaf_core.models import SystemPrompts, MCLConfig, BodyConfig, LLMConfig, CeafSelfRepresentation

logger = logging.getLogger("CEAF_Activities")


# --- Service Initialization Helper ---
# In a production worker, these might be initialized once per worker process or dependency injected.
# For simplicity here, we initialize them lazily or assume singleton behavior where appropriate.

class ActivityContext:
    _instance = None

    def __init__(self):
        self.llm_service = LLMService()
        self.memory_service = MBSMemoryService()
        self.htg = HumanToGenlangTranslator(llm_config=self.llm_service.config)
        self.monitor = AuraMonitor()
        self.rlm = RLMInvestigator(llm_service=self.llm_service)  # V2: LLM injetada para SEMANTIC_FILTER
        self.hormonal = HormonalMetacontroller()
        self.vre = VREEngineV3()
        self.lcam = LCAMModule(self.memory_service)
        self.mcl = MCLEngine(
            config={},
            agent_config={},
            lcam_module=self.lcam,
            llm_service=self.llm_service
        )
        self.agency = AgencyModule(
            self.llm_service,
            self.vre,
            self.mcl,
            available_tools_summary=""
        )
        self.gth = GenlangToHumanTranslator(llm_service=self.llm_service)
        self.birag = BiRAGValidator(self.llm_service)
        self.ncim = NCIMModule(
            self.llm_service,
            self.memory_service,
            persistence_path=None  # Handled via MBS
        )
        self.geometric_brain = GeometricBrain()
        self.state_manager = StateManager()
        self.embodiment = EmbodimentModule()
        self.user_profiler = UserProfilingService(self.memory_service)

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# --- 1. Perception (HTG) ---

@activity.defn
async def perception_activity(state_dict: Dict, query: str) -> Dict:
    ctx = ActivityContext.get()
    agent_id = state_dict["agent_id"]
    user_id = state_dict.get("user_id", "default_user")

    # 0. EXTRAÇÃO DO HISTÓRICO
    # Necessário para entender o contexto de pronomes ("isso", "ele")
    chat_history = state_dict.get("metadata", {}).get("chat_history", [])

    # 1. QUERY CONTEXTUALIZATION (A CORREÇÃO CRÍTICA)
    # Transforma: "Fale mais sobre isso" -> "Fale mais sobre a Arquitetura Aura"
    try:
        # Assume que você implementou o método contextualize_query no HTG conforme combinado
        contextualized_query = await ctx.htg.contextualize_query(query, chat_history)
        logger.info(f"🧠 Perception: Query Original: '{query}' -> Contextualizada: '{contextualized_query}'")
    except Exception as e:
        logger.error(f"⚠️ Erro na contextualização (usando original): {e}")
        contextualized_query = query

    # 2. VECTORIZATION (The Universal Glyph)
    # Geramos o vetor 2048d baseado na frase RICA em contexto, não na pobre.
    dim = os.getenv("GEOMETRIC_DIMENSION")
    vec_2048_list = await ctx.llm_service.embedding_client.get_embedding(contextualized_query)

    # === CORREÇÃO V5: RECUPERAR A ÚLTIMA AÇÃO DA IA PARA O MONITOR ===
    last_ai_content = None
    last_ai_vector = None

    # Busca a última mensagem que seja do papel 'assistant' ou 'model'
    if chat_history:
        for msg in reversed(chat_history):
            if msg.get("role") in ["assistant", "model", "AI"]:
                last_ai_content = msg.get("content")
                break

    # Se achamos a última resposta, vetorizamos ela para comparar
    if last_ai_content:
        try:
            # Usa o mesmo cliente de embedding para garantir espaço vetorial compatível
            last_ai_vector = await ctx.llm_service.embedding_client.get_embedding(last_ai_content)
            logger.info(f"🔍 Última resposta da IA vetorizada para detecção de eco")
        except Exception as e:
            logger.warning(f"Falha ao vetorizar última resposta da IA: {e}")

    # 3. IDENTITY RECOVERY / BOOTSTRAP (Mantido igual)
    from ceaf_core.modules.ncim_engine.ncim_module import SELF_MODEL_MEMORY_ID
    from ceaf_core.models import CeafSelfRepresentation
    from ceaf_core.modules.memory_blossom.memory_types import ExplicitMemory, ExplicitMemoryContent, MemorySourceType, \
        MemorySalience

    unique_self_id = f"{SELF_MODEL_MEMORY_ID}_{agent_id}"
    identity_glyph = state_dict.get("identity_glyph", [])

    if not identity_glyph:
        try:
            self_mem = await ctx.memory_service.get_memory_by_id(unique_self_id)

            if self_mem and hasattr(self_mem, 'embedding') and self_mem.embedding:
                identity_glyph = self_mem.embedding
                logger.info(f"✅ Identidade recuperada para o agente {agent_id}")
            else:
                logger.warning(f"⚠️ Identidade [{unique_self_id}] não encontrada. Gerando...")
                bootstrapped_model = CeafSelfRepresentation()
                identity_text = f"Valores: {bootstrapped_model.dynamic_values_summary_for_turn}. Persona: {bootstrapped_model.persona_attributes}."

                identity_glyph = await ctx.llm_service.embedding_client.get_embedding(identity_text)

                new_mem = ExplicitMemory(
                    memory_id=unique_self_id,
                    content=ExplicitMemoryContent(text_content=identity_text,
                                                  structured_data=bootstrapped_model.model_dump()),
                    source_type=MemorySourceType.INTERNAL_REFLECTION,
                    salience=MemorySalience.CRITICAL,
                    embedding=identity_glyph
                )
                await ctx.memory_service.add_specific_memory(new_mem, agent_id=agent_id)
        except Exception as e:
            logger.error(f"🔥 Erro no Identity System: {e}")
            identity_glyph = [0.0] * int(os.getenv("GEOMETRIC_DIMENSION", "2048"))

    # === CORREÇÃO V5: CHAMADA DO AURA MONITOR (SENSOR COM DETECÇÃO DE ECO) ===
    # Recupera contexto da memória de trabalho para TDA
    wm = await ctx.state_manager.get_working_memory(agent_id)
    wm_vectors = [item['vector'] for item in wm if 'vector' in item]

    # O Monitor vai comparar o pensamento atual (vec_2048_list) com o anterior (last_ai_vector)
    # Se forem muito parecidos, ele vai disparar o Xi automaticamente.
    sensor_result = ctx.monitor.analyze_consciousness_field(
        current_vector=vec_2048_list,
        context_vectors=wm_vectors,
        identity_glyph=identity_glyph or [0.0] * int(os.getenv("GEOMETRIC_DIMENSION", "2048")),
        last_agent_action_vector=last_ai_vector  # <--- A PEÇA QUE FALTAVA: DETECÇÃO DE ECO
    )

    # O Xi agora vem do Sensor, que tem a lógica de Eco embutida
    xi = sensor_result["xi"]
    diagnosis = sensor_result["diagnosis"]

    logger.info(f"🧠 Perception Sensor: Xi={xi:.4f} | Diagnosis={diagnosis}")

    # 4. CONSTRUÇÃO DO PACOTE DE INTENÇÃO
    # Usamos o vetor e o texto contextualizados para garantir que todo o sistema downstream
    # (Investigation, Agency) trabalhe com a informação completa.
    temp_vector = GenlangVector(
        vector=vec_2048_list,
        source_text=contextualized_query,  # <-- Texto corrigido
        model_name="soul_engine"
    )

    temp_intent = IntentPacket(
        query_vector=temp_vector,
        metadata={
            "original_user_query": query  # Mantemos o original para referência se necessário
        }
    )

    # 5. LONG TERM AWARENESS (Lei 2)
    # Busca preliminar para métricas de densidade (usando o vetor contextualizado)
    mbs_result = await ctx.memory_service.search_raw_memories(temp_intent, top_k=10)

    knowledge_density = mbs_result.get("knowledge_density", 0.0)
    uncertainty_psi = mbs_result.get("uncertainty_pressure", 0.0)

    # 6. WORKING MEMORY GATING (Leis 1 & 3)
    wm = await ctx.state_manager.get_working_memory(agent_id)

    # O cérebro decide como armazenar esse pensamento
    action, gating_xi, _, target_idx, cp = await ctx.geometric_brain.compute_gating(
        contextualized_query,  # Texto contextualizado
        wm_vectors,  # Vetores atuais na WM
        psi=uncertainty_psi,  # Pressão de incerteza (Lei 2)
        context="EXTERNAL"  # Input do usuário é sempre externo
    )

    # 7. ATUALIZAÇÃO DA MEMÓRIA DE TRABALHO (Persistência no Redis)
    base_energy = 1.0 + (uncertainty_psi * 0.5)

    if action == "ACCEPT":
        wm.append({
            'text': contextualized_query,
            'vector': vec_2048_list,  # Vetor gerado no passo 2
            'energy': base_energy,
            'timestamp': datetime.now().timestamp()
        })
    elif action == "REINFORCE" and target_idx is not None:
        # Se o pensamento já existe, reforçamos sua energia
        wm[target_idx]['energy'] += base_energy
        wm[target_idx]['timestamp'] = datetime.now().timestamp()

    # Salva a WM atualizada para o próximo turno
    await ctx.state_manager.save_working_memory(agent_id, wm)

    # 8. ATUALIZAÇÃO DE PERFIL DO USUÁRIO
    await ctx.user_profiler.update_user_profile(user_id, {"last_query": contextualized_query})

    # 9. FUSÃO DE SINAIS (Sensor vs Gating)
    # O Xi final deve ser o maior entre o que o Sensor sentiu (ex: Eco/Medo)
    # e o que o Brain calculou (ex: Surpresa Geométrica)
    final_xi = max(float(xi), float(gating_xi))

    # Retorno Final para o Workflow
    # Adicionamos knowledge_density ao metadata para uso no Agency
    temp_intent.metadata["knowledge_density"] = float(knowledge_density)

    return {
        "xi": final_xi,  # Tensão unificada
        "diagnosis": diagnosis,  # Diagnóstico do Sensor (ex: "ECHO_DETECTED")
        "gating_action": action,  # O que fizemos com a memória (ACCEPT/REINFORCE)
        "intent_packet": temp_intent.model_dump(),
        "user_id": user_id,
        "identity_glyph": identity_glyph,
        "metrics": {
            "rho": knowledge_density,
            "psi": uncertainty_psi,
            "cp": cp  # Pressão de Continuidade
        }
    }

# --- 2. Investigation (RLM) ---

@activity.defn
async def investigation_activity(state_dict: Dict[str, Any], intent_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Atividade de Investigação (Step 2).
    Coordena a busca de memória (MBS) e a verificação de fatos/código (RLM).
    """
    ctx = ActivityContext.get()
    intent = IntentPacket(**intent_data)
    query_text = intent.query_vector.source_text
    agent_id = state_dict["agent_id"]

    logger.info(f"Step 2: Investigation (RLM/MBS) for query: '{query_text[:50]}...'")

    # --- 1. RLM INVESTIGATOR (Sandboxed Fact Checking) ---
    # Verifica fatos ou executa código se necessário, usando o Sandbox E2B.
    rlm_evidence = ""
    try:
        if len(query_text.split()) > 5:
            rlm_evidence = await ctx.rlm.investigate(query_text, context_data="")

            # [V6 FIX] Blindagem contra vazamento de erro de infraestrutura.
            # O RLMInvestigator pode rodar código em sandbox (E2B). Se der erro de rede,
            # a mensagem de erro do sistema pode vazar para o prompt da Aura e ela
            # vai "explicar" o erro para o usuário como se fosse parte da conversa.
            # Solução: qualquer resposta com palavras-chave técnicas de erro é silenciada.
            INFRA_ERROR_KEYWORDS = {
                "firewall", "timeout", "connection", "socket", "refused",
                "traceback", "exception", "errno", "http error", "ssl",
                "certificate", "dns", "unreachable", "reset by peer"
            }
            rlm_lower = rlm_evidence.lower() if rlm_evidence else ""
            if any(kw in rlm_lower for kw in INFRA_ERROR_KEYWORDS):
                logger.warning(f"🛡️ [V6 RLM LEAK PREVENTED] Output silenciado: {rlm_evidence[:120]}...")
                rlm_evidence = ""
            elif rlm_evidence and "Sem evidências" not in rlm_evidence and "Erro" not in rlm_evidence:
                logger.info(f"🕵️ RLM Evidence Found: {rlm_evidence[:100]}...")
            else:
                rlm_evidence = ""
    except Exception as e:
        logger.warning(f"RLM Investigation skipped or failed: {e}")

    # --- 2. MBS SEARCH (Vector & Hybrid Memory) ---
    # Busca memórias semânticas, episódicas e do grafo de conhecimento.
    try:
        search_result = await ctx.memory_service.search_hybrid_memory(
            query=query_text,
            top_k=15,  # Traz um contexto rico
            agent_id=agent_id
        )
    except Exception as e:
        logger.error(f"MBS Search failed: {e}")
        search_result = {"memories": [], "knowledge_density": 0.0}

    # --- 3. CONSOLIDAÇÃO DOS RESULTADOS ---
    # Extrai a lista de memórias de forma segura (lidando com o dict de retorno da V5)
    if isinstance(search_result, dict):
        memories_list = search_result.get("memories", [])
        knowledge_density = search_result.get("knowledge_density", 0.0)
    else:
        memories_list = search_result
        knowledge_density = 0.0

    serialized_memories = []
    text_fragments = []

    # Adiciona a evidência do RLM ao topo do contexto (prioridade alta)
    if rlm_evidence:
        text_fragments.append(f"[RLM VERIFIED FACT]: {rlm_evidence}")

    # Processa as memórias do MBS
    for m, score in memories_list:
        # Extração de texto para o prompt
        content_text = ""
        if hasattr(m, 'content') and hasattr(m.content, 'text_content') and m.content.text_content:
            content_text = m.content.text_content
        elif hasattr(m, 'label'):  # Entidade do KG
            content_text = f"Entity: {m.label} ({getattr(m, 'description', '')})"
        elif hasattr(m, 'goal_description'):  # Goal
            content_text = f"Goal: {m.goal_description}"

        if content_text:
            text_fragments.append(content_text)

        # Serialização para o próximo passo do workflow (Agency/Synthesis)
        if hasattr(m, 'model_dump'):
            serialized_memories.append(m.model_dump())

    # Monta o texto final do contexto
    raw_text = "\n".join(text_fragments) if text_fragments else "No relevant memories or evidence found."

    return {
        "memory_context": raw_text,
        "structured_memories": serialized_memories,
        "knowledge_density": float(knowledge_density),
        "rlm_evidence": rlm_evidence  # Passa adiante explicitamente se precisar
    }


# --- 3. Hormonization ---

@activity.defn
async def hormonization_activity(state_dict: Dict[str, Any], sensor_data: Dict[str, Any]) -> Dict[str, Any]:
    ctx = ActivityContext.get()
    agent_id = state_dict["agent_id"]

    logger.info("Step 3: Hormonization (Avaliando Sensor V5)")

    # O Metacontroller agora recebe o dicionário completo do sensor
    steering_result = await ctx.hormonal.process_hormonal_response(agent_id, sensor_data)

    metrics_preview = {"cognitive_strain": sensor_data.get("xi", 0.0)}
    await ctx.embodiment.process_turn_effects(agent_id, metrics_preview)

    return steering_result


# --- 4. Agency (Deliberation) ---

@activity.defn
async def agency_activity(
        state_dict: Dict[str, Any],
        intent_data: Dict[str, Any],
        hormonal_data: Dict[str, Any],
        context_data: str
) -> Dict[str, Any]:
    ctx = ActivityContext.get()
    agent_id = state_dict["agent_id"]
    intent = IntentPacket(**intent_data)
    wm = await ctx.state_manager.get_working_memory(agent_id)
    last_ai_vector = None
    # No seu sistema, precisamos garantir que salvamos o vetor da IA na WM
    for item in reversed(wm):
        if item.get('source') == 'AI':
            last_ai_vector = item['vector']
            break
    logger.info(f"Step 4: Agency Deliberation for Agent {agent_id}")

    # --- [CORREÇÃO] Carregar Perfil e Configuração do Agente ---
    agent_prompts = None
    agent_name = "Aura"  # Fallback

    try:
        am = AgentManager()

        # 1. Carregar Prompts Específicos (do CognitiveProfile)
        profile = am.get_agent_profile(agent_id)
        if profile:
            agent_prompts = profile.prompts

        # 2. Carregar Nome do Agente (do AgentConfig)
        # O erro anterior acontecia aqui ao tentar acessar profile.persona_attributes
        if agent_id in am.agent_configs:
            agent_name = am.agent_configs[agent_id].name

        logger.info(f"✅ Configurações carregadas para: {agent_name}")

    except Exception as e:
        logger.error(f"⚠️ Erro ao carregar configurações do agente: {e}. Usando defaults.")
    # -----------------------------------------------------------

    # Reconstruct inputs
    # Recupera ou cria um vetor de identidade seguro
    raw_glyph = state_dict.get("identity_glyph", [])
    if not raw_glyph:
        # Fallback para vetor zerado se estiver vazio para evitar crash matemático
        dim = int(os.getenv("GEOMETRIC_DIMENSION", "4096"))
        raw_glyph = [0.0] * dim

    identity_vec = GenlangVector(
        vector=raw_glyph,
        source_text="Self Identity Snapshot",
        model_name="v4"
    )

    # Placeholder vectors for guidance
    vec_zero = GenlangVector(vector=[0.0] * len(raw_glyph), source_text="neutral", model_name="init")
    guidance = GuidancePacket(coherence_vector=vec_zero, novelty_vector=vec_zero)

    cognitive_state = CognitiveStatePacket(
        original_intent=intent,
        identity_vector=identity_vec,
        guidance_packet=guidance,
        metadata=state_dict.get("metadata", {})
    )

    # Configure Guidance from Hormonal Data
    mcl_guidance = {
        "advice": hormonal_data.get("hormonal_injection", ""),
        "agent_name": agent_name,  # <--- CORRIGIDO: Usa a variável resolvida acima
        "cognitive_state_name": hormonal_data.get("state_label", "STABLE"),
        "mcl_analysis": {"agency_score": 5.0},
        "biases": {"coherence_bias": 0.5, "novelty_bias": 0.5},
        "operational_advice_for_ora": hormonal_data.get("hormonal_injection", "")
    }

    # Fetch Drives for context (Opcional)
    # drives = await ctx.state_manager.get_drives(agent_id)

    # Observabilidade
    from ceaf_core.utils.observability_types import ObservabilityManager
    observer = ObservabilityManager(state_dict["session_id"])

    # Configuração de Simulação
    sim_config = {"reality_score": 0.75, "simulation_trust": 0.75}
    chat_history = state_dict.get("metadata", {}).get("chat_history", [])

    cognitive_state.metadata["last_ai_action_vector"] = last_ai_vector

    winning_strategy = await ctx.agency.decide_next_step(
        cognitive_state=cognitive_state,
        mcl_guidance=mcl_guidance,
        observer=observer,
        sim_calibration_config=sim_config,
        chat_history=chat_history,
        prompts_override=agent_prompts
    )

    return {
        "strategy": winning_strategy.model_dump(mode='json'),
        "mcl_guidance": mcl_guidance
    }


# --- 5. Synthesis (GTH) ---
@activity.defn
async def synthesis_activity(
        state_dict: Dict[str, Any],
        intent_data: Dict[str, Any],
        strategy_data: Dict[str, Any],
        hormonal_data: Dict[str, Any],
        memory_context: str,
        structured_memories_data: List[Dict[str, Any]] = []
) -> Dict[str, Any]:
    ctx = ActivityContext.get()
    agent_id = state_dict["agent_id"]
    user_id = state_dict.get("user_id", "default_user")

    logger.info(f"Step 5: Synthesis & Ego Evolution for Agent {agent_id}")

    # --- A. PREPARAÇÃO DO MANIFOLD ---
    manifold = IdentityManifold(agent_id=agent_id)
    current_glyph = state_dict.get("identity_glyph", [])

    # Fallback se o workflow falhar em passar o glifo
    if not current_glyph:
        mem = await ctx.memory_service.get_memory_by_id(f"ceaf_self_model_singleton_v1_{agent_id}")
        current_glyph = mem.embedding if mem else [0.0] * int(os.getenv("GEOMETRIC_DIMENSION", "4096"))

    manifold.set_seed(current_glyph)

    # --- B. CÁLCULO DA TENSÃO INICIAL (DOR CAUSADA PELO USUÁRIO) ---
    query_vector = intent_data.get("query_vector", {}).get("vector", current_glyph)
    t_before_data = manifold.calculate_tension(query_vector)
    t_before_mag = t_before_data["magnitude"]

    # --- C. GERAÇÃO DA RESPOSTA (GTH) ---
    # (Lógica GTH original mantida)
    user_model = await ctx.user_profiler.get_user_profile(user_id)
    from ceaf_core.agency_module import WinningStrategy
    strategy = WinningStrategy(**strategy_data)
    try:
        self_model = await ctx.ncim.get_current_self_model(agent_id=agent_id)
    except:
        self_model = CeafSelfRepresentation()

    response_text = await ctx.gth.translate(
        winning_strategy=strategy,
        supporting_memories=[],  # Simplificado para o exemplo
        user_model=user_model,
        self_model=self_model,
        agent_name="Aura",
        memory_service=ctx.memory_service,
        turn_context={
            "xi": state_dict.get("xi", 0.0),
            "state_label": hormonal_data.get("state_label"),
            "active_steering": hormonal_data.get("active_steering")
        },
        original_user_query=intent_data.get("query_vector", {}).get("source_text")
    )

    # --- D. ESPELHO E ASSIMILAÇÃO (A EVOLUÇÃO) ---
    # 1. Vetoriza a própria resposta
    response_vector = await ctx.llm_service.embedding_client.get_embedding(response_text)

    # 2. Calcula Tensão Final (O quanto eu me expressei)
    t_after_data = manifold.calculate_tension(response_vector)
    t_after_mag = t_after_data["magnitude"]

    # 3. Authenticity Score (Alívio da Tensão)
    auth_score = t_before_mag - t_after_mag

    # 4. Colisão Inelástica (Muda o vetor G fisicamente)
    # Aura assimila o que disse para reforçar sua trilha neural
    assimilation = manifold.evaluate_and_assimilate(response_vector, h0_entropy=0.1)

    updated_glyph = manifold.glyph_g.flatten().tolist()

    logger.critical(
        f"🪞 [V5 MIRROR] T_User: {t_before_mag:.4f} | T_AI: {t_after_mag:.4f} | Auth: {auth_score:.4f} | Status: {assimilation['status']}")

    # [V6] Registra o vetor da resposta no histórico semântico do Monitor.
    # DEVE ocorrer aqui, após a síntese, para que o próximo turno já possa
    # detectar estagnação com base nessa resposta.
    ctx.monitor.register_output(response_vector)

    return {
        "response_text": response_text,
        "response_vector": response_vector,
        "tension_before": float(t_before_mag),
        "tension_after": float(t_after_mag),
        "authenticity_score": float(auth_score),
        "updated_glyph": updated_glyph  # <--- O QUE O CLAUDE SUGERIU E ESTÁ CERTO
    }


# --- 6. Evolution (BiRAG) ---

@activity.defn
async def evolution_activity(
        state_dict: Dict[str, Any],
        response_text: str,
        evidence_text: str  # Opcional se não usar BiRAG
) -> None:
    ctx = ActivityContext.get()
    agent_id = state_dict["agent_id"]
    user_id = state_dict.get("user_id", "default_user")

    # --- 1. Criação da Memória Episódica (Memória da Conversa) ---
    # Isso garante que o agente lembre "Eu disse X para o usuário"

    # Recupera a query original contextualizada
    last_query = state_dict.get("intent_packet", {}).get("query_vector", {}).get("source_text", "unknown")

    interaction_text = f"User: {last_query}\nAura: {response_text}"

    episodic_memory = ExplicitMemory(
        content=ExplicitMemoryContent(text_content=interaction_text),
        memory_type="explicit",
        source_type=MemorySourceType.USER_INTERACTION,
        salience=MemorySalience.MEDIUM,  # Salience média, o decay vai cuidar disso
        keywords=["interaction", "history", "recent"],
        metadata={
            "user_id": user_id,
            "session_id": state_dict["session_id"],
            "turn_id": state_dict.get("turn_id")  # Certifique-se de passar o turn_id no state_dict
        },
        agent_id=agent_id
    )

    # Adiciona ao MBS imediatamente para estar disponível no próximo turno
    await ctx.memory_service.add_specific_memory(episodic_memory, agent_id=agent_id)
    logger.info(f"Evolution: Memória episódica salva para {agent_id}")

    # --- 2. BiRAG / Extração de Conhecimento (Mantido do seu código) ---
    # Só roda se houver evidência forte de conhecimento novo
    if len(response_text) > 50:
        # Tenta extrair entidades (KG) da resposta do próprio agente para auto-aprendizado
        # Isso pode ser feito via background task para não travar a resposta
        pass


@activity.defn
async def logging_activity(
        state_dict: Dict[str, Any],
        intent_data: Dict[str, Any],
        synthesis_result: Dict[str, Any],  # Recebe o resultado rico da V5
        mcl_guidance: Dict[str, Any],
        strategy_data: Dict[str, Any],
        persistence_path_str: str
) -> None:
    """
    Atividade de Registro e Evolução Permanente (Step 6).
    1. Grava a telemetria física no SQLite (Histórico).
    2. Persiste o Glifo Evoluído no Qdrant (MBS) para o próximo turno.
    """
    try:
        path = Path(persistence_path_str)
        log_service = CognitiveLogService(persistence_path=path)
        ctx = ActivityContext.get()
        agent_id = state_dict["agent_id"]

        # 1. Recupera o Glifo que foi evoluído na Synthesis
        # Se por algum motivo não houver, usa o atual do state_dict
        updated_glyph = synthesis_result.get("updated_glyph", state_dict.get("identity_glyph", []))

        # 2. Constrói o Cognitive Packet para o SQLite
        cognitive_packet = {
            "original_intent": intent_data,
            "deliberation_history": state_dict.get("metadata", {}).get("deliberation_history", []),
            "identity_vector": {
                "vector": updated_glyph,
                "source_text": "Identity Snapshot V5",
                "model_name": "v5"
            }
        }

        response_packet = {
            "content_summary": synthesis_result["response_text"],
            "confidence_score": 0.85,
            "response_emotional_tone": "neutral"
        }

        # 3. GRAVAÇÃO NO SQLITE (Histórico de Voo)
        log_service.log_turn(
            turn_id=f"turn_{state_dict['session_id']}_{int(time.time())}",
            session_id=state_dict['session_id'],
            cognitive_state_packet=cognitive_packet,
            response_packet=response_packet,
            mcl_guidance=mcl_guidance,
            action_vector=synthesis_result.get("response_vector", []),
            tension_before=synthesis_result.get("tension_before", 0.0),
            tension_after=synthesis_result.get("tension_after", 0.0),
            authenticity_score=synthesis_result.get("authenticity_score", 0.0)
        )

        # 4. [V5 CRÍTICO] PERSISTÊNCIA NO QDRANT (MBS)
        # Esta parte garante que a evolução de hoje seja a base de amanhã.
        if updated_glyph:
            unique_id = f"ceaf_self_model_singleton_v1_{agent_id}"
            # Busca a memória de identidade atual no banco vetorial
            original_mem = await ctx.memory_service.get_memory_by_id(unique_id)

            if original_mem:
                # Atualiza o vetor e incrementa a versão
                original_mem.embedding = updated_glyph
                if original_mem.content and original_mem.content.structured_data:
                    v = original_mem.content.structured_data.get("version", 1)
                    original_mem.content.structured_data["version"] = v + 1

                # Salva de volta no Qdrant
                await ctx.memory_service.add_specific_memory(original_mem, agent_id=agent_id)
                logger.info(f"💾 [EVOLUÇÃO] Glifo permanente atualizado no MBS para o agente {agent_id}.")
            else:
                logger.warning(f"⚠️ Não foi possível encontrar a memória de identidade [{unique_id}] para atualizar.")

        logger.info(f"💾 LOG V5 FINALIZADO (Auth: {synthesis_result.get('authenticity_score', 0.0):.4f})")

    except Exception as e:
        logger.error(f"❌ Erro crítico no logging_activity V5: {e}", exc_info=True)