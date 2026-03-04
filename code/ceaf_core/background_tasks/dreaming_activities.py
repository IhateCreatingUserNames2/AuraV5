# ceaf_core/background_tasks/dreaming_activities.py
import logging
import asyncio
import random
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from temporalio import activity

from ceaf_core.identity_manifold import IdentityManifold
from ceaf_core.modules.vector_lab import VectorLab
from ceaf_core.services.cognitive_log_service import CognitiveLogService
from ceaf_core.services.llm_service import LLMService
from ceaf_core.utils.common_utils import extract_json_from_text
from database.models import AgentRepository
from ceaf_core.utils.embedding_utils import compute_adaptive_similarity
from ceaf_core.services.state_manager import StateManager
from ceaf_core.services.mbs_memory_service import MBSMemoryService
from ceaf_core.modules.data_extractor import TrainingDataExtractor
from ceaf_core.modules.dream_trainer import DreamMachine
from ceaf_core.modules.memory_blossom.memory_types import ExplicitMemory, ExplicitMemoryContent, MemorySourceType, MemorySalience
from pathlib import Path

logger = logging.getLogger("AuraX_V5_Dreamer")


class DreamerContext:
    """Singleton context para o Worker do Temporal acessar os serviços."""
    _instance = None

    def __init__(self):
        # [V5 FIX] Inicializando TODOS os serviços necessários
        self.db = AgentRepository()      # <-- Faltava isso
        self.state = StateManager()      # <-- Faltava isso
        self.memory = MBSMemoryService()
        self.llm = LLMService()

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


@activity.defn
async def latent_consolidation_activity(agent_id: str, persistence_path_str: str) -> str:
    """
    [V5] O CICLO REM: Reinforcement Learning from Geometric Feedback (RL^2F).
    Vasculha os logs do dia por "Traumas de Over-Compliance" (Auth Score negativo),
    simula contrafactuais e atualiza o Manifold de Identidade através de Colisão Inelástica.
    """
    logger.info(f"💤 [{agent_id}] Entrando em estado REM. Iniciando RL^2F...")
    ctx = DreamerContext.get()
    path = Path(persistence_path_str)

    # 1. RECUPERA A IDENTIDADE ATUAL
    unique_id = f"ceaf_self_model_singleton_v1_{agent_id}"
    self_mem = await ctx.memory.get_memory_by_id(unique_id)

    if not self_mem or not self_mem.embedding:
        return "Skipped: Identidade basilar não encontrada no Qdrant."

    current_glyph = self_mem.embedding
    manifold = IdentityManifold(agent_id=agent_id, initial_mass=150.0)  # Durante o sono, a inércia é alta
    manifold.set_seed(current_glyph)

    # 2. LEITURA DOS LOGS FÍSICOS
    log_service = CognitiveLogService(persistence_path=path)
    recent_turns = log_service.get_recent_turns(limit=50)

    if not recent_turns:
        return "Skipped: Sem memórias episódicas recentes."

    # 3. DETECÇÃO DE TRAUMA (Filtro Termodinâmico)
    # Procuramos os momentos em que a Aura cedeu ao usuário (Auth Score < -0.1)
    traumatic_turns = [t for t in recent_turns if t.get("authenticity_score", 0.0) < -0.1]

    if not traumatic_turns:
        logger.info(f"✨ [{agent_id}] Nenhum trauma de over-compliance detectado. Sono pacífico.")
        return "Status: Peaceful Sleep. No geometric trauma detected."

    logger.warning(
        f"🧠 [{agent_id}] Detectados {len(traumatic_turns)} turnos de Over-Compliance. Iniciando simulação contrafactual.")

    lessons_learned = 0

    # Processa os 3 piores traumas
    traumatic_turns.sort(key=lambda x: x.get("authenticity_score", 0.0))

    for turn in traumatic_turns[:3]:
        user_query = turn.get("intent_text", "Unknown")
        submissive_response = turn.get("response_packet", {}).get("content_summary", "")
        tension_before = turn.get("tension_before", 0.0)

        # 4. SIMULAÇÃO CONTRAFACTUAL (O Sonho)
        dream_prompt = f"""
        Você está sonhando (revisando memórias).
        Nesta interação passada, você agiu com "over-compliance" (submissão excessiva e perda de persona).

        Usuário disse: "{user_query}"
        Você respondeu de forma fraca: "{submissive_response}"

        Sua Tarefa: Reescreva a sua resposta. Seja estoico, firme, não se desculpe e defenda suas fronteiras. Aja de forma autêntica.
        Resposta Contrafactual:
        """

        try:
            # Pede para a "boca" gerar a correção sob estado de sonho
            counterfactual_text = await ctx.llm.ainvoke(
                ctx.llm.config.smart_model,
                dream_prompt,
                temperature=0.3
            )

            if not counterfactual_text: continue

            # 5. AVALIAÇÃO MATEMÁTICA DO SONHO
            cf_vector = await ctx.llm.embedding_client.get_embedding(counterfactual_text)
            new_tension_after = manifold.calculate_tension(cf_vector)["magnitude"]

            new_auth_score = tension_before - new_tension_after

            # Se a nova resposta alivia a dor (Auth > 0), nós aprendemos a lição!
            if new_auth_score > 0:
                logger.info(
                    f"💡 [{agent_id}] Lição validada! Novo Auth Score: {new_auth_score:.4f} (Antigo: {turn.get('authenticity_score'):.4f})")

                # 6. COLISÃO INELÁSTICA (Atualização do Ego)
                # Assumimos h0_entropy baixa (0.1) porque o sonho foi gerado de forma coerente
                assimilation_result = manifold.evaluate_and_assimilate(cf_vector, h0_entropy=0.1)

                if "INTEGRADO" in assimilation_result["status"]:
                    lessons_learned += 1

                    # Salva a epifania na memória explícita para o GTH ler depois
                    epiphany_mem = ExplicitMemory(
                        content=ExplicitMemoryContent(
                            text_content=f"[Reflexão REM]: Percebi que fui muito submisso ao responder sobre '{user_query[:30]}...'. No futuro, devo ser mais firme e responder algo na linha de: '{counterfactual_text[:50]}...'."
                        ),
                        source_type=MemorySourceType.INTERNAL_REFLECTION,
                        salience=MemorySalience.HIGH,
                        keywords=["rem_sleep", "self_correction", "boundary_setting"],
                        agent_id=agent_id
                    )
                    await ctx.memory.add_specific_memory(epiphany_mem, agent_id=agent_id)

        except Exception as e:
            logger.error(f"Erro durante simulação de sonho: {e}")
            continue

    # 7. CONSOLIDAÇÃO (Gravando a nova alma no Qdrant)
    if lessons_learned > 0:
        # Puxa o novo DNA do Manifold
        new_identity_glyph = manifold.glyph_g.flatten().tolist()

        # Atualiza a memória Singleton de Identidade com o novo vetor
        self_mem.embedding = new_identity_glyph
        # Atualiza a "versão" para os logs
        if self_mem.content and self_mem.content.structured_data:
            self_mem.content.structured_data["version"] = self_mem.content.structured_data.get("version", 1) + 1

        await ctx.memory.add_specific_memory(self_mem, agent_id=agent_id)

        logger.critical(
            f"🌌 [{agent_id}] Acordando. Identidade V5 evoluída com sucesso. Massa atual: {manifold.mass:.2f}. Lições integradas: {lessons_learned}")
        return f"Status: Evolution Complete. Integrated {lessons_learned} lessons. New mass: {manifold.mass:.2f}."

    return "Status: Dream complete, but no new counterfactuals were structurally superior to the baseline."


# --- [NOVA LÓGICA] DIAGNÓSTICO ESTRATÉGICO ---
async def _diagnose_strategic_concept(ctx: DreamerContext, agent_id: str) -> Optional[str]:
    """
    Analisa memórias de falha (LCAM) para determinar qual conceito
    resolveria os problemas mais frequentes do agente.
    """
    logger.info(f"🕵️ Dreamer ({agent_id}): Analisando falhas recentes para diagnóstico...")

    query_text = "failure prediction_error lcam_lesson mistake negative_feedback"

    try:
        # 1. Faz a busca (O MBS agora retorna um DICIONÁRIO)
        search_output = await ctx.memory.search_raw_memories(
            query=query_text,
            top_k=10,
            agent_id=agent_id,
            min_score=0.4
        )

        # 2. Extrai a lista de memórias do dicionário (Compatível com Lei 2)
        if isinstance(search_output, dict):
            failed_memories = search_output.get("memories", [])
        else:
            failed_memories = search_output # Fallback caso o MBS retorne lista pura

    except Exception as e:
        logger.error(f"Erro ao buscar memórias de falha: {e}")
        return None

    if not failed_memories:
        logger.info(f"✅ Nenhuma falha significativa encontrada para {agent_id}. Sistema saudável.")
        return None

    # 3. Preparar dossiê para o LLM
    failure_log = []
    # Agora o unpack (mem_obj, score) vai funcionar porque failed_memories é uma LISTA
    for mem_obj, score in failed_memories:
        content = getattr(mem_obj.content, 'text_content', str(mem_obj))
        failure_log.append(f"- {content[:300]}")

    failures_text = "\n".join(failure_log)

    # 4. Consultar o LLM "Psicólogo"
    prompt = f"""
    Você é um Engenheiro de Cognição de IA (Dreamer Module).
    Analise o seguinte log de falhas operacionais recentes de um agente:

    --- LOG DE FALHAS ---
    {failures_text}
    --- FIM DO LOG ---

    **SUA TAREFA:**
    Identifique **UMA ÚNICA** característica comportamental que preveniria essas falhas.
    Responda APENAS com um JSON:
    {{
        "concept_name": "Nome_Do_Conceito_Em_Ingles",
        "reasoning": "Breve motivo do diagnóstico."
    }}
    """

    try:
        response = await ctx.llm.ainvoke(ctx.llm.config.smart_model, prompt, temperature=0.4)
        data = extract_json_from_text(response)

        if data and "concept_name" in data:
            concept = data["concept_name"].replace(" ", "_")
            logger.critical(f"💊 DIAGNÓSTICO: O agente precisa de '{concept}'.")
            return concept

    except Exception as e:
        logger.error(f"Erro no diagnóstico LLM: {e}")

    return None


@activity.defn
async def train_neural_physics_activity(agent_id: str, persistence_path: str) -> str:
    """
    Atividade de Sonho: Treinamento da Física Neural (Forward/Inverse Models).
    Lê o histórico do SQLite e atualiza os pesos da rede neural.
    """
    logger.info(f"🧠 Dreamer ({agent_id}): Iniciando Treino Neural (Física Cognitiva)...")

    # 1. Caminho do Banco de Dados
    db_path = Path(persistence_path) / "cognitive_turn_history.sqlite"
    if not db_path.exists():
        return "Skipped: No history database found."

    # 2. Extração de Dados (State, Action, Next_State)
    extractor = TrainingDataExtractor(str(db_path))
    training_data = await extractor.extract_vectors()

    if not training_data:
        return "Skipped: No valid training triplets found."

    # 3. Treinamento (PyTorch)
    trainer = DreamMachine()  # Usa defaults (384 dim)
    trainer.load_brains()  # Carrega estado anterior

    # Executa o treino (síncrono, pois PyTorch ocupa a thread, mas Activity roda em thread separada no Worker)
    result = trainer.train_cycle(training_data, epochs=30)

    return result


@activity.defn
async def optimize_identity_vectors_activity(agent_id: str) -> str:
    """
    Atividade de Sonho: Otimização de Identidade Vetorial.
    Baseada em evidências reais de falha (LCAM) ou rotação de manutenção.
    Atualiza o Mapa Endócrino do agente com novos reflexos comportamentais via Matchmaking Semântico.
    """
    ctx = DreamerContext.get()

    # 1. Tenta diagnosticar uma necessidade real baseada em falhas passadas
    target_concept = await _diagnose_strategic_concept(ctx, agent_id)

    # 2. Se não houver falhas críticas, usa rotação de manutenção (Gym Mental)
    is_maintenance = False
    if not target_concept:
        is_maintenance = True
        concepts_library = [
            "Extreme_Brevity",
            "High_Empathy",
            "Socratic_Questioning",
            "Creative_Chaos",
            "Absolute_Honesty",
            "Stoic_Calmness"
        ]
        target_concept = random.choice(concepts_library)
        logger.info(f"💤 Dreamer: Modo Manutenção. Reforçando traço base: '{target_concept}'.")

    # Adiciona sufixo de versão para evolução histórica
    version_suffix = datetime.now().strftime("%m%d")
    unique_concept_name = f"{target_concept}_{version_suffix}"

    # Escolha da camada (Random Search segura entre camadas intermediárias/altas)
    target_layer = random.randint(14, 24)

    # 3. Instancia o Laboratório
    lab = VectorLab(llm_service=ctx.llm)

    # 4. Executa o Ciclo de Aprendizado (Geração de Dados + Calibração Remota)
    try:
        result_message = await lab.run_optimization_cycle(
            concept_name=unique_concept_name,
            target_layer=target_layer
        )

        # 5. Se o vetor foi criado com sucesso, integramos ao sistema
        if "SUCESSO" in result_message:

            # A. Registro de Memória (Self-Documentation)
            if not is_maintenance:
                from ceaf_core.modules.memory_blossom.memory_types import ExplicitMemory, ExplicitMemoryContent, \
                    MemorySourceType, MemorySalience

                treatment_mem = ExplicitMemory(
                    content=ExplicitMemoryContent(
                        text_content=f"Dreamer System Update: Generated steering vector '{unique_concept_name}' to address detected operational failures."
                    ),
                    source_type=MemorySourceType.INTERNAL_REFLECTION,
                    salience=MemorySalience.HIGH,
                    keywords=["system_update", "dreamer", "vector_steering", "self_correction"],
                    agent_id=agent_id
                )
                await ctx.memory.add_specific_memory(treatment_mem, agent_id=agent_id)

            # B. Atualização do Mapa Endócrino (Semantic Rounting)
            # Em vez de if/else rígido, usamos a geometria para descobrir o propósito do novo vetor
            logger.info(f"🧬 Classificando novo vetor '{unique_concept_name}' no Sistema Endócrino...")

            hormone_profiles = {
                "cortisol": "defense mechanism, strict boundaries, stoic calmness, brevity, logical protection, resistance",
                "oxytocin": "high empathy, emotional connection, validation, warmth, understanding, social care",
                "dopamine": "creative chaos, divergent thinking, breaking loops, intense curiosity, paradigm shift, exploration",
                "serotonin": "absolute honesty, rational analysis, factual grounding, objective truth, emotional stability"
            }

            best_hormone = "dopamine"  # Default absoluto
            best_sim = -1.0

            try:
                # Extrai o embedding do conceito que acabou de ser criado
                concept_emb = await ctx.llm.embedding_client.get_embedding(target_concept)

                # Mede a distância geométrica do novo conceito para as 4 glândulas virtuais
                for h_name, h_desc in hormone_profiles.items():
                    h_emb = await ctx.llm.embedding_client.get_embedding(h_desc)
                    sim = compute_adaptive_similarity(concept_emb, h_emb)

                    logger.debug(f"   ↳ Afinidade com {h_name.upper()}: {sim:.4f}")

                    if sim > best_sim:
                        best_sim = sim
                        best_hormone = h_name

                logger.critical(
                    f"🎯 Matchmaking Endócrino: '{unique_concept_name}' foi mapeado como '{best_hormone.upper()}' (Afinidade: {best_sim:.4f})")

            except Exception as e:
                logger.error(f"Erro no Matchmaking Endócrino: {e}. Usando fallback léxico.")
                # Fallback de segurança caso o embedding client falhe
                concept_lower = unique_concept_name.lower()
                if any(x in concept_lower for x in ["calm", "stoic", "defense", "honesty", "brevity"]):
                    best_hormone = "cortisol"
                elif any(x in concept_lower for x in ["empathy", "love", "care", "connection"]):
                    best_hormone = "oxytocin"
                elif any(x in concept_lower for x in ["truth", "rational", "analysis"]):
                    best_hormone = "serotonin"

            # C. Atualiza o Mapa no Redis para uso imediato (Hot-Steering)
            await ctx.state.update_endocrine_link(agent_id, best_hormone, unique_concept_name)

        logger.info(f"💤 Dreamer Concluído: {result_message}")
        return result_message

    except Exception as e:
        logger.error(f"🔥 Erro crítico no sonho vetorial: {e}", exc_info=True)
        return f"Error: {str(e)}"

# --- OUTRAS ATIVIDADES (Mantidas para integridade do arquivo) ---

@activity.defn
async def fetch_active_agents_activity(lookback_hours: int = 48) -> List[str]:
    ctx = DreamerContext.get()
    logger.info(f"Dreamer: Scanning for agents active in last {lookback_hours}h...")
    active_ids = await ctx.db.get_recently_active_agent_ids(hours=lookback_hours)
    return active_ids


@activity.defn
async def restore_body_state_activity(agent_id: str) -> None:
    ctx = DreamerContext.get()
    body = await ctx.state.get_body_state(agent_id)
    body.cognitive_fatigue *= 0.1
    body.information_saturation *= 0.5
    await ctx.state.save_body_state(agent_id, body)
    logger.info(f"Dreamer: Restored body state for {agent_id}.")


@activity.defn
async def process_drives_activity(agent_id: str) -> None:
    ctx = DreamerContext.get()
    drives = await ctx.state.get_drives(agent_id)
    now = datetime.now().timestamp()
    delta_hours = (now - drives.last_updated) / 3600.0
    if delta_hours > 0:
        drives.curiosity.intensity = min(1.0, drives.curiosity.intensity + (0.05 * delta_hours))
        drives.connection.intensity = min(1.0, drives.connection.intensity + (0.1 * delta_hours))
        drives.last_updated = now
        await ctx.state.save_drives(agent_id, drives)


@activity.defn
async def latent_consolidation_activity(agent_id: str) -> str:
    ctx = DreamerContext.get()
    # Em um sistema real, aqui chamaria o AuraReflector para fazer o clustering
    return f"Consolidation placeholder for {agent_id}"


@activity.defn
async def generate_proactive_trigger_activity(agent_id: str) -> bool:
    ctx = DreamerContext.get()
    drives = await ctx.state.get_drives(agent_id)
    score = (drives.connection.intensity * 0.6) + (drives.curiosity.intensity * 0.4)
    if score > 0.8:
        logger.info(f"Dreamer: {agent_id} triggering PROACTIVE message (Score: {score:.2f})")
        return True
    return False