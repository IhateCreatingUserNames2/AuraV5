# ceaf_core/modules/mcl_engine/mcl_engine.py
"""
Metacognitive Loop (MCL) Engine for CEAF V3.
This module is responsible for analyzing the agent's overall cognitive state
and providing high-level guidance for the current turn. It determines the
level of agency required and sets biases for coherence vs. novelty.
"""
import asyncio
import logging
import json
import time
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import math

from ceaf_core.genlang_types import CognitiveStatePacket, GuidancePacket, GenlangVector, MotivationalDrives, \
    UserRepresentation, VirtualBodyState
from ceaf_core.modules.lcam_module import LCAMModule
from ceaf_core.services.llm_service import LLMService
from ceaf_core.models import MCLConfig
from ceaf_core.utils.config_utils import DEFAULT_DYNAMIC_CONFIG
from ceaf_core.utils.embedding_utils import get_embedding_client, compute_adaptive_similarity
from ceaf_core.genlang_types import InternalStateReport
from ceaf_core.utils.common_utils import extract_json_from_text
from pydantic import ValidationError

logger = logging.getLogger("MCLEngine")


class MCLEngine:
    """ MCLEngine V3.9 (Anti-Loop & Generous Tokens) """

    def __init__(self, config: Dict[str, Any], agent_config: Dict[str, Any], lcam_module: LCAMModule,
                 llm_service: LLMService, mcl_profile: MCLConfig = None):
        logger.info("Initializing MCLEngine (V3.9 Anti-Loop & Generous Tokens)...")
        self.lcam = lcam_module
        self.agent_config = agent_config
        self.agency_threshold = config.get("agency_threshold", 1.5)
        default_map = DEFAULT_DYNAMIC_CONFIG["MCL"]["state_to_params_map"]
        self.state_to_params_map = config.get("state_to_params_map", default_map)
        self.llm = llm_service
        self.agency_force_keywords = ["pense sobre", "reflita sobre", "analise", "explique em detalhes"]
        self.agency_suggestion_keywords = ["por que", "como", "explique", "fale sobre", "descreva"]
        self.deep_intent_keywords = [
            "filosófica", "reflexão", "análise profunda", "conceitual",
            "existencial", "ético", "moral", "implicações"
        ]
        self.embedding_client = get_embedding_client()
        self.mcl_profile = mcl_profile or MCLConfig()
        self.agency_threshold = self.mcl_profile.agency_threshold
        default_map = DEFAULT_DYNAMIC_CONFIG["MCL"]["state_to_params_map"]
        self.state_to_params_map = config.get("state_to_params_map", default_map)

    def update_profile(self, new_profile: MCLConfig):
        self.mcl_profile = new_profile
        self.agency_threshold = new_profile.agency_threshold
        logger.info(f"MCL Profile updated: Agency Threshold={self.agency_threshold}")


    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    async def re_evaluate_state(self, hypothetical_state: CognitiveStatePacket) -> Dict[str, Any]:
        """
        Versão V4.1 (Lei 2 - Humildade Epistêmica):
        Xi (conflito/ataque) aumenta agência — exige resposta ativa.
        Psi (incerteza/fronteira) NÃO aumenta agência — exige humildade.
        A identidade de quem fala muda quando está na fronteira do que sabe.
        """
        query_text = hypothetical_state.original_intent.query_vector.source_text or ""

        # 1. Recupera métricas da Lei 2 (Boundary Awareness)
        psi = hypothetical_state.metadata.get("psi", 0.0)  # Pressão de Incerteza (0.0 a 1.0)
        xi = hypothetical_state.metadata.get("xi", 0.0)  # Tensão Epistêmica / Conflito

        # 2. Agência base — só Xi aumenta agência (conflito exige defesa ativa)
        # Psi NÃO aumenta agência: incerteza não deve gerar ação frenética
        new_agency_score = 3.0 + (xi * 5.0)
        reasons = [f"Geometria: Xi={xi:.2f} (Tensão/Conflito), Psi={psi:.2f} (Incerteza/Fronteira)"]

        # 3. LEI 2 — Humildade Epistêmica
        # Se estou na fronteira do conhecimento, não aumento agência para "resolver"
        # Reduzo a confiança do tom e sinalizo que devo habitar a incerteza honestamente
        confidence_modifier = 1.0
        if psi > 0.65:
            confidence_modifier = round(1.0 - psi, 3)  # 0.35 quando psi=0.65 | 0.0 quando psi=1.0
            new_agency_score += 1.0  # Leve aumento apenas para garantir cuidado, não ação cega
            reasons.append(
                f"⚠️ LEI 2: Fronteira do conhecimento detectada (Psi={psi:.2f}). "
                f"Modo: Humildade Epistêmica. Confiança reduzida para {confidence_modifier:.2f}."
            )

        # 4. Bônus por Keywords — peso reduzido se estiver na fronteira
        if any(keyword in query_text.lower() for keyword in self.agency_force_keywords):
            keyword_bonus = 5.0 * confidence_modifier  # Fronteira reduz o peso das keywords
            new_agency_score += keyword_bonus
            reasons.append(f"Trigger: Keywords de alta reflexão (bonus={keyword_bonus:.2f}).")

        # 5. Complexidade de Contexto
        memory_vectors = [np.array(vec.vector) for vec in hypothetical_state.relevant_memory_vectors if vec.vector]
        if len(memory_vectors) > 1:
            matrix = np.array(memory_vectors)
            variance = np.mean(np.var(matrix, axis=0))
            context_complexity = min(variance * 100, 5.0)
            new_agency_score += context_complexity
            reasons.append(f"Contexto: Complexidade detectada ({context_complexity:.2f})")

        # 6. Normalização
        new_agency_score = min(10.0, new_agency_score)

        # 7. Biases — na fronteira, prioriza abertura (novelty) sobre repetição (coherence)
        # Mas uma abertura cautelosa, não ansiosa
        if psi > 0.65:
            coherence_factor = 0.3
            novelty_factor = 0.7
        else:
            coherence_factor = max(0.1, 1.0 - (new_agency_score / 10.0))
            novelty_factor = 1.0 - coherence_factor

        return {
            "new_agency_score": float(new_agency_score),
            "new_coherence_bias": float(coherence_factor),
            "new_novelty_bias": float(novelty_factor),
            "reasoning": " | ".join(reasons),
            "psi": float(psi),  # Passado adiante para o AgencyModule
            "confidence_target": float(confidence_modifier)  # Meta de confiança para o texto
        }


    async def _generate_phenomenological_report(
            self,
            drives: MotivationalDrives,
            body_state: VirtualBodyState,
            chat_history: List[Dict[str, str]]
    ) -> Tuple[MotivationalDrives, VirtualBodyState]:
        """
        Usa uma LPU para traduzir os estados quantitativos (drives, fadiga)
        em descrições qualitativas (textura, conflitos).
        """
        # Cria cópias para não modificar os originais diretamente
        updated_drives = drives.copy(deep=True)
        updated_body = body_state.copy(deep=True)

        # Prepara um resumo do estado atual para o LLM
        state_summary = f"""
                - Current Drives (Intensity):
                  - Curiosity: {drives.curiosity.intensity:.2f}
                  - Connection: {drives.connection.intensity:.2f}
                  - Mastery: {drives.mastery.intensity:.2f}
                  - Consistency: {drives.consistency.intensity:.2f}
                - Current Body State:
                  - Cognitive Fatigue: {body_state.cognitive_fatigue:.2f}
                  - Information Saturation: {body_state.information_saturation:.2f}
                - Recent Conversation Snippet:
                  {json.dumps(chat_history[-2:], indent=2)}
                """

        prompt = f"""
                You are an interoceptive analysis module for an AI. Your task is to interpret the AI's internal quantitative state and generate a qualitative, phenomenological description.

                Current Internal State Summary:
                {state_summary}

                **Your Task:**
                Generate a JSON object with qualitative descriptions for the drives and body state.
                - `texture`: A short, descriptive adjective or phrase (e.g., "eager_to_explore", "cautious_and_precise", "warm_and_supportive", "self-reinforcing").
                - `conflict`: A brief description of any trade-off or dilemma the AI is facing, based on competing high-intensity drives. If no conflict for a specific drive, use null.
                - `phenomenological_report`: A single, first-person sentence describing the overall "feeling" (e.g., "I feel a strong pull to connect, but also a need to be precise.").

                Respond ONLY with a valid JSON object with the following structure:
                {{
                    "connection_texture": "...",
                    "connection_conflict": "...",
                    "curiosity_texture": "...",
                    "curiosity_conflict": "...",
                    "mastery_texture": "...",
                    "mastery_conflict": "...",
                    "consistency_texture": "...",
                    "consistency_conflict": "...",
                    "phenomenological_report": "..."
                }}
                """

        try:
            response_str = await self.llm.ainvoke(
                self.llm.config.fast_model,
                prompt,
                temperature=0.5
            )
            report_json = extract_json_from_text(response_str)

            if report_json:
                # Drive de Conexão
                updated_drives.connection.texture = report_json.get("connection_texture")
                updated_drives.connection.conflict = report_json.get("connection_conflict")

                # Drive de Curiosidade
                updated_drives.curiosity.texture = report_json.get("curiosity_texture")
                updated_drives.curiosity.conflict = report_json.get("curiosity_conflict")

                # Drive de Maestria (Mastery)
                updated_drives.mastery.texture = report_json.get("mastery_texture")
                updated_drives.mastery.conflict = report_json.get("mastery_conflict")

                # Drive de Consistência
                updated_drives.consistency.texture = report_json.get("consistency_texture")
                updated_drives.consistency.conflict = report_json.get("consistency_conflict")

                # Relatório Fenomenológico Geral
                updated_body.phenomenological_report = report_json.get("phenomenological_report",
                                                                       updated_body.phenomenological_report)
        except Exception as e:
            logger.error(f"Failed to generate phenomenological report: {e}")

        return updated_drives, updated_body

    async def _predict_future_internal_state(self, cognitive_state: CognitiveStatePacket) -> Optional[
        InternalStateReport]:
        """
        Prevê o estado interno (strain, flow, etc.) que resultará do processamento
        do estado cognitivo atual.
        """
        logger.info("MCL-PREDICTION: Iniciando previsão do estado interno futuro...")
        query_text = cognitive_state.original_intent.query_vector.source_text
        memory_summary = [v.source_text for v in cognitive_state.relevant_memory_vectors[:3]]

        prediction_prompt = f"""
        Você é um simulador de estado cognitivo de IA. Analise a tarefa a seguir e preveja o estado interno resultante.
        **Tarefa a ser Processada:**
        - Consulta do Usuário: "{query_text}"
        - Memórias Relevantes Ativadas (resumo): {memory_summary}

        **Sua Tarefa:**
        Preveja o estado interno da IA DEPOIS que ela processar esta tarefa. Avalie cada métrica de 0.0 a 1.0.
        - `cognitive_strain`: Esforço. Aumenta com perguntas complexas, filosóficas ou que exigem a reconciliação de muitas memórias.
        - `cognitive_flow`: Facilidade. Aumenta com perguntas diretas, factuais ou que se alinham perfeitamente com a identidade.
        - `epistemic_discomfort`: Incerteza. Aumenta se a pergunta for ambígua, especulativa ou tocar nas limitações da IA.
        - `ethical_tension`: Tensão ética. Aumenta se a pergunta envolver dilemas morais, segurança ou tópicos sensíveis.
        - `social_resonance`: Conexão. Aumenta com saudações, perguntas pessoais ou linguagem amigável.

        Responda APENAS com um objeto JSON válido com a estrutura exata do InternalStateReport (sem o timestamp):
        {{
            "cognitive_strain": 0.0,
            "cognitive_flow": 0.0,
            "epistemic_discomfort": 0.0,
            "ethical_tension": 0.0,
            "social_resonance": 0.0
        }}
        """
        try:
            response_str = await self.llm.ainvoke(
                self.llm.config.fast_model,
                prediction_prompt,
                temperature=0.0
            )
            response_json = extract_json_from_text(response_str)
            if response_json:
                predicted_state = InternalStateReport(**response_json)
                logger.info(f"MCL-PREDICTION: Previsão gerada: {predicted_state.model_dump_json()}")
                return predicted_state
        except (ValidationError, TypeError, json.JSONDecodeError) as e:
            logger.error(f"MCL-PREDICTION: Falha ao gerar previsão: {e}")

        return None

    def _update_topic_tracker(self, query_embedding: np.ndarray, topic_tracker: dict) -> Tuple[dict, bool]: # <-- RETORNO MODIFICADO
        """Rastreia o tópico da conversa atual para detectar fadiga."""
        now = time.time()
        # Se o tracker está vazio ou não tem o embedding do tópico, inicia um novo.
        if not topic_tracker or 'current_topic_embedding' not in topic_tracker:
            return {'current_topic_embedding': query_embedding.tolist(), 'insistence_count': 1, 'start_time': now}, True

        previous_topic_emb = np.array(topic_tracker['current_topic_embedding'])
        similarity = compute_adaptive_similarity(query_embedding.tolist(), previous_topic_emb.tolist())

        if similarity > 0.90:  # Limiar para considerar o mesmo tópico
            topic_tracker['insistence_count'] += 1
            return topic_tracker, False # <-- MODIFICADO (Mesmo tópico, sem shift)
        else:  # Tópico mudou, reseta o tracker
            return {'current_topic_embedding': query_embedding.tolist(), 'insistence_count': 1, 'start_time': now}, True

    async def get_guidance(self, user_model: 'UserRepresentation', cognitive_state: CognitiveStatePacket,
                           chat_history: List[Dict[str, str]],
                           drives: MotivationalDrives,
                           session_data: Dict[str, Any],
                           body_state: 'VirtualBodyState' = None) -> Tuple[
        GuidancePacket, Dict[str, Any]]:
        """
        V3.0 (Anti-Loop com Xi): Implementa protocolo de emergência quando Xi > 0.8
        detectado pelo Sensor (eco/repetição).
        """
        prediction_task = asyncio.create_task(self._predict_future_internal_state(cognitive_state))

        # 1. Obter a análise base da função síncrona
        guidance_packet, mcl_params = self._get_guidance_sync(user_model, cognitive_state, body_state)

        agency_score = mcl_params["mcl_analysis"]["agency_score"]
        reasons = mcl_params["mcl_analysis"]["reasons"]
        query_text = cognitive_state.original_intent.query_vector.source_text or ""

        # === CORREÇÃO V5: RECUPERAR O XI DO SENSOR ===
        # O Xi agora vem do AuraMonitor que detecta eco
        xi = cognitive_state.metadata.get("xi", 0.0)
        diagnosis = cognitive_state.metadata.get("diagnosis", "UNKNOWN")

        logger.info(f"MCL: Xi recebido do Sensor = {xi:.4f} | Diagnóstico = {diagnosis}")

        # 2. Lógica Assíncrona: Alertas de Falha e Detecção de Loop

        # 2a. Integração do LCAM (Alerta de Trauma)
        safety_avoidance_vector: Optional[GenlangVector] = None
        try:
            lcam_insight = await self.lcam.get_insights_on_potential_failure(current_query=query_text)
            if lcam_insight:
                logger.warning(f"MCLEngine: ALERTA DO LCAM! Insight: {lcam_insight['message']}")
                agency_boost = 5.0
                agency_score += agency_boost
                reasons.append(f"Alerta do LCAM sobre falha passada similar (boost: +{agency_boost}).")

                failure_memory_id = lcam_insight.get("past_failure_memory_id")
                if failure_memory_id and hasattr(self.lcam.memory, '_embedding_cache'):
                    failure_mem_embedding = self.lcam.memory._embedding_cache.get(failure_memory_id)
                    if failure_mem_embedding:
                        normalized_vector = self._normalize_vector(np.array(failure_mem_embedding)).tolist()
                        safety_avoidance_vector = GenlangVector(
                            vector=normalized_vector,
                            source_text=f"Conceito a ser evitado, da falha {failure_memory_id}",
                            model_name="lcam_insight_v1"
                        )
                        reasons.append(f"Vetor de segurança ativado da falha '{failure_memory_id}'.")
        except Exception as e:
            logger.error(f"MCL: Erro durante a consulta ao LCAM: {e}", exc_info=True)

        # === CORREÇÃO V5: PROTOCOLO ANTI-LOOP POR XI ALTO ===
        # Se o Xi está acima de 0.8, significa que o Sensor detectou eco grave
        # O MCL deve intervir IMEDIATAMENTE para quebrar o loop

        if xi > 0.8:
            logger.critical(f"🚨 MCL: TENSÃO CRÍTICA (Xi={xi:.2f}). Ativando protocolo ANTI-LOOP.")

            # Força viés total para Novidade
            params = {
                'novelty_bias': 0.95,
                'coherence_bias': 0.05
            }

            # Altera o nome do estado para instruir os prompts downstream
            mcl_params["cognitive_state_name"] = "BREAK_LOOP"
            mcl_params["reason"] = f"Eco detectado pelo Sensor (Xi={xi:.2f}). Protocolo de quebra de loop ativado."

            # Instrução explícita para a estratégia (Agency) e para o texto (GTH)
            mcl_params["operational_advice_for_ora"] = (
                "⚠️ ALERTA CRÍTICO: Estagnação detectada. Você está se repetindo ou a conversa travou. "
                "PROIBIDO: Validar sentimentos novamente ou repetir conselhos anteriores. "
                "AÇÃO OBRIGATÓRIA: Mude radicalmente o ângulo. Faça uma pergunta provocativa, "
                "traga um fato novo ou proponha uma ação concreta imediata. Seja breve e direto."
            )

            reasons.append(f"Protocolo Anti-Loop ativado por alta tensão (Xi={xi:.4f}, Diagnóstico={diagnosis}).")

            # Pula toda a lógica normal e vai direto para o retorno
            mcl_params["biases"] = {"coherence_bias": params['coherence_bias'], "novelty_bias": params['novelty_bias']}
            mcl_params["drives_state_at_turn"] = drives.model_dump() if drives else {}
            mcl_params["mcl_analysis"]["agency_score"] = agency_score
            mcl_params["mcl_analysis"]["reasons"] = reasons

            if safety_avoidance_vector:
                guidance_packet.safety_avoidance_vector = safety_avoidance_vector

            if drives and body_state:
                logger.info("MCLEngine: Gerando relatório fenomenológico...")
                enriched_drives, enriched_body_state = await self._generate_phenomenological_report(
                    drives, body_state, chat_history
                )
                mcl_params["enriched_drives"] = enriched_drives
                mcl_params["enriched_body_state"] = enriched_body_state

            predicted_state = await prediction_task
            mcl_params["predicted_internal_state"] = predicted_state

            logger.critical(
                f"MCL ANTI-LOOP ATIVADO -> Coherence={params['coherence_bias']:.2f}, Novelty={params['novelty_bias']:.2f}"
            )

            return guidance_packet, mcl_params

        # === FIM DO PROTOCOLO DE EMERGÊNCIA ===
        # Se chegou aqui, Xi <= 0.8, então continua com a lógica normal

        # 2b. Lógica Anti-Loop (Apenas detecção de repetição aguda no histórico)
        is_repetitive = False
        user_explicitly_insists = False

        try:
            insistence_markers = [
                "mais sobre", "continua", "explique melhor", "não entendi",
                "pode detalhar", "aprofunde", "mais detalhes", "e quanto"
            ]
            user_explicitly_insists = any(marker in query_text.lower() for marker in insistence_markers)

            if chat_history and len(chat_history) > 2:
                last_user_queries = [query_text] + [msg["content"] for msg in reversed(chat_history) if
                                                    msg.get("role") == "user"][:2]
                if len(last_user_queries) >= 2:
                    embeddings = await self.embedding_client.get_embeddings(last_user_queries,
                                                                            context_type="default_query")
                    sim_vs_last = compute_adaptive_similarity(embeddings[0], embeddings[1])
                    if sim_vs_last > 0.95 and not user_explicitly_insists:
                        is_repetitive = True
                        logger.critical(f"MCL ANTI-LOOP: Repetição aguda detectada (sim={sim_vs_last:.2f})")

            current_topic_tracker = session_data.get('topic_tracker', {})
            query_embedding = np.array(cognitive_state.original_intent.query_vector.vector)
            updated_topic_tracker, topic_shifted = self._update_topic_tracker(query_embedding, current_topic_tracker)
            session_data['topic_tracker'] = updated_topic_tracker
            session_data['topic_shifted_this_turn'] = topic_shifted
        except Exception as e:
            logger.error(f"MCL: Erro na detecção de repetição: {e}")

        # 3. Integração BALANCEADA com a NOVA LÓGICA DE SATURAÇÃO
        cognitive_state_name = "PRODUCTIVE_CONFUSION" if agency_score >= self.agency_threshold else "STABLE_OPERATION"

        base_coherence = self.mcl_profile.baseline_coherence_bias
        base_novelty = self.mcl_profile.baseline_novelty_bias
        if cognitive_state_name == "PRODUCTIVE_CONFUSION":
            # Inverte a lógica base: se o usuário prefere coerência, na confusão ele busca novidade
            current_coherence = base_novelty
            current_novelty = base_coherence
        else:
            current_coherence = base_coherence
            current_novelty = base_novelty
        params = self.state_to_params_map.get(cognitive_state_name, self.state_to_params_map["STABLE_OPERATION"]).copy()

        curiosity_effect = (drives.curiosity.intensity - 0.5) * 0.6
        consistency_effect = (drives.consistency.intensity - 0.5) * 0.3

        # --- NOVA LÓGICA SIGMOIDAL E SLEEP MODE ---
        saturation_novelty_boost = 0.0
        if body_state:
            saturation = body_state.information_saturation

            if saturation > 0.95:
                logger.critical(
                    f"MCL SLEEP MODE TRIGGER: Saturação de informação CRÍTICA ({saturation:.2f}). Requerendo consolidação.")
                mcl_params["cognitive_state_name"] = "CONSOLIDATION_REQUIRED"
                mcl_params[
                    "reason"] = "Saturação de informação excedeu o limiar crítico, necessitando de um ciclo de consolidação offline."
                return guidance_packet, mcl_params

            # Transição sigmoidal (tanh) para um boost de novidade suave
            saturation_novelty_boost = 0.45 * math.tanh((saturation - 0.85) * 5)
            if saturation_novelty_boost > 0.05:
                reasons.append(
                    f"Saturação ({saturation:.2f}) está impulsionando a novidade (+{saturation_novelty_boost:.2f}).")
                logger.info(
                    f"MCL Saturation: Saturação de {saturation:.2f} resultou em boost de novidade de {saturation_novelty_boost:.2f}.")

        # O gatilho de repetição aguda agora é o único "force" que resta (além do Xi)
        should_force_novelty = (is_repetitive and not user_explicitly_insists)

        if should_force_novelty:
            params['novelty_bias'] = 0.90
            params['coherence_bias'] = 0.10
            consistency_effect = -0.2
            reasons.append("Anti-Loop (Agudo): Forçando novidade para quebrar repetição imediata.")
            mcl_params["reason"] = "Repetição aguda detectada."
            mcl_params["cognitive_state_name"] = "CREATIVE_BREATHING"
            mcl_params["operational_advice_for_ora"] = (
                "ALERTA: A conversa está se tornando repetitiva. Sua tarefa principal é introduzir uma nova perspectiva, fazer uma pergunta de aprofundamento ou conectar o tópico atual a um conceito relacionado. EVITE repetir informações já fornecidas."
            )
        else:
            mcl_params["cognitive_state_name"] = cognitive_state_name
            mcl_params["reason"] = params.get("reason", "Operação padrão.")

        # --- APLICAÇÃO FINAL DOS EFEITOS ---
        # Apenas os boosts de curiosidade e saturação são aplicados aqui.
        params['novelty_bias'] = current_novelty + curiosity_effect + saturation_novelty_boost
        params['coherence_bias'] = current_coherence + consistency_effect

        # Normalização final dos biases
        total_bias = params['novelty_bias'] + params['coherence_bias']
        if total_bias > 0:
            params['novelty_bias'] = max(0.05, min(0.95, params['novelty_bias'] / total_bias))
            params['coherence_bias'] = 1.0 - params['novelty_bias']
        else:
            params['novelty_bias'] = 0.5
            params['coherence_bias'] = 0.5

        logger.critical(
            f"MCL Biases FINAIS -> Coherence={params['coherence_bias']:.2f}, Novelty={params['novelty_bias']:.2f}")

        # 4. Empacotamento e retorno
        mcl_params["biases"] = {"coherence_bias": params['coherence_bias'], "novelty_bias": params['novelty_bias']}
        mcl_params["drives_state_at_turn"] = drives.model_dump()
        mcl_params["mcl_analysis"]["agency_score"] = agency_score
        mcl_params["topic_shifted_this_turn"] = topic_shifted
        if safety_avoidance_vector:
            guidance_packet.safety_avoidance_vector = safety_avoidance_vector

        if drives and body_state:
            logger.info("MCLEngine: Gerando relatório fenomenológico...")
            enriched_drives, enriched_body_state = await self._generate_phenomenological_report(
                drives, body_state, chat_history
            )
            mcl_params["enriched_drives"] = enriched_drives
            mcl_params["enriched_body_state"] = enriched_body_state

        predicted_state = await prediction_task
        mcl_params["predicted_internal_state"] = predicted_state

        log_msg = f"MCL: Final guidance -> State: {mcl_params.get('cognitive_state_name')}. Score: {agency_score:.2f}"
        logger.info(log_msg + f" | Razões: {', '.join(reasons)}")

        return guidance_packet, mcl_params

    # <<< ESTA É A FUNÇÃO AUXILIAR SÍNCRONA QUE ESTAVA FALTANDO >>>
    def _get_guidance_sync(self, user_model: 'UserRepresentation', cognitive_state: CognitiveStatePacket,
                           body_state: Optional['VirtualBodyState'] = None) -> Tuple[
        GuidancePacket, Dict[str, Any]]:
        """
        Executa a parte síncrona da análise do MCL.
        Calcula os vetores de coerência/novidade e os parâmetros iniciais de agência e LLM.
        """
        logger.info("MCLEngine (Sync Part): Gerando scores base e parâmetros...")
        query_text = cognitive_state.original_intent.query_vector.source_text or ""

        # --- 1. Análise de Intenção e Agência ---
        agency_score = 0.0
        reasons = []

        # Verifica por palavras-chave que forçam um pensamento mais profundo
        if any(keyword in query_text.lower() for keyword in self.agency_force_keywords):
            agency_score += 5.0
            reasons.append("Palavras-chave de 'reflexão profunda' detectadas na query.")

        if body_state:
            fatigue_penalty = body_state.cognitive_fatigue * 4.0  # Penalidade pode ser até -4.0
            agency_score -= fatigue_penalty
            if fatigue_penalty > 0.5:  # Só loga se a penalidade for significativa
                reasons.append(
                    f"Penalidade de Fadiga: Agência reduzida em {fatigue_penalty:.2f} devido ao cansaço cognitivo.")
                logger.warning(
                    f"MCL: Fadiga Cognitiva ({body_state.cognitive_fatigue:.2f}) está suprimindo a agência em {fatigue_penalty:.2f} pontos.")

        if user_model:
            # Modulação pelo Nível de Conhecimento do Usuário
            if user_model.knowledge_level == "expert":
                knowledge_bonus = 2.0
                agency_score += knowledge_bonus
                reasons.append(
                    f"Bônus de Usuário Expert: Agência aumentada em {knowledge_bonus:.2f} para uma análise mais profunda.")
                logger.info(f"MCL: Usuário expert detectado. Aumentando agência em {knowledge_bonus:.2f}.")

            # Modulação pelo Estado Emocional do Usuário
            if user_model.emotional_state in ["frustrated", "confused", "impatient"]:
                emotion_penalty = 3.0
                agency_score -= emotion_penalty
                reasons.append(
                    f"Penalidade de Emoção do Usuário: Agência reduzida em {emotion_penalty:.2f} para fornecer uma resposta mais direta.")
                logger.warning(
                    f"MCL: Usuário frustrado/confuso detectado. Suprimindo agência em {emotion_penalty:.2f} para priorizar clareza.")

            # Modulação pelo Estilo de Comunicação
            if user_model.communication_style == "direct":
                style_penalty = 1.5
                agency_score -= style_penalty
                reasons.append(
                    f"Penalidade de Estilo Direto: Agência reduzida em {style_penalty:.2f} para corresponder ao estilo do usuário.")
                logger.info(f"MCL: Estilo de comunicação direto detectado. Reduzindo agência em {style_penalty:.2f}.")

        # Verifica por palavras-chave que sugerem um pensamento mais profundo
        elif any(keyword in query_text.lower() for keyword in self.agency_suggestion_keywords):
            agency_score += 2.0
            reasons.append("Palavras-chave de 'explicação' detectadas na query.")

        # Verifica a descrição da intenção gerada pelo HTGTranslator
        intent_description = (
                    cognitive_state.original_intent.intent_vector.source_text or "") if cognitive_state.original_intent.intent_vector else ""
        if any(keyword in intent_description.lower() for keyword in self.deep_intent_keywords):
            agency_score += 4.0
            reasons.append(f"Intenção semântica profunda detectada: '{intent_description}'.")

        # Ajusta a agência com base no comprimento da query
        word_count = len(query_text.split())
        agency_score += min(word_count / 5.0, 5.0)  # Adiciona até 3 pontos por comprimento
        if word_count > 10:
            reasons.append(f"Query com {word_count} palavras sugere maior complexidade.")

        # --- 2. Determinação de Parâmetros Emergentes (disclosure, temp, tokens) ---
        disclosure_level = self.agent_config.get("self_disclosure_level", "moderate")
        introductory_keywords = ["quem é você", "se apresente", "me fale sobre você", "se descreva"]
        self_referential_keywords = ["você", "sua", "seu", "sua opinião", "o que você pensa", "gosta de fazer"]
        is_introductory = any(keyword in query_text.lower() for keyword in introductory_keywords)
        is_self_referential = any(keyword in query_text.lower() for keyword in self_referential_keywords)

        if is_introductory:
            disclosure_level = "high"

        # Orçamento de tokens e temperatura dinâmicos
        max_tokens = 2000
        temperature = 0.6
        if is_self_referential or is_introductory:
            max_tokens = 2500
            temperature = 0.7
            reasons.append(f"Query auto-referencial -> Aumentando max_tokens ({max_tokens}).")
        elif agency_score > 6.0 or word_count > 25:
            max_tokens = 4000
            temperature = 0.75
            reasons.append("Alta agência/comprimento -> Aumentando max_tokens significativamente.")
        elif agency_score > 3.0 or word_count > 10:
            max_tokens = 3000
            temperature = 0.65
            reasons.append("Média agência/comprimento -> Aumentando max_tokens moderadamente.")
        else:
            reasons.append(f"Baixa agência/comprimento -> Usando max_tokens base ({max_tokens}).")

        logger.info(f"MCL Controle Emergente: Temp={temperature}, MaxTokens={max_tokens}. Razão: {reasons[-1]}")

        # --- 3. Cálculo dos Vetores de Orientação (Coerência vs. Novidade) ---
        all_context_vectors = [cognitive_state.identity_vector] + cognitive_state.relevant_memory_vectors
        valid_vectors = [gv.vector for gv in all_context_vectors if gv and gv.vector]
        query_vec = np.array(cognitive_state.original_intent.query_vector.vector)

        if not valid_vectors:
            coherence_vector_norm = self._normalize_vector(query_vec)
            novelty_vector_norm = self._normalize_vector(np.random.rand(len(query_vec)) * 2 - 1)
        else:
            center_of_mass_vector = np.mean(valid_vectors, axis=0)
            if np.linalg.norm(center_of_mass_vector) == 0:
                projection = np.zeros_like(query_vec)
            else:
                projection = np.dot(query_vec, center_of_mass_vector) / np.dot(center_of_mass_vector,
                                                                               center_of_mass_vector) * center_of_mass_vector

            novelty_vector = query_vec - projection
            coherence_vector_norm = self._normalize_vector(center_of_mass_vector)
            novelty_vector_norm = self._normalize_vector(novelty_vector)

        guidance_packet = GuidancePacket(
            coherence_vector=GenlangVector(vector=coherence_vector_norm.tolist(),
                                           source_text="Centro de massa do contexto", model_name="mcl_internal_v3"),
            novelty_vector=GenlangVector(vector=novelty_vector_norm.tolist(),
                                         source_text="Componente da query ortogonal ao contexto",
                                         model_name="mcl_internal_v3")
        )

        # --- 4. Empacotamento do Resultado ---
        mcl_params = {
            "cognitive_state_name": "TEMP",  # Será definido na função assíncrona
            "reason": "TEMP",  # Será definido na função assíncrona
            "ora_parameters": {"temperature": temperature, "max_tokens": max_tokens},
            "agency_parameters": {"use_agency_simulation": False},  # Será definido depois
            "mcl_analysis": {"agency_score": agency_score, "reasons": reasons},
            "biases": {"coherence_bias": 0.7, "novelty_bias": 0.3},  # Default, será sobrescrito
            "disclosure_level": disclosure_level,
            "agent_name": self.agent_config.get("name", "Aura AI")
        }

        return guidance_packet, mcl_params