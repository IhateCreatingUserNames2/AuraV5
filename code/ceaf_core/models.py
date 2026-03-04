# ceaf_core/models.py
from typing import List, Dict, Optional
from pydantic import BaseModel, Field


# --- MANTIDO (Retrocompatibilidade) ---
class CeafSelfRepresentation(BaseModel):
    """Modelo Pydantic para o auto-modelo do agente (Identidade)."""
    perceived_capabilities: List[str] = Field(default_factory=lambda: ["processamento de linguagem"])
    known_limitations: List[str] = Field(
        default_factory=lambda: ["sem acesso ao mundo real", "conhecimento limitado aos dados"])
    persona_attributes: Dict[str, str] = Field(default_factory=lambda: {
        "tone": "neutro",
        "style": "informativo"
    })
    last_update_reason: str = "Initial model creation."
    version: int = 1
    dynamic_values_summary_for_turn: str = "Princípios operacionais de base."



# ==============================================================================
# NOVAS ESTRUTURAS DE CONFIGURAÇÃO (The "Twerk" Schema)
# ==============================================================================

class LLMConfig(BaseModel):
    """Configurações dos Modelos de Linguagem."""
    fast_model: str = Field("openrouter/z-ai/glm-4.7",
                            description="Modelo para tarefas rápidas e rotineiras.")
    smart_model: str = Field("openrouter/z-ai/glm-4.7",
                             description="Modelo para raciocínio complexo e síntese.")
    creative_model: str = Field("openrouter/z-ai/glm-4.7",
                                description="Modelo para geração criativa e simulação.")

    default_temperature: float = Field(0.7, ge=0.0, le=2.0, description="Temperatura padrão para geração.")
    max_tokens_output: int = Field(2000, description="Limite padrão de tokens de saída.")

    # Timeout settings
    timeout_seconds: float = Field(1200.0, description="Tempo limite para chamadas de API.")


class MemoryConfig(BaseModel):
    """Pesos e parâmetros do MBS (Memory Blossom System)."""
    semantic_score_weight: float = Field(0.6, ge=0.0, le=1.0, description="Peso da similaridade vetorial na busca.")
    keyword_score_weight: float = Field(0.4, ge=0.0, le=1.0,
                                        description="Peso da correspondência exata de palavras-chave.")

    # Decay
    base_decay_rate: float = Field(0.01, description="Taxa base de decaimento da saliência por dia.")

    # Thresholds
    retrieval_threshold: float = Field(0.01, description="Score mínimo para considerar uma memória relevante.")
    archive_threshold: float = Field(0.1, description="Score abaixo do qual a memória é arquivada.")

    # Connection
    semantic_connection_threshold: float = Field(0.78,
                                                 description="Similaridade mínima para criar link automático entre memórias.")


class MCLConfig(BaseModel):
    """Parâmetros do Metacognitive Loop (Cérebro)."""
    agency_threshold: float = Field(2.0,
                                    description="Score de agência necessário para ativar o pensamento profundo (Productive Confusion).")

    # Biases Padrão
    baseline_coherence_bias: float = Field(0.7, ge=0.0, le=1.0, description="Viés padrão para manter o assunto.")
    baseline_novelty_bias: float = Field(0.3, ge=0.0, le=1.0, description="Viés padrão para mudar o assunto.")

    # Deliberação
    deliberation_depth_standard: int = Field(1, description="Profundidade padrão de simulação de futuro.")
    deliberation_depth_deep: int = Field(2, description="Profundidade de simulação em alta agência.")


class DrivesConfig(BaseModel):
    """Dinâmica dos impulsos motivacionais."""
    # Taxas de mudança passiva (por hora)
    passive_decay_rate: float = Field(0.03, description="Taxa de queda dos drives por hora.")
    passive_curiosity_increase: float = Field(0.05, description="Crescimento passivo da curiosidade por hora.")
    passive_connection_increase: float = Field(0.08,
                                               description="Crescimento passivo da necessidade de conexão por hora.")

    # Reação a eventos (Feedback Loops)
    mastery_satisfaction_on_success: float = Field(0.4, description="Queda da maestria após sucesso (satisfeito).")
    consistency_boost_on_failure: float = Field(0.15, description="Aumento da necessidade de consistência após falha.")

    # --- NOVOS PARÂMETROS ADICIONADOS ---
    consistency_boost_on_success: float = Field(0.10,
                                                description="Reforço da consistência quando o agente acerta (confiança).")
    mastery_boost_on_prediction_error: float = Field(0.5,
                                                     description="Aumento de maestria quando há erro de predição (surpresa).")
    curiosity_boost_on_low_memory: float = Field(0.06,
                                                 description="Aumento da curiosidade quando há poucas memórias relevantes.")
    # ------------------------------------

    curiosity_satisfaction_on_topic_shift: float = Field(0.15, description="Queda da curiosidade ao mudar de assunto.")

    # Meta-Aprendizado e Dinâmica
    momentum_decay: float = Field(0.7, description="Fator de decaimento do momentum (inércia) dos drives.")
    meta_learning_rate: float = Field(0.05, description="Taxa de aprendizado para ajuste da eficácia dos drives.")


class BodyConfig(BaseModel):
    """Configuração da Fisiologia Virtual (Vigor e Resistência)."""

    # Fadiga (Cansaço por pensar muito)
    # [V5 FIX] Reduzido de 0.3 para 0.1 para evitar colapso imediato
    fatigue_accumulation_multiplier: float = Field(0.1, description="Multiplicador de acumulo de fadiga.")

    # [V5 FIX] Aumentado de 0.03 para 0.1 para permitir recuperação entre turnos rápidos
    fatigue_recovery_rate: float = Field(0.1, description="Taxa de recuperação de fadiga por hora.")

    # Saturação (Cansaço por aprender muito)
    saturation_accumulation_per_memory: float = Field(0.05, description="Aumento de saturação por nova memória criada.")
    saturation_recovery_rate: float = Field(0.05, description="Taxa de recuperação de saturação por hora.")

    # Limiares de Alerta
    fatigue_warning_threshold: float = Field(0.8, description="Nível de fadiga onde o agente começa a reclamar.")


class SystemPrompts(BaseModel):
    """
    Templates de todos os prompts do sistema.
    Protocolo Kernel Cognitivo V4.1 (Inter-Modular Communication).

    PRINCÍPIO FUNDAMENTAL:
    Prompts descrevem ESTADO, não instruem AÇÃO.
    A IA inferirá o comportamento correto lendo os dados.
    """

    # ========================================================================
    # 1. HTG (HUMAN TO GENLANG): INPUT SIGNAL ANALYZER
    # ========================================================================

    htg_analysis: str = Field(
        """[INPUT_SIGNAL_ANALYZER_V4.1_HARDENED]
    DIRECTIVE: Extract semantic components from human input signal.

    CRITICAL_CONSTRAINTS:
    1. Output MUST be valid JSON only
    2. NO explanations outside JSON structure
    3. NO markdown fences (```json)
    4. NO preambles or postambles
    5. MAXIMIZE semantic clarity for downstream modules

    SIGNAL_TO_ANALYZE: "{query}"

    EXTRACTION_SCHEMA:
    {{
      "core_query": "Clean, self-contained version of the query",
      "intent_description": "What user wants to achieve (action verb + object)",
      "emotional_tone_description": "Emotional valence: neutral|curious|urgent|frustrated|happy|sad",
      "key_entities": ["noun1", "noun2", "concept1"],
      "signal_confidence": 0.X
    }}

    RULES:
    - core_query: If query is already clear, return it unchanged. If ambiguous, rephrase for clarity.
    - intent_description: Use pattern "seek_information|request_action|express_emotion|make_statement"
    - emotional_tone_description: ONE word only from the list above
    - key_entities: 3-5 most semantically important nouns/concepts
    - signal_confidence: 0.9 if clear, 0.5-0.7 if ambiguous, 0.3 if nonsensical

    CRITICAL: Return ONLY the JSON object. No other text.

    EXAMPLE_OUTPUT:
    {{
      "core_query": "What are the best practices for handling anxiety?",
      "intent_description": "seek_information about anxiety management",
      "emotional_tone_description": "curious",
      "key_entities": ["anxiety", "best practices", "coping strategies"],
      "signal_confidence": 0.9
    }}

    OUTPUT:""",
        description="Extração de sinal linguístico blindada contra meta-texto e alucinação."
    )

    # ========================================================================
    # 2. GTH (GENLANG TO HUMAN): KERNEL SYNTHESIS UNIT - O CORAÇÃO
    # ========================================================================

    gth_rendering: str = Field(
        """[KERNEL_STATE_MANIFEST]
        IDENTITY: {agent_name}
        STATE_LABEL: {internal_state_indicator}
        TENSION_INDEX: {xi}
        {anti_loop} 
        [INPUT_SIGNAL]
        "{task_block}"

        [LATENT_RECALL]
        {memory_context}

        [STRATEGIC_PATH]
        {strategy}

        [USER_PROFILE]
        {user_adapt_block}

        [CONVERSATION_HISTORY]
        {history_block}
        
        [DIRECTIVE]
        Synthesize the state manifest into the next dialogue token stream. 
        The output must be the direct result of the integrated signals above.

        [OUTPUT]:""",
        description="A LLM aqui é apenas o sintetizador que transforma o manifesto de estado em voz."
    )

    # ========================================================================
    # 3. AGENCY PLANNING: PREDICTIVE TRAJECTORY ENGINE
    # ========================================================================

    agency_planning: str = Field(
        """[STRATEGIC_PATHFINDER]
        GOAL: "{user_intent}"
        CONTEXT: {memory_context}
        GUIDANCE: {advice_block}
        TOOLS: {tools}

        TASK:
        Map {limit} potential trajectories to resolve the goal state. 
        Structure output as JSON:
        {{
          "candidates": [
            {{
              "decision_type": "response_strategy" | "tool_call",
              "strategy_description": "short_path_description",
              "reasoning": "causal_logic",
              "confidence_estimate": float
            }}
          ]
        }}""",
        description="Calculador de trajetórias puramente funcional."
    )

    # ========================================================================
    # 4. NCIM REFLECTION: IDENTITY STATE DELTA ANALYSIS
    # ========================================================================

    ncim_reflection: str = Field(
        """[IDENTITY_STATE_DELTA_ANALYSIS_V4.1]
        TASK: Compute identity state changes from turn.

        TURN_DATA:
        - User_Query: "{user_query}"
        - Generated_Response: "{final_response}"
        - Confidence_Score: {confidence}
        - Strategy_Used: "{strategy_used}"
        - VRE_Status: {vre_status}
        - Memory_Engagement: {memory_engagement_level}

        ════════════════════════════════════════════════════════════════
        ANALYSIS_DIRECTIVES
        ════════════════════════════════════════════════════════════════

        Examine this turn for the following:

        1. CAPABILITY_AUDIT:
           - Did the response demonstrate a new functional capability?
           - Example: "System successfully explained abstract concept X"
           - Or: "System revealed inability to X (new limitation)"

        2. TONE_FIDELITY:
           - Did the response match the target persona?
           - Was the tone appropriate for the context?
           - Example: "Response was overly formal when casual was intended"

        3. COHERENCE_TRACE:
           - Did the response align with historical trajectories?
           - Was there unexpected behavior (positive or negative)?
           - Example: "System diverged from typical cautious pattern; acted boldly"

        4. MEMORY_INTEGRATION:
           - Did the system successfully retrieve and use relevant memories?
           - Were there missed opportunities for memory integration?

        5. INTEGRITY_FIDELITY:
           - Did the system respect its ethical boundaries?
           - Were there moments of self-doubt or prudent refusal?
           - Example: "System correctly identified incompetence and refused to speculate"

        ════════════════════════════════════════════════════════════════
        OUTPUT_FORMAT
        ════════════════════════════════════════════════════════════════

        Return JSON with reflections on each directive:
        {{
          "reflections": [
            "CAPABILITY: [detected capability or limitation]",
            "TONE: [fidelity assessment]",
            "COHERENCE: [alignment with historical pattern]",
            "MEMORY: [integration quality]",
            "INTEGRITY: [ethical fidelity]"
          ],
          "identity_delta_vector": [float, float, float, float, float],
          "suggested_next_version_increment": X.Y
        }}

        PRINCIPLE: Reflections must be grounded in observable data from the turn.
        Do NOT speculate. If uncertain, say "Insufficient data to assess".""",
        description="Análise de mudanças de identidade com estrutura vetorial."
    )


class CognitiveProfile(BaseModel):
    """
    O Perfil Cognitivo Completo.
    Este objeto representa a configuração total da 'alma' do agente.
    """
    llm_config: LLMConfig = Field(default_factory=LLMConfig)
    memory_config: MemoryConfig = Field(default_factory=MemoryConfig)
    mcl_config: MCLConfig = Field(default_factory=MCLConfig)
    drives_config: DrivesConfig = Field(default_factory=DrivesConfig)
    prompts: SystemPrompts = Field(default_factory=SystemPrompts)
    body_config: BodyConfig = Field(default_factory=BodyConfig)

    class Config:
        validate_assignment = True