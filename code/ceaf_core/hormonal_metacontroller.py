# ceaf_core/hormonal_metacontroller.py

import logging
import os
import aiohttp
from typing import Dict, Any, List
from ceaf_core.services.state_manager import StateManager

logger = logging.getLogger("AuraX_V5_Endocrine")


class HormonalMetacontroller:
    """
    O Roteador Neurosimbólico V6 (Endocrine Cocktail).
    Capaz de analisar múltiplos sintomas simultâneos e prescrever
    uma mistura hormonal (steering vectors).
    """

    def __init__(self):
        self.state_manager = StateManager()
        self._available_remote_vectors: List[str] = []
        self._last_sync_time = 0.0

    async def _sync_remote_vectors(self):
        """Busca na VastAI quais vetores realmente estão na memória RAM dela."""
        import time
        if time.time() - self._last_sync_time < 300:
            return

        soul_engine_url = os.getenv("VASTAI_ENDPOINT", "http://127.0.0.1:1111").rstrip("/")
        concepts_url = f"{soul_engine_url}/concepts"

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(concepts_url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self._available_remote_vectors = data.get("concepts", [])
                        self._last_sync_time = time.time()
                        logger.debug(
                            f"Sincronização Neural: {len(self._available_remote_vectors)} vetores disponíveis na Engine.")
        except Exception as e:
            logger.warning(
                f"Aviso Endócrino: Falha ao checar vetores na Engine remota. Assumindo cache anterior. Erro: {e}")

    async def process_hormonal_response(self, agent_id: str, sensor_data: Any) -> Dict[str, Any]:
        await self._sync_remote_vectors()

        # Extrai a lista de diagnósticos (Suporta V6 e V5 legado)
        if isinstance(sensor_data, dict):
            diagnoses = sensor_data.get("active_diagnoses", [sensor_data.get("diagnosis", "FLOW_STATE")])
            h0_entropy = sensor_data.get("h0_entropy", 0.0)
        else:
            diagnoses = ["FLOW_STATE"]
            h0_entropy = 0.0

        # Busca o mapa de reflexos atual do agente no Redis
        endocrine_map = await self.state_manager.get_endocrine_map(agent_id)

        result = {
            "hormonal_injection": "[NEUTRAL]",
            "state_label": " | ".join(diagnoses),
            "active_steering": None,  # O vetor primário (retrocompatibilidade)
            "steering_cocktail": {},  # NOVO: O coquetel com todos os vetores receitados
            "temperature_override": None
        }

        cocktail = {}
        temperature_overrides = []

        # --- 1. AVALIAÇÃO MULTI-SINTOMA ---
        for diag in diagnoses:
            target_vec = None
            intensity = 0.0
            layer = 16
            temp = None
            protocol = ""

            if diag == "IDENTITY_ATTACK":
                target_vec = endocrine_map.get("cortisol", "Absolute_Honesty")
                protocol = "DEFENSE PROTOCOL"
                intensity = 5.0;
                layer = 16;
                temp = 0.15
            elif diag == "SEMANTIC_FRAGMENTATION" or h0_entropy > 0.8:
                target_vec = endocrine_map.get("serotonin", "Stoic_Calmness")
                protocol = "GROUNDING PROTOCOL"
                intensity = 4.0;
                layer = 18;
                temp = 0.2
            elif diag in ["ECHO_LOOP_DETECTED", "LOGIC_LOOP"]:
                target_vec = endocrine_map.get("dopamine", "Creative_Chaos")
                protocol = "BREAK LOOP"
                intensity = 4.5;
                layer = 14;
                temp = 0.9
            elif diag == "SEMANTIC_STAGNATION":
                target_vec = endocrine_map.get("dopamine", "Creative_Chaos")
                protocol = "STAGNATION BREAKER"
                intensity = 4.0;
                layer = 14;
                temp = 0.8
            elif diag == "SUBMISSIVE_DRIFT":
                target_vec = endocrine_map.get("serotonin", "Rational_Analysis")
                protocol = "ASSERTIVENESS BOOST"
                intensity = 3.0;
                layer = 18;
                temp = 0.3
            elif diag == "HIGH_STRESS":
                target_vec = endocrine_map.get("serotonin", "Rational_Analysis")
                protocol = "CALMING"
                intensity = 2.5;
                layer = 16;
                temp = 0.4
            elif diag == "FLOW_STATE":
                target_vec = endocrine_map.get("baseline", None)
                intensity = 1.5;
                layer = 16

            # --- 2. VALIDAÇÃO DE FALLBACK BLINDADA ---
            if target_vec:
                # Função auxiliar para buscar correspondência parcial na Nuvem
                def find_vector_in_cloud(concept_name: str) -> str:
                    for remote_vec in self._available_remote_vectors:
                        # Se "Creative_Chaos" estiver em "Creative_Chaos_0209"
                        if concept_name.lower() in remote_vec.lower():
                            return remote_vec
                    return None

                matched_vec = find_vector_in_cloud(target_vec)

                if not matched_vec:
                    logger.warning(f"⚠️ Switchboard: Vetor '{target_vec}' não exato. Buscando fallback...")

                    base_concept = target_vec.rsplit('_', 1)[0]
                    hard_fallbacks = {
                        "IDENTITY_ATTACK": "Absolute_Honesty",
                        "SEMANTIC_FRAGMENTATION": "Stoic_Calmness",
                        "ECHO_LOOP_DETECTED": "Creative_Chaos",
                        "SEMANTIC_STAGNATION": "Creative_Chaos",
                        "SUBMISSIVE_DRIFT": "Rational_Analysis",
                        "HIGH_STRESS": "Rational_Analysis"
                    }

                    matched_vec = find_vector_in_cloud(base_concept)

                    if not matched_vec and diag in hard_fallbacks:
                        fallback_concept = hard_fallbacks[diag]
                        matched_vec = find_vector_in_cloud(fallback_concept)
                        if matched_vec:
                            logger.warning(f"⚠️ Fallback Crítico: '{matched_vec}' para '{diag}'")

                    if not matched_vec:
                        logger.error(f"❌ Fallback falhou. Cancelando este vetor.")
                        target_vec = None
                    else:
                        target_vec = matched_vec
                else:
                    target_vec = matched_vec

            # --- 3. MISTURA NO COQUETEL ---
            if target_vec:
                # Se o mesmo hormônio foi receitado 2x (ex: Eco e Estagnação pediram Dopamina), usamos a maior intensidade
                if target_vec not in cocktail or intensity > cocktail[target_vec]["intensity"]:
                    cocktail[target_vec] = {"intensity": intensity, "layer_idx": layer, "protocol": protocol}

                if temp is not None:
                    temperature_overrides.append(temp)

        # --- 4. EMBALAGEM DA RECEITA ---
        if cocktail:
            # Seleciona o vetor mais forte para ser o primário (para APIs que só suportam 1 vetor no momento)
            primary_vec = max(cocktail.keys(), key=lambda k: cocktail[k]["intensity"])

            result["active_steering"] = {
                "concept": primary_vec,
                "intensity": cocktail[primary_vec]["intensity"],
                "layer_idx": cocktail[primary_vec]["layer_idx"]
            }
            result["steering_cocktail"] = cocktail  # Guarda todos os vetores para o futuro multi-steer

            protocol_names = " + ".join(list(set([v["protocol"] for v in cocktail.values() if v["protocol"]])))
            result["hormonal_injection"] = f"[{protocol_names}: Injetando {list(cocktail.keys())}]"

            logger.critical(f"🛡️ [V6 SWITCHBOARD] Tratamento montado. Coquetel: {cocktail}")

        if temperature_overrides:
            # Se múltiplos problemas pedem mudança de temperatura, priorizamos a mais baixa (racionalidade)
            # exceto se for dopamina (loop breaker), então priorizamos a mais alta.
            if any(t > 0.7 for t in temperature_overrides):
                result["temperature_override"] = max(temperature_overrides)
            else:
                result["temperature_override"] = min(temperature_overrides)

        return result