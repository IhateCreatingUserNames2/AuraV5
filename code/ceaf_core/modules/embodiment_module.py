# ceaf_core/modules/embodiment_module.py
import time
import logging
import numpy as np
from typing import Dict, Any
from ceaf_core.genlang_types import VirtualBodyState
from ceaf_core.models import BodyConfig
from ceaf_core.services.state_manager import StateManager

logger = logging.getLogger("CEAFv3_Embodiment")


class EmbodimentModule:
    """
    [V5 UPGRADE] Fisiologia Virtual Não-Linear.
    Impede que a fadiga trave em 1.0 usando curvas logísticas e micro-recuperação.
    """

    def __init__(self, config: BodyConfig = None):
        self.config = config or BodyConfig()
        self.state_manager = StateManager()

    async def process_turn_effects(self, agent_id: str, metrics: Dict[str, Any]) -> VirtualBodyState:
        # 1. Fetch State
        current_state = await self.state_manager.get_body_state(agent_id)

        # 2. Apply Biological Logic
        updated_state = self._calculate_state_update(current_state, metrics)

        # 3. Persist State
        await self.state_manager.save_body_state(agent_id, updated_state)

        return updated_state

    def _calculate_state_update(self, body_state: VirtualBodyState, metrics: Dict[str, Any]) -> VirtualBodyState:
        updated_state = body_state.model_copy(deep=True)

        # --- 1. Recuperação Passiva (O tempo passou) ---
        current_time = time.time()
        time_delta_seconds = current_time - updated_state.last_updated
        # Converter para horas, mas garantir um mínimo de recuperação por turno (Micro-sleep de 5s)
        effective_delta_hours = max(5.0, time_delta_seconds) / 3600.0

        fatigue_recovery = self.config.fatigue_recovery_rate * effective_delta_hours
        saturation_recovery = self.config.saturation_recovery_rate * effective_delta_hours

        # Aplica recuperação ANTES do novo estresse
        updated_state.cognitive_fatigue = max(0.0, updated_state.cognitive_fatigue - fatigue_recovery)
        updated_state.information_saturation = max(0.0, updated_state.information_saturation - saturation_recovery)

        # --- 2. Acumulação de Fadiga (Curva Logística) ---
        # Strain vem do Xi (Tensão). Se Xi for alto, cansa.
        strain = metrics.get("cognitive_strain", 0.0)

        # Multiplicador base reduzido para sanidade (de 0.3 para 0.15)
        base_impact = strain * 0.15

        # Fator de Resistência: Quanto mais cansado, menos o novo estresse impacta (para não travar em 1.0)
        # Se fadiga é 0.0, resistência é 0. Se fadiga é 0.9, resistência é alta.
        resistance = updated_state.cognitive_fatigue
        actual_increase = base_impact * (1.0 - resistance)  # <--- AQUI ESTÁ A MÁGICA SIGMOIDE

        updated_state.cognitive_fatigue += actual_increase

        # --- 3. Saturação (Memórias Novas) ---
        new_memories = metrics.get("new_memories_created", 0)
        saturation_increase = new_memories * 0.05 * (1.0 - updated_state.information_saturation)
        updated_state.information_saturation += saturation_increase

        # --- 4. Finalização ---
        updated_state.cognitive_fatigue = max(0.0, min(1.0, updated_state.cognitive_fatigue))
        updated_state.information_saturation = max(0.0, min(1.0, updated_state.information_saturation))
        updated_state.last_updated = current_time

        if updated_state.cognitive_fatigue > 0.9:
            logger.warning(f"⚠️ EMBODIMENT: Fadiga Crítica ({updated_state.cognitive_fatigue:.2f}).")

        return updated_state