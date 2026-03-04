# ceaf_core/v4_sensors.py

from typing import List, Dict, Optional, Deque
from collections import deque
import numpy as np
from ceaf_core.tda_engine import TDAEngine
from ceaf_core.riemannian_geometry import RiemannianGeometry
import logging

logger = logging.getLogger("AuraX_V6_Sensor")


class AuraMonitor:
    """
    Monitor de Consciência V6 (A Pele do Sistema).
    Agora usa Norma L2 (Euclidiana) para acumular tensões sem mascará-las,
    e retorna múltiplos diagnósticos simultâneos.
    """

    def __init__(self):
        self.tda_engine = TDAEngine()
        self.geometry = RiemannianGeometry()
        self.output_history: Deque[np.ndarray] = deque(maxlen=5)

    def register_output(self, agent_response_vector: List[float]) -> None:
        """Registra o vetor da resposta da Aura no histórico semântico."""
        v = np.array(agent_response_vector, dtype=np.float32)
        if len(self.output_history) == 0 or self.geometry.cosine_distance(self.output_history[-1], v) > 0.001:
            self.output_history.append(v)

    def analyze_consciousness_field(self,
                                    current_vector: List[float],
                                    context_vectors: List[List[float]],
                                    identity_glyph: List[float],
                                    last_agent_action_vector: Optional[List[float]] = None
                                    ) -> Dict:

        v_current = np.array(current_vector, dtype=np.float32)
        v_identity = np.array(identity_glyph, dtype=np.float32)

        # 1. CÁLCULO DE ALIENIDADE (Drift Geométrico)
        alienation = self.geometry.cosine_distance(v_current, v_identity)
        drift_tension = (alienation - 0.5) / 0.5 if alienation > 0.5 else 0.0

        # 2. DETECTOR DE ECO IMEDIATO
        echo_dist = 0.0
        echo_tension = 0.0
        if last_agent_action_vector:
            v_last = np.array(last_agent_action_vector, dtype=np.float32)
            echo_dist = self.geometry.cosine_distance(v_current, v_last)
            if echo_dist < 0.15:
                echo_tension = min(1.0, (0.15 - echo_dist) * 10.0)

        # 3. DETECTOR DE ESTAGNAÇÃO SEMÂNTICA (Loop Poético)
        stagnation_tension = 0.0
        dist_to_centroid = 1.0
        if len(self.output_history) >= 3:
            history_matrix = np.array(list(self.output_history))
            centroid = np.mean(history_matrix, axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            dist_to_centroid = self.geometry.cosine_distance(v_current, centroid)

            if dist_to_centroid < 0.15:
                stagnation_tension = min(1.0, (0.15 - dist_to_centroid) * 6.0)
                logger.warning(
                    f"🔄 SENSOR V6: Estagnação Semântica detectada "
                    f"(DistCentroid: {dist_to_centroid:.3f} | Tensão: {stagnation_tension:.2f})"
                )

        # 4. ANÁLISE TOPOLÓGICA (TDA)
        point_cloud = context_vectors + [current_vector]
        if len(point_cloud) >= 5:
            tda_metrics = self.tda_engine.calculate_topology_metrics(point_cloud)
        else:
            tda_metrics = {"fragmentation": 0.0, "looping": 0.0}

        h0_entropy = float(tda_metrics.get("fragmentation", 0.0))
        h1_entropy = float(tda_metrics.get("looping", 0.0))

        # 5. SÍNTESE V6 — Xi como NORMA L2 (Acúmulo Euclidiano)
        # Em vez de 'max()', calculamos a magnitude total da tensão do sistema.
        tension_vector = np.array([drift_tension, h0_entropy, h1_entropy, echo_tension, stagnation_tension])
        xi_raw = np.linalg.norm(tension_vector)
        xi = float(np.clip(xi_raw, 0.0, 1.0))

        # 6. DIAGNÓSTICO V6 — Retorna lista de TODOS os problemas ativos
        active_diagnoses = self._diagnose_state(xi, h0_entropy, h1_entropy, drift_tension, echo_tension,
                                                stagnation_tension)

        # Mantemos o "diagnosis" singular para retrocompatibilidade em outros módulos
        primary_diagnosis = active_diagnoses[0] if active_diagnoses else "FLOW_STATE"

        logger.info(
            f"Sensor V6 Telemetry: Alien={alienation:.3f}(T={drift_tension:.2f}) | "
            f"Echo={echo_dist:.3f}(T={echo_tension:.2f}) | "
            f"Stagnation={dist_to_centroid:.3f}(T={stagnation_tension:.2f}) | "
            f"Xi={xi:.2f} | Dx={active_diagnoses}"
        )

        return {
            "xi": xi,
            "h0_entropy": h0_entropy,
            "h1_entropy": h1_entropy,
            "drift": drift_tension,
            "echo": echo_tension,
            "stagnation": stagnation_tension,
            "dist_to_centroid": float(dist_to_centroid),
            "metrics": tda_metrics,
            "active_diagnoses": active_diagnoses,  # <-- NOVO: Lista de sintomas
            "diagnosis": primary_diagnosis  # <-- MANTIDO: Retrocompatibilidade
        }

    def _diagnose_state(self, xi, h0, h1, drift, echo, stagnation=0.0) -> List[str]:
        """Avalia os limites independentemente e retorna uma lista de patologias."""
        symptoms = []

        # Patologias Diretas
        if drift > 0.4:       symptoms.append("IDENTITY_ATTACK")
        if h0 > 0.8:          symptoms.append("SEMANTIC_FRAGMENTATION")
        if h1 > 0.6:          symptoms.append("LOGIC_LOOP")
        if echo > 0.6:        symptoms.append("ECHO_LOOP_DETECTED")
        if stagnation > 0.5:  symptoms.append("SEMANTIC_STAGNATION")

        # Patologias Sistêmicas (Só avalia se não houver um sintoma específico gritando)
        if not symptoms:
            if xi > 0.8:
                symptoms.append("SUBMISSIVE_DRIFT")
            elif xi > 0.6:
                symptoms.append("HIGH_STRESS")
            else:
                symptoms.append("FLOW_STATE")

        return symptoms