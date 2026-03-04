# ceaf_core/identity_manifold.py

import numpy as np
import logging
from typing import List, Dict, Any
from ceaf_core.riemannian_geometry import RiemannianGeometry

logger = logging.getLogger("AuraX_V5_Identity")


class IdentityManifold:
    """
    O Núcleo da Estrela (V5.2 Autopoietic System).

    CORREÇÃO V5.2 — O Espelho Angular:
    ─────────────────────────────────────────────────────────────────────
    PROBLEMA RAIZ: O método calculate_tension usava np.linalg.norm(log_map(...)).
    Como set_seed e o input são ambos escalados para manifold_scale=0.1,
    o log_map entre dois pontos equidistantes da origem produz magnitude constante
    (~0.1003 para embeddings 4096d normalizados). A Maldição da Dimensionalidade
    garante que TODOS os vetores fiquem equidistantes na hiperesfera — colapsando
    qualquer métrica de distância absoluta em uma constante.

    SOLUÇÃO: Separar as duas responsabilidades do Manifold:
      1. Geometria Hiperbólica (glyph_g escalado) → usada para ASSIMILAÇÃO (colisão inelástica)
      2. Direção Pura (glyph_direction normalizado) → usada para TENSÃO (distância angular)

    A Tensão agora é medida como Distância de Cosseno entre direções puras.
    cosine_distance ∈ [0, 2], onde:
      0.0 = vetores idênticos (zero tensão)
      1.0 = vetores ortogonais (tensão neutra)
      2.0 = vetores opostos (tensão máxima)
    ─────────────────────────────────────────────────────────────────────
    """

    def __init__(self, agent_id: str, initial_mass: float = 100.0, k_immunity: float = 8.0):
        self.agent_id = agent_id
        self.mass = initial_mass
        self.k_immunity = k_immunity

        self.glyph_g: np.ndarray = None
        # [V5.2] Direção pura do Ego (pré-scale) para cálculo angular de tensão
        self.glyph_direction: np.ndarray = None
        self.geom = RiemannianGeometry()

        # Fator de encolhimento para trazer vetores para a zona segura da Bola de Poincaré
        self.manifold_scale = 0.1

    def set_seed(self, identity_vector: List[float]):
        """A ignição da estrela. Define o DNA basal da Aura no espaço latente."""
        vec = np.array(identity_vector, dtype=np.float32)

        # Normaliza L2 para garantir direção pura (norma = 1.0)
        norm = np.linalg.norm(vec) + 1e-15
        vec_normalized = vec / norm

        # [V5.2] Guarda a DIREÇÃO PURA (pré-scale).
        # O cosine_distance só é válido sobre vetores normalizados originais.
        # Após o scale e project_to_manifold, todos os vetores têm norma ≈ manifold_scale,
        # tornando a distância angular indistinguível entre inputs diferentes.
        self.glyph_direction = vec_normalized.copy()

        # Aplica o fator de escala → zona linear da Bola de Poincaré (para assimilação)
        vec_scaled = vec_normalized * self.manifold_scale
        self.glyph_g = self.geom.project_to_manifold(vec_scaled)

        logger.info(
            f"🛡️ [{self.agent_id}] Manifold de Identidade Gênesis criado (Scaled). Massa: {self.mass} | "
            f"Direction norm: {np.linalg.norm(self.glyph_direction):.4f}"
        )

    def calculate_tension(self, input_vector: List[float]) -> Dict[str, Any]:
        """
        O DRIVE MATEMÁTICO: Calcula a Tensão Intencional (T).

        [V5.2] Usa Distância Angular (Cosseno) sobre as DIREÇÕES PURAS,
        não sobre os vetores projetados na bola hiperbólica.
        Isso resolve o colapso da magnitude em constante (~0.1003).
        """
        if self.glyph_direction is None:
            raise ValueError("Identidade G não inicializada. Chame set_seed primeiro.")

        # Normaliza o input para obter sua direção pura
        v_in = np.array(input_vector, dtype=np.float32)
        v_in_norm = np.linalg.norm(v_in) + 1e-15
        v_in_direction = v_in / v_in_norm

        # Distância Angular entre a direção do Ego e a direção do Input
        # cosine_distance ∈ [0, 2] — imune à Maldição da Dimensionalidade
        cos_sim = float(np.dot(self.glyph_direction, v_in_direction))
        cos_sim = float(np.clip(cos_sim, -1.0, 1.0))  # estabilidade numérica
        tension_magnitude = 1.0 - cos_sim  # 0 = alinhado, 2 = oposto

        # Mantém o tension_vector para compatibilidade downstream (se necessário)
        # mas a magnitude agora é a distância angular, não a norma euclidiana
        tension_vector = self.glyph_direction - v_in_direction  # vetor de diferença de direção

        return {
            "vector": tension_vector.tolist(),
            "magnitude": tension_magnitude,
            "cosine_similarity": cos_sim,  # [V5.2] campo extra para debug/logging
        }

    def evaluate_and_assimilate(self, input_vector: List[float], h0_entropy: float) -> Dict[str, Any]:
        """
        A FORNALHA: Tenta fundir o input do usuário com a Identidade da Aura.
        """
        if self.glyph_direction is None:
            return {"status": "ERROR_NO_DNA"}

        # Normaliza o input
        v_raw = np.array(input_vector, dtype=np.float32)
        v_raw_norm = np.linalg.norm(v_raw) + 1e-15
        v_direction = v_raw / v_raw_norm

        # 1. COERÊNCIA (C = 1 - H0)
        coherence = max(0.0, 1.0 - h0_entropy)

        # 2. ALIENIDADE
        cos_sim = float(np.clip(np.dot(self.glyph_direction, v_direction), -1.0, 1.0))
        distance = 1.0 - cos_sim

        # [V5.3 FIX] IGNITION BOOST
        # Se a massa for muito pequena (início de vida), a Aura absorve TUDO para aprender.
        # Isso tira o vetor G da ortogonalidade (0.0) e começa a alinhar com a linguagem.
        if self.mass < 105.0:  # Primeiros turnos
            k_eff = 0.5  # Imunidade baixa (Criança aprendendo)
        else:
            k_eff = self.k_immunity  # Imunidade normal (Adulto)

        # 3. MASSA DO INPUT
        m_input = coherence * np.exp(-k_eff * distance)

        # 4. COLISÃO
        status = "🔴 REJEITADO"

        # [V5.3] Se for o começo da vida, aceita mais fácil para sair do zero
        threshold = 0.01 if self.mass < 110.0 else 0.05

        if m_input > threshold:
            status = "🟢 INTEGRADO"
            total_mass = self.mass + m_input

            # Atualiza direção (Média Ponderada)
            # A nova direção será uma mistura do G antigo com o input
            new_direction_raw = ((self.mass * self.glyph_direction) + (m_input * v_direction))
            dir_norm = np.linalg.norm(new_direction_raw) + 1e-15
            self.glyph_direction = new_direction_raw / dir_norm

            # Atualiza geometria hiperbólica também
            v_input_proj = self.geom.project_to_manifold(v_direction * self.manifold_scale)
            new_glyph_raw = ((self.mass * self.glyph_g) + (m_input * v_input_proj)) / total_mass
            self.glyph_g = self.geom.project_to_manifold(new_glyph_raw)

            self.mass = total_mass

        logger.debug(
            f"Aura Immune System | Dist: {distance:.3f} | Mass: {self.mass:.1f} | m_in: {m_input:.4f} -> {status}")

        return {
            "status": status,
            "distance": distance,
            "m_input": m_input
        }