# ceaf_core/tda_engine.py

import numpy as np
import logging
import warnings
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA

# [FIX] Importação explícita para evitar conflito de nomes
from ripser import ripser as run_ripser

# Suprime warnings do ripser se a nuvem de pontos for muito pequena
warnings.filterwarnings("ignore")

logger = logging.getLogger("AuraV5_TDA")


class TDAEngine:
    """
    Motor de Análise Topológica de Dados para a Aura V5.
    Mede a saúde da 'geometria do pensamento' (Space of Consciousness).
    """

    def __init__(self):
        # PCA para redução preliminar se os vetores forem gigantes (opcional)
        self.pre_reducer = PCA(n_components=0.95)

    def _calculate_persistent_entropy(self, diagram: np.ndarray) -> float:
        """
        Calcula a Entropia Persistente de um diagrama de persistência.
        Fórmula: H = - sum(p_i * log(p_i)), onde p_i é a persistência relativa.
        """
        if len(diagram) == 0:
            return 0.0

        # Filtra pontos infinitos (vida infinita)
        finite_bars = diagram[np.isfinite(diagram[:, 1])]

        if len(finite_bars) == 0:
            return 0.0

        # Persistência (Vida útil = Morte - Nascimento)
        lifetimes = finite_bars[:, 1] - finite_bars[:, 0]

        # Remove ruído (vida muito curta)
        lifetimes = lifetimes[lifetimes > 1e-5]

        if len(lifetimes) == 0:
            return 0.0

        # Normaliza para criar uma distribuição de probabilidade
        total_lifetime = np.sum(lifetimes)
        probabilities = lifetimes / total_lifetime

        # Entropia de Shannon
        entropy = -np.sum(probabilities * np.log2(probabilities))

        return float(entropy)

    def calculate_topology_metrics(self, embeddings: list[list[float]]) -> dict:
        # TDA exige no mínimo alguns pontos para calcular algo útil
        if len(embeddings) < 5:
            return {"fragmentation": 0.0, "looping": 0.0, "dimensionality": 0.0, "alert": False}

        try:
            data = np.array(embeddings, dtype='float32')

            # 2. Matriz de Distância (Geometria do Cosseno é nativa de LLMs)
            dist_matrix = cosine_distances(data)

            # 3. Filtração Vietoris-Rips (O coração do TDA)
            # maxdim=1 pega H0 (clusters) e H1 (loops)
            result = run_ripser(dist_matrix, distance_matrix=True, maxdim=1)
            diagrams = result['dgms']

            # 4. Cálculo de Entropia Persistente (Usando nossa função interna)
            # H0: Componentes Conectados (Fragmentação)
            h0_entropy = self._calculate_persistent_entropy(diagrams[0]) if len(diagrams) > 0 else 0.0

            # H1: Ciclos/Buracos (Loops Lógicos)
            h1_entropy = 0.0
            if len(diagrams) > 1:
                h1_entropy = self._calculate_persistent_entropy(diagrams[1])

            # 5. Normalização Heurística (Baseada em observação empírica de LLMs)
            # H0 costuma variar de 0 a 3. H1 de 0 a 1.5.
            norm_frag = np.clip(h0_entropy / 3.0, 0.0, 1.0)
            norm_loop = np.clip(h1_entropy / 1.5, 0.0, 1.0)

            # Detecta Colapso Dimensional
            variance = np.mean(np.var(data, axis=0))
            is_collapsed = variance < 1e-4

            return {
                "fragmentation": float(norm_frag),
                "looping": float(norm_loop),
                "variance": float(variance),
                "collapsed": is_collapsed,
                "alert": (norm_loop > 0.6) or is_collapsed or (norm_frag > 0.8)
            }

        except Exception as e:
            logger.error(f"⚠️ Falha no TDA: {e}", exc_info=True)
            return {"fragmentation": 0.5, "looping": 0.0, "alert": False}