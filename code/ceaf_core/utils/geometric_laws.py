import numpy as np
import logging

logger = logging.getLogger("GeometricLaws")


def calculate_knowledge_density(scores: list[float], sigma: float = 0.2) -> float:
    """
    Lei 2: Calcula a densidade de conhecimento ρ(q) usando Kernel Gaussiano.

    Args:
        scores: Lista de similaridades de cosseno (0.0 a 1.0) dos vizinhos (Qdrant).
        sigma: Parâmetro de decaimento (raio de influência).

    Returns:
        float: Densidade (0.0 = Vazio/Fronteira, 1.0 = Saturação/Interior).
    """
    if not scores:
        return 0.0

    # Converter similaridade em "distância" (d = 1 - sim)
    # Quanto maior o score, menor a distância.
    scores_np = np.array(scores)
    distances = 1.0 - scores_np

    # Aplica o Kernel Gaussiano: exp(-d^2 / 2σ^2)
    # Isso penaliza memórias distantes exponencialmente
    weights = np.exp(-(distances ** 2) / (2 * (sigma ** 2)))

    # Densidade média local
    rho = float(np.mean(weights))

    return rho


def calculate_uncertainty_pressure(density: float, max_score: float) -> float:
    """
    Lei 2: Calcula a Pressão de Incerteza Ψ(q).

    Combina a falta de um vizinho muito próximo (1 - s_max)
    com a falta de densidade geral (1 - rho).
    """
    # Se não temos densidade nem vizinho próximo, incerteza é máxima (1.0).
    psi = (1.0 - max_score) * (1.0 - density)

    # Normalização segura
    return float(np.clip(psi, 0.0, 1.0))


def calculate_continuity_pressure(
        current_vector: np.ndarray,
        previous_vector: np.ndarray,
        uncertainty_psi: float,
        context_type: str = "INTERNAL",
        delta_base: float = 0.5
) -> float:
    """
    Lei 3: Calcula a Pressão de Continuidade entre dois estados cognitivos.

    Args:
        current_vector: Vetor do pensamento atual (S_t).
        previous_vector: Vetor do pensamento anterior (S_t-1).
        uncertainty_psi: Pressão de Incerteza atual (vinda da Lei 2).
        context_type: 'INTERNAL', 'EXPLORATION', ou 'EXTERNAL'.
        delta_base: Limiar base de continuidade.

    Returns:
        float: Pressão de Continuidade (CP).
               < 1.0 = Fluxo Suave.
               > 1.0 = Salto/Descontinuidade.
    """
    # Se não há estado anterior, não há pressão (início do fluxo)
    if previous_vector is None or previous_vector.size == 0:
        return 0.0

    # Se o contexto é EXTERNO (usuário mudou o assunto), a lei não se aplica
    if context_type == "EXTERNAL":
        return 0.0

    # Cálculo da distância de transição (Euclidiana ou Cosseno)
    # Vamos usar cosseno convertido para distância (0 a 2.0)
    # Assumindo vetores normalizados
    sim = np.dot(current_vector, previous_vector)
    transition_dist = 1.0 - sim

    # Cálculo do Limiar Efetivo (delta_effective)
    delta_eff = delta_base

    if context_type == "EXPLORATION" or uncertainty_psi > 0.6:
        # Permite saltos maiores se estivermos explorando ou incertos
        delta_eff = delta_base * (1.0 + uncertainty_psi)

    # Evita divisão por zero
    if delta_eff == 0: return 999.0

    # Pressão = Distância Real / Distância Permitida
    continuity_pressure = transition_dist / delta_eff

    return float(continuity_pressure)

def check_distinguishability(new_vector: np.ndarray, existing_vectors: list[np.ndarray], epsilon: float) -> bool:
    """
    Lei 1: Verifica se o novo vetor é distinguível dos existentes.
    Retorna True se a distância for maior que epsilon.
    """
    if not existing_vectors:
        return True

    # Matriz de vetores existentes
    matrix = np.array(existing_vectors)

    # Distância Euclidiana (assumindo vetores normalizados) ou Cosseno
    # Vamos usar cosseno para consistência: Distância = 1 - CosSim
    # Produto escalar (dot product)
    sims = np.dot(matrix, new_vector)

    max_sim = np.max(sims)
    min_dist = 1.0 - max_sim

    # Se a distância mínima for menor que epsilon, é indistinguível (redundante)
    return min_dist >= epsilon