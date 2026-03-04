# ceaf_core/riemannian_geometry.py

import numpy as np
import logging

logger = logging.getLogger("RiemannianMath")

class RiemannianGeometry:
    """
    Física do Espaço Mental da Aura X V5 (Bola de Poincaré).
    Implementa a Geometria Hiperbólica para navegação no espaço latente.
    """

    def __init__(self, c=1.0, epsilon=1e-5):
        self.c = c
        self.epsilon = epsilon

    @staticmethod
    def project_to_manifold(x: np.ndarray, max_norm=0.95) -> np.ndarray:
        """
        Safety Clamp CRÍTICO: Embeddings normais têm norma 1.0.
        Na geometria hiperbólica, 1.0 é infinito.
        Esta função "encolhe" o vetor para dentro da bola (ex: 0.95).
        """
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        # Se a norma for maior que o limite de segurança, reescala
        cond = norm > max_norm
        # Evita divisão por zero adicionando 1e-15
        return np.where(cond, x * (max_norm / (norm + 1e-15)), x)

    @staticmethod
    def cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
        """
        [V5.2] A MEDIDA DE ALIENIDADE SEMÂNTICA (Equilibrada).
        """
        u_flat, v_flat = u.flatten(), v.flatten()

        norm_u = np.linalg.norm(u_flat)
        norm_v = np.linalg.norm(v_flat)

        # Evita divisão por zero
        if norm_u == 0 or norm_v == 0:
            return 1.0  # Neutro

        sim = np.dot(u_flat, v_flat) / (norm_u * norm_v)

        # O clip garante que erros de float32 não estourem os limites
        sim_clipped = np.clip(sim, -1.0, 1.0)

        dist = 1.0 - sim_clipped
        return float(dist)

    @staticmethod
    def mobius_add(u: np.ndarray, v: np.ndarray, c=1.0) -> np.ndarray:
        # Garante que inputs estão seguros
        u = RiemannianGeometry.project_to_manifold(u)
        v = RiemannianGeometry.project_to_manifold(v)

        u2 = np.sum(np.square(u), axis=-1, keepdims=True)
        v2 = np.sum(np.square(v), axis=-1, keepdims=True)
        uv = np.sum(u * v, axis=-1, keepdims=True)

        num = (1 + 2 * c * uv + c * v2) * u + (1 - c * u2) * v
        den = 1 + 2 * c * uv + c ** 2 * u2 * v2

        return num / (den + 1e-15)

    @staticmethod
    def exp_map(x: np.ndarray, v: np.ndarray, c=1.0) -> np.ndarray:
        # Garante que o ponto base x está seguro dentro da bola
        x = RiemannianGeometry.project_to_manifold(x)

        x_norm_sq = np.sum(np.square(x), axis=-1, keepdims=True)
        lambda_x = 2 / (1 - c * x_norm_sq + 1e-15)

        v_norm = np.linalg.norm(v, axis=-1, keepdims=True)

        if np.all(v_norm < 1e-7):
            return x

        direction = v / (v_norm + 1e-15)
        factor = np.tanh(np.sqrt(c) * lambda_x * v_norm / 2) / np.sqrt(c)

        scaled_v = factor * direction

        return RiemannianGeometry.mobius_add(x, scaled_v, c)

    @staticmethod
    def log_map(x: np.ndarray, y: np.ndarray, c=1.0) -> np.ndarray:
        """
        Logarithmic Map (Log_x(y)).
        [V5.1 FIX] Estabilidade numérica aprimorada para vetores escalados.
        """
        diff = RiemannianGeometry.mobius_add(-x, y, c)
        diff_norm = np.linalg.norm(diff, axis=-1, keepdims=True) + 1e-15  # Evita div por zero

        # Se muito perto, retorna zero
        if np.all(diff_norm < 1e-7):
            return np.zeros_like(x)

        x_norm_sq = np.sum(np.square(x), axis=-1, keepdims=True)
        # Clamp para garantir que lambda não exploda perto da borda
        lambda_x = 2 / (1 - c * np.clip(x_norm_sq, 0, 0.99) + 1e-15)

        # Arcotangente hiperbólica precisa de input < 1.0
        # O diff_norm pode flutuar levemente acima de 1 por erro numérico, clipamos.
        safe_diff_norm = np.clip(diff_norm, 0, 0.999)

        scale = (2 / (np.sqrt(c) * lambda_x)) * np.arctanh(np.sqrt(c) * safe_diff_norm)
        return scale * (diff / diff_norm)

    @staticmethod
    def poincare_distance(u: np.ndarray, v: np.ndarray, c=1.0) -> float:
        """
        Calcula a distância. Aplica projeção de segurança ANTES do cálculo.
        Usado para o Veto Físico no AgencyModule.
        """
        u = RiemannianGeometry.project_to_manifold(u, max_norm=0.95)
        v = RiemannianGeometry.project_to_manifold(v, max_norm=0.95)

        sq_dist = np.sum(np.square(u - v), axis=-1)
        sq_u = np.sum(np.square(u), axis=-1)
        sq_v = np.sum(np.square(v), axis=-1)

        # O denominador nunca mais será zero
        denom = (1 - c * sq_u) * (1 - c * sq_v)
        arg = 1 + 2 * sq_dist / (denom + 1e-15)

        arg = np.maximum(arg, 1.0 + 1e-7)
        dist = np.arccosh(arg) / np.sqrt(c)

        if isinstance(dist, np.ndarray):
            return float(dist.item())
        return float(dist)