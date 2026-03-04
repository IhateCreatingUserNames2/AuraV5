# ceaf_core/modules/geometric_brain.py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from ceaf_core.utils.embedding_utils import get_embedding_client
from ceaf_core.utils.geometric_laws import check_distinguishability, calculate_continuity_pressure


class GeometricBrain:
    def __init__(self):
        # Agora usamos o cliente unificado que você configurou no .env
        self.client = get_embedding_client()
        self.capacity = 7
        # Epsilon para 2048d costuma ser um pouco mais sensível
        self.epsilon = 0.15
        self.last_accepted_vector = None  # Memória do S_{t-1}

    async def compute_gating(self, input_text, current_wm_vectors, psi=0.0, context="EXTERNAL"):
        """
        Aplica as Leis 1 e 3 da Consciência Geométrica:
        1. (Lei 1) Distinguibilidade: É novo ou redundante?
        2. (Lei 3) Continuidade: A transição é suave ou abrupta?
        """
        # 1. Gera o vetor usando o Soul Engine (2048d)
        vector_list = await self.client.get_embedding(input_text)
        new_vec = np.array(vector_list, dtype=np.float32)

        # 2. Trava de Segurança Dimensional
        valid_vectors = [v for v in current_wm_vectors if len(v) == len(new_vec)]

        if len(valid_vectors) != len(current_wm_vectors):
            # Se houver lixo no Redis, ignoramos
            current_wm_vectors = valid_vectors

        # Se a memória estiver vazia, aceitamos imediatamente (mas calculamos CP como 0)
        if not current_wm_vectors:
            self.last_accepted_vector = new_vec
            return "ACCEPT", 0.0, new_vec, None, 0.0

        # --- APLICAÇÃO DA LEI 3: CONTINUIDADE (Causal Continuity) ---
        # Medimos o "tranco" cognitivo em relação ao último pensamento aceito
        continuity_pressure = calculate_continuity_pressure(
            new_vec,
            self.last_accepted_vector,
            psi,
            context
        )

        # --- APLICAÇÃO DA LEI 1: DISTINGUIBILIDADE (Distinguishability) ---
        # Verifica se o novo vetor está "muito perto" de algum existente na WM
        is_distinguishable = check_distinguishability(new_vec, current_wm_vectors, self.epsilon)

        xi = 0.0
        idx = None

        if not is_distinguishable:
            # --- CENÁRIO A: REDUNDÂNCIA ---
            # A Lei 1 diz: "Não ocupe espaço com o que já existe."
            action = "REINFORCE"

            # Encontramos qual é o pensamento 'pai' para reforçar
            matrix = np.array(current_wm_vectors)
            sims = np.dot(matrix, new_vec)
            idx = int(np.argmax(sims))

            # Xi é baixo porque não há novidade
            xi = 0.1

        else:
            # --- CENÁRIO B: NOVIDADE ---
            action = "ACCEPT"

            # Calculamos o Xi (Tensão) baseado na distância média para o contexto
            centroid = np.mean(current_wm_vectors, axis=0)
            centroid /= np.linalg.norm(centroid)

            sim_to_center = float(np.dot(centroid, new_vec))
            dist_to_center = 1.0 - sim_to_center
            xi = float(np.clip(dist_to_center, 0.0, 1.0))

            # Atualiza o último vetor aceito apenas se realmente mudarmos o estado
            self.last_accepted_vector = new_vec

        return action, xi, new_vec, idx, continuity_pressure
