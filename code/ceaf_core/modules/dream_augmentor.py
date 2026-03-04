# ceaf_core/modules/dream_augmentor.py
import torch
import numpy as np
import logging

logger = logging.getLogger("DreamAugmentor")


class SemanticFoVInjector:
    """
    Implementa Fatores de Variação (FoV) Semânticos inspirados no SWM.
    Injeta perturbações controladas nos vetores de estado para garantir
    que o World Model aprenda dinâmicas robustas e não apenas memorize trajetórias.
    """

    def __init__(self, noise_level=0.02, dropout_prob=0.1):
        self.noise_level = noise_level
        self.dropout_prob = dropout_prob

    def apply_fov(self, state_tensor: torch.Tensor, mode='training') -> torch.Tensor:
        """
        Aplica variações ao vetor de estado (S_t).

        Args:
            state_tensor: Tensor [Batch, Dim] representando o estado cognitivo.
            mode: 'training' aplica ruído, 'eval' retorna limpo.
        """
        if mode != 'training':
            return state_tensor

        # 1. Gaussian Jitter (Simula pequenas variações de humor/contexto)
        # SWM: Equivalente a mudar levemente a iluminação ou atrito.
        noise = torch.randn_like(state_tensor) * self.noise_level
        augmented_state = state_tensor + noise

        # 2. Semantic Dropout (Simula esquecimento ou falha de atenção)
        # SWM: Equivalente a oclusão parcial de objetos.
        mask = torch.bernoulli(torch.full_like(state_tensor, 1 - self.dropout_prob))
        augmented_state = augmented_state * mask

        # Renormaliza para manter a geometria na hiperesfera (se necessário)
        # augmented_state = torch.nn.functional.normalize(augmented_state, p=2, dim=1)

        return augmented_state

    def augment_batch(self, s_t, u_t, a_t):
        """Aumenta um lote de treinamento aplicando FoV apenas no Estado e Usuário, não na Ação."""
        aug_s = self.apply_fov(s_t)
        aug_u = self.apply_fov(u_t)  # O input do usuário também pode ser ambíguo
        return aug_s, aug_u, a_t