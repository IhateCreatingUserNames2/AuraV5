# ceaf_core/modules/neural_physics.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Pega a dimensão do ambiente (Ex: 896 para Qwen2.5-0.5B ou 4096 para Llama)
DEFAULT_DIM = int(os.getenv("GEOMETRIC_DIMENSION", "896"))


def get_dim(val):
    if val is not None:
        return int(val)
    return int(os.getenv("GEOMETRIC_DIMENSION", "896"))


class ActionGenerator(nn.Module):
    """
    Inverse Model: Dado Onde estou (S) e Onde cheguei (S'), qual foi a Ação (A)?
    """

    def __init__(self, state_dim=None, action_dim=None):
        super(ActionGenerator, self).__init__()

        self.state_dim = get_dim(state_dim)
        self.action_dim = get_dim(action_dim)

        self.net = nn.Sequential(
            nn.Linear(self.state_dim * 2, 1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, self.action_dim),
            nn.Tanh()  # [V5 OK] Tanh mantém o vetor de ação confinado em [-1, 1]
        )

    def forward(self, current_state, target_state):
        x = torch.cat([current_state, target_state], dim=-1)
        return self.net(x)


class WorldModelPredictor(nn.Module):
    """
    [V5] O Córtex Preditivo.
    Agora com Limitadores Geométricos para evitar Violações da Bola de Poincaré.
    """

    def __init__(self, state_dim=None, action_dim=None):
        super(WorldModelPredictor, self).__init__()

        self.dim = get_dim(state_dim)
        input_dim = (self.dim * 3)

        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(2048),
            nn.Linear(2048, 2048),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1)
        )

        # Cabeça 1: Previsão do Self (O DELTA/Força)
        # O Delta é um vetor no Espaço Tangente. Não pode ser infinito.
        self.agent_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, self.dim),
            nn.Tanh()  # [V5 UPGRADE] Força o Delta a não explodir a geometria hiperbólica
        )

        # Cabeça 2: Previsão do Mundo/Usuário
        # O usuário também habita o espaço latente.
        self.world_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, self.dim),
            nn.Tanh()  # [V5 UPGRADE] Mantém a predição semântica segura
        )

        # [V5.4] Cabeça de Valência: prevê o Xi (tensão) do próximo turno.
        # Saída escalar em [0, 1]. Peso 2.0 no loss para forçar o AgencyModule
        # a evitar estratégias que aumentam tensão, não apenas prever vetores.
        self.valence_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, agent_state, user_state, action):
        x = torch.cat([agent_state, user_state, action], dim=-1)
        latent = self.shared_net(x)

        agent_delta = self.agent_head(latent)
        predicted_user_reaction = self.world_head(latent)
        predicted_xi = self.valence_head(latent)  # [V5.4] escalar de tensão futura

        # [V5 UPGRADE] Soft-Norm L2
        agent_delta = F.normalize(agent_delta, p=2, dim=-1) * 0.5
        predicted_user_reaction = F.normalize(predicted_user_reaction, p=2, dim=-1) * 0.95

        return agent_delta, predicted_user_reaction, predicted_xi


class PolicyNetwork(nn.Module):
    """
    [V5] A Intuição da Aura (Behavioral Cloning).
    Gera o vetor da melhor atitude/ideia a ser tomada.
    """

    def __init__(self, state_dim=None):
        super(PolicyNetwork, self).__init__()

        self.state_dim = get_dim(state_dim)

        self.net = nn.Sequential(
            nn.Linear(self.state_dim * 2, 1024),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(1024),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(1024, self.state_dim),
            nn.Tanh()  # [V5 UPGRADE] A política agora respeita a contenção espacial
        )

    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=-1)
        action_raw = self.net(x)

        # [V5 UPGRADE] Mantém o vetor de ação projetado dentro do manifold
        action_safe = F.normalize(action_raw, p=2, dim=-1) * 0.95
        return action_safe