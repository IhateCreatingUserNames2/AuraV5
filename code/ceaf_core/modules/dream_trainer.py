# ceaf_core/modules/dream_trainer.py
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset
import os
import logging
import numpy as np

from neural_physics import ActionGenerator, PolicyNetwork, WorldModelPredictor
from ceaf_core.modules.dream_augmentor import SemanticFoVInjector
from ceaf_core.modules.world_model_evaluator import WorldModelEvaluator


logger = logging.getLogger("DreamTrainer")
DIM = int(os.getenv("GEOMETRIC_DIMENSION"))

MODEL_PATH = "./aura_brain/"
os.makedirs(MODEL_PATH, exist_ok=True)


class DreamMachine:
    def __init__(self, state_dim=None, action_dim=None):
        self.state_dim = state_dim if state_dim else DIM
        self.action_dim = action_dim if action_dim else DIM

        logger.info(f"🧠 Inicializando DreamMachine com dimensão: {self.state_dim}")

        # 1. World Model (Substitui o antigo Forward Model simples)
        # Ele agora prevê S_next e U_next (reação do usuário)
        self.world_model = WorldModelPredictor(state_dim=self.state_dim, action_dim=self.action_dim)

        # 2. Inverse Model (Mantido para análise causal)
        # Prevê Ação dado S_current e S_next
        self.inverse_model = ActionGenerator(state_dim=self.state_dim, action_dim=self.action_dim)

        # 3. Policy Network (Mantido para intuição rápida)
        # Prevê Ação dado S_current e S_goal
        self.policy_model = PolicyNetwork(state_dim=self.state_dim)

        self.criterion = nn.MSELoss()

        self.augmentor = SemanticFoVInjector()  # Inicia o augmentor do SWM
        self.evaluator = WorldModelEvaluator(self.world_model)
    def load_brains(self):
        """Carrega pesos existentes se houver"""
        try:
            # Tenta carregar o novo World Model
            # Se não existir, tenta carregar o antigo Forward Model (compatibilidade parcial)
            if os.path.exists(f"{MODEL_PATH}world_model.pth"):
                self.world_model.load_state_dict(torch.load(f"{MODEL_PATH}world_model.pth"))
                logger.info("🧠 World Model (JEPA) carregado.")
            elif os.path.exists(f"{MODEL_PATH}forward.pth"):
                # Aviso: Pesos antigos podem não ser compatíveis se a arquitetura mudou muito
                logger.warning("⚠️ Forward Model antigo encontrado. Recomendado treinar World Model do zero.")
            else:
                logger.info("🌱 Novo World Model iniciado.")

            # Carrega Inverse Model
            if os.path.exists(f"{MODEL_PATH}inverse.pth"):
                self.inverse_model.load_state_dict(torch.load(f"{MODEL_PATH}inverse.pth"))
                logger.info("🧠 Inverse Model carregado.")

            # Carrega Policy Network
            if os.path.exists(f"{MODEL_PATH}policy.pth"):
                self.policy_model.load_state_dict(torch.load(f"{MODEL_PATH}policy.pth"))
                logger.info("🧠 Policy Network carregada.")

        except Exception as e:
            logger.warning(f"⚠️ Erro ao carregar cérebros: {e}. Iniciando limpo.")

    def train_cycle(self, training_data, epochs=3):
        """
        O Processo de 'Sonhar' (REM Cycle) otimizado pelo protocolo SWM.
        Inclui Augmentation (Semantic FoV) e Validação Offline com filtro de integridade.
        """
        if not training_data or len(training_data) < 3:
            logger.info("💤 Poucos dados para um ciclo estável (min 3).")
            return "Skipped (Insufficient Data)"

        # --- FASE 1: FILTRO DE INTEGRIDADE GEOMÉTRICA (CORREÇÃO DO ERRO) ---
        # Garante que todos os vetores no lote tenham exatamente o mesmo tamanho (DIM)
        valid_samples = []
        for i, sample in enumerate(training_data):
            try:
                # Suporta tuplas com 5 elementos (legado) ou 6 (V5.4 com xi_next)
                if len(sample) == 6:
                    s, u, a, s_next, u_next, xi_next = sample
                elif len(sample) == 5:
                    s, u, a, s_next, u_next = sample
                    xi_next = np.float32(0.0)  # fallback: tensão neutra para dados antigos
                else:
                    logger.warning(f"⚠️ Amostra #{i} descartada: tupla com {len(sample)} elementos.")
                    continue

                checks = [
                    isinstance(s, np.ndarray) and s.shape == (self.state_dim,),
                    isinstance(u, np.ndarray) and u.shape == (self.state_dim,),
                    isinstance(a, np.ndarray) and a.shape == (self.state_dim,),
                    isinstance(s_next, np.ndarray) and s_next.shape == (self.state_dim,),
                    isinstance(u_next, np.ndarray) and u_next.shape == (self.state_dim,),
                    isinstance(xi_next, (float, np.floating)) and 0.0 <= float(xi_next) <= 1.0,
                ]

                if all(checks):
                    valid_samples.append((s, u, a, s_next, u_next, np.float32(xi_next)))
                else:
                    logger.warning(f"⚠️ Amostra #{i} descartada: Geometria inconsistente.")
            except Exception as e:
                logger.error(f"❌ Erro ao validar amostra #{i}: {e}")
                continue

        if len(valid_samples) < 10:
            logger.error(f"❌ Dados insuficientes após filtragem: {len(valid_samples)} válidos.")
            return f"Failed: Inhomogeneous data (Only {len(valid_samples)} valid)"

        # --- FASE 2: PREPARAÇÃO DE TENSORES ---
        try:
            batch_s, batch_u, batch_a, batch_s_next, batch_u_next, batch_xi_next = zip(*valid_samples)

            t_s = torch.tensor(np.stack(batch_s), dtype=torch.float32)
            t_u = torch.tensor(np.stack(batch_u), dtype=torch.float32)
            t_a = torch.tensor(np.stack(batch_a), dtype=torch.float32)
            t_s_next = torch.tensor(np.stack(batch_s_next), dtype=torch.float32)
            t_u_next = torch.tensor(np.stack(batch_u_next), dtype=torch.float32)
            t_xi_next = torch.tensor(np.array(batch_xi_next, dtype=np.float32))  # shape: (N,)

            full_dataset = TensorDataset(t_s, t_u, t_a, t_s_next, t_u_next, t_xi_next)
        except Exception as e:
            logger.error(f"🔥 Erro crítico na conversão de tensores: {e}", exc_info=True)
            return f"Failed: {str(e)}"

        # 3. Divisão de Treino/Validação (80/20) - Protocolo SWM
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=min(32, train_size), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=min(32, val_size))

        # 4. Configuração de Otimizadores
        opt_world = optim.AdamW(self.world_model.parameters(), lr=1e-4)
        opt_inv = optim.AdamW(self.inverse_model.parameters(), lr=1e-4)
        opt_policy = optim.AdamW(self.policy_model.parameters(), lr=1e-4)

        logger.info(f"🌙 Ciclo REM: {train_size} treinos | {val_size} testes | {epochs} épocas")

        self.world_model.train()
        self.inverse_model.train()
        self.policy_model.train()

        # Loop de Treinamento
        for epoch in range(epochs):
            total_w_loss = 0
            for s, u, a, s_next, u_next, xi_next in train_loader:
                aug_s, aug_u, aug_a = self.augmentor.augment_batch(s, u, a)

                # --- 1. Treino do World Model ---
                opt_world.zero_grad()
                pred_agent_delta, pred_user_reaction, pred_xi = self.world_model(aug_s, aug_u, aug_a)

                target_delta = s_next - s
                loss_self = self.criterion(pred_agent_delta, target_delta)
                loss_user = self.criterion(pred_user_reaction, u_next)

                # [V5.4] Perda de Valência: penaliza erros de predição de tensão futura.
                # Peso 2.0 > peso do usuário (1.5) para priorizar evitar "dor" sobre
                # acertar o vetor semântico exato da reação.
                loss_xi = self.criterion(pred_xi, xi_next.unsqueeze(1))
                loss_world = loss_self + (1.5 * loss_user) + (2.0 * loss_xi)

                loss_world.backward()
                opt_world.step()
                total_w_loss += loss_world.item()

                # --- 2. Treino do Inverse Model ---
                opt_inv.zero_grad()
                pred_action_inv = self.inverse_model(s, s_next)
                loss_inv = self.criterion(pred_action_inv, a)
                loss_inv.backward()
                opt_inv.step()

                # --- 3. Treino da Policy Network (Behavioral Cloning) ---
                opt_policy.zero_grad()
                pred_action_pol = self.policy_model(s, s_next)
                loss_policy = self.criterion(pred_action_pol, a)
                loss_policy.backward()
                opt_policy.step()

        # 5. AVALIAÇÃO OFFLINE (Protocolo SWM de Robustez)
        report = self.evaluator.evaluate_offline(val_loader)

        # 6. Persistência
        torch.save(self.world_model.state_dict(), f"{MODEL_PATH}world_model.pth")
        torch.save(self.inverse_model.state_dict(), f"{MODEL_PATH}inverse.pth")
        torch.save(self.policy_model.state_dict(), f"{MODEL_PATH}policy.pth")

        summary = (
            f"☀️ Acordando... Acc Agente: {report['agent_state_prediction_accuracy']:.2%} | "
            f"Acc Teoria da Mente: {report['theory_of_mind_accuracy']:.2%} | "
            f"Status: {report['status']}"
        )

        logger.info(summary)
        return summary