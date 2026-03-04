# ceaf_core/modules/world_model_evaluator.py
import torch
import numpy as np
import logging
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("WMEvaluator")


class WorldModelEvaluator:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def evaluate_offline(self, val_loader: DataLoader):
        """
        Roda o benchmark offline inspirado no protocolo SWM.
        [V5.4 FIX] Atualizado para aceitar tuplas de 6 elementos (com valência).
        """
        logger.info("🧪 Iniciando Benchmark Offline do World Model (SWM Protocol)...")

        total_cosine_sim_agent = 0.0
        total_cosine_sim_user = 0.0
        total_mse_valence = 0.0  # Novo tracking
        total_steps = 0

        with torch.no_grad():
            # [FIX] Agora desempacota 6 valores. O *args pega qualquer extra se houver.
            for batch in val_loader:
                if len(batch) == 6:
                    s, u, a, s_next_real, u_next_real, xi_next_real = batch
                else:
                    # Fallback para compatibilidade com dados antigos (5 elementos)
                    s, u, a, s_next_real, u_next_real = batch[:5]
                    xi_next_real = None

                s = s.to(self.device)
                u = u.to(self.device)
                a = a.to(self.device)
                s_next_real = s_next_real.to(self.device)
                u_next_real = u_next_real.to(self.device)
                if xi_next_real is not None:
                    xi_next_real = xi_next_real.to(self.device)

                # 1. Previsão do Modelo
                # O modelo V5.4 retorna 3 valores: delta, reação, valência
                outputs = self.model(s, u, a)

                # Desempacota com segurança (caso o modelo seja antigo e retorne só 2)
                if len(outputs) == 3:
                    pred_agent_delta, pred_user_reaction, pred_xi = outputs
                else:
                    pred_agent_delta, pred_user_reaction = outputs
                    pred_xi = None

                # Reconstrói o estado previsto do agente (S_t + delta)
                s_next_pred = s + pred_agent_delta

                # 2. Métricas
                sim_agent = torch.nn.functional.cosine_similarity(s_next_pred, s_next_real)
                sim_user = torch.nn.functional.cosine_similarity(pred_user_reaction, u_next_real)

                total_cosine_sim_agent += sim_agent.mean().item()
                total_cosine_sim_user += sim_user.mean().item()

                # Validação de Valência
                if pred_xi is not None and xi_next_real is not None:
                    mse_xi = torch.nn.functional.mse_loss(pred_xi, xi_next_real.unsqueeze(1))
                    total_mse_valence += mse_xi.item()

                total_steps += 1

        avg_agent_acc = total_cosine_sim_agent / total_steps
        avg_user_acc = total_cosine_sim_user / total_steps
        avg_valence_loss = total_mse_valence / total_steps if total_mse_valence > 0 else 0.0

        # 3. Relatório SWM
        report = {
            "agent_state_prediction_accuracy": avg_agent_acc,
            "theory_of_mind_accuracy": avg_user_acc,
            "valence_prediction_loss": avg_valence_loss,
            "status": "HEALTHY" if avg_agent_acc > 0.85 else "NEEDS_TRAINING"
        }

        logger.info(f"📊 Benchmark Result: {report}")
        return report