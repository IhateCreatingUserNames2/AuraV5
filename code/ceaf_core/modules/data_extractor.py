# ceaf_core/modules/data_extractor.py
import asyncio
import os
import sqlite3
import json
import numpy as np
import logging
from ceaf_core.utils.embedding_utils import get_embedding_client

logger = logging.getLogger("DataExtractor")
DIM = int(os.getenv("GEOMETRIC_DIMENSION"))


class TrainingDataExtractor:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.embedding_client = get_embedding_client()

    async def extract_vectors(self):
        """
        Extrai (S_t, U_t, A_t) -> (S_t+1, U_t+1)
        """
        logger.info(f"⛏️ Extraindo dados de World Model de {self.db_path}...")

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            # [V5.4] Agora também puxa tension_after como proxy de Xi futuro.
            # tension_after reflete o estado de tensão APÓS a resposta da Aura,
            # que é o Xi que o próximo turno vai "herdar" como estado inicial.
            cursor.execute(
                """SELECT turn_id, cognitive_state_packet, response_packet, intent_text, tension_after
                   FROM turn_history ORDER BY timestamp ASC"""
            )
            rows = cursor.fetchall()
            conn.close()
        except Exception as e:
            logger.error(f"Erro ao ler DB: {e}")
            return []

        training_data = []
        text_cache = {}

        for i in range(len(rows) - 1):
            try:
                curr_row = rows[i]
                next_row = rows[i + 1]

                # --- 1. Extração Segura de S_t ---
                curr_cog = json.loads(curr_row[1])
                s_t_list = curr_cog.get('identity_vector', {}).get('vector')
                if not s_t_list or len(s_t_list) != DIM: continue
                s_t = np.array(s_t_list, dtype=np.float32)

                # --- 2. Extração Segura de U_t ---
                user_text_t = curr_row[3]
                if not user_text_t: continue
                if user_text_t not in text_cache:
                    u_vec = await self.embedding_client.get_embedding(user_text_t)
                    if len(u_vec) == DIM: text_cache[user_text_t] = np.array(u_vec, dtype=np.float32)
                if user_text_t not in text_cache: continue
                u_t = text_cache[user_text_t]

                # --- 3. Extração Segura de A_t ---
                curr_res = json.loads(curr_row[2])
                response_text = curr_res.get('content_summary', '')
                if not response_text: continue
                if response_text not in text_cache:
                    a_vec = await self.embedding_client.get_embedding(response_text)
                    if len(a_vec) == DIM: text_cache[response_text] = np.array(a_vec, dtype=np.float32)
                if response_text not in text_cache: continue
                a_t = text_cache[response_text]

                # --- 4. Extração Segura de S_next ---
                next_cog = json.loads(next_row[1])
                s_next_list = next_cog.get('identity_vector', {}).get('vector')
                if not s_next_list or len(s_next_list) != DIM: continue
                s_next = np.array(s_next_list, dtype=np.float32)

                # --- 5. Extração Segura de U_next ---
                user_text_next = next_row[3]
                if not user_text_next: continue
                if user_text_next not in text_cache:
                    un_vec = await self.embedding_client.get_embedding(user_text_next)
                    if len(un_vec) == DIM: text_cache[user_text_next] = np.array(un_vec, dtype=np.float32)
                if user_text_next not in text_cache: continue
                u_next = text_cache[user_text_next]

                # --- 6. [V5.4] Xi do próximo turno ---
                # Usamos o tension_after do turno ATUAL como o Xi que o próximo turno herdará.
                # Se a coluna não existir (dados legados), assume 0.0 (tensão neutra).
                raw_xi = curr_row[4]
                xi_next = np.float32(raw_xi) if raw_xi is not None else np.float32(0.0)
                # Clamp para garantir que está em [0, 1] para o Sigmoid da valence_head
                xi_next = np.clip(xi_next, 0.0, 1.0)

                training_data.append((s_t, u_t, a_t, s_next, u_next, xi_next))

            except Exception as e:
                logger.debug(f"Pulo na linha {i} por erro de dados: {e}")
                continue

        logger.info(f"📊 Extraídos {len(training_data)} cenários GEOMETRICAMENTE VÁLIDOS.")
        return training_data