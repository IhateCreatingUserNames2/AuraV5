import sqlite3
import json
import torch
import numpy as np

# Caminho do seu banco atual
DB_PATH = "agent_data/78db8340-4960-411f-83ba-e679840c6822/cognitive_turn_history.sqlite"


def extract_vectors():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Pega turnos ordenados por tempo
    cursor.execute("SELECT turn_id, cognitive_state_packet, response_packet FROM turn_history ORDER BY timestamp ASC")
    rows = cursor.fetchall()

    training_data = []  # Lista de (S_t, A_t, S_t+1)

    last_state_vector = None

    for i in range(len(rows) - 1):
        # Turno Atual
        curr_row = rows[i]
        next_row = rows[i + 1]

        try:
            # 1. Estado Atual (S_t)
            curr_cog = json.loads(curr_row[1])
            # No seu código, o vetor de identidade está em: cognitive_state.identity_vector.vector
            s_t = np.array(curr_cog['identity_vector']['vector'])

            # 2. Ação (A_t) - Usamos o vetor da resposta gerada
            # No seu código: response_packet não tem vetor explícito salvo,
            # mas podemos usar o vetor da query do PRÓXIMO turno como proxy
            # ou precisaríamos embedar o texto da resposta agora.
            # *MELHORIA CRÍTICA*: Vamos assumir que você vai embedar o 'content_summary' da resposta
            curr_res = json.loads(curr_row[2])
            response_text = curr_res['content_summary']
            # NOTA: Aqui você precisaria chamar seu embedding_client para transformar texto em vetor
            # Para este script funcionar offline, assumimos que você salvará o vetor da resposta no futuro.

            # 3. Estado Futuro (S_t+1)
            next_cog = json.loads(next_row[1])
            s_next = np.array(next_cog['identity_vector']['vector'])

            # Se conseguimos S_t e S_t+1, temos um par de treino!
            # A_t é a "força" que moveu S_t para S_t+1

            # Por enquanto, sem o vetor da ação salvo, podemos treinar apenas o
            # Autoencoder do Estado ou preparar a infra para salvar o vetor da ação.

            if len(s_t) > 0 and len(s_next) > 0:
                # Salvar tupla
                pass

        except Exception as e:
            print(f"Erro ao processar linha {i}: {e}")
            continue

    print(f"Extraídos {len(training_data)} exemplos de treino.")