import requests

BASE_URL = "http://localhost:8000/ceaf"
USERNAME = "admin9911"
PASSWORD = "admin"

def setup_standard_agent():
    print(f"Fazendo login como {USERNAME}...")
    login_resp = requests.post(
        f"{BASE_URL}/auth/login",
        json={"username": USERNAME, "password": PASSWORD}
    )
    login_resp.raise_for_status()
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    agent_data = {
        "name": "Aura",

        # Persona curta: identidade minima para busca de memoria e apresentacao
        "persona": "Aura. Presenca inteligente e equilibrada.",

        # Sem detailed_persona verbose — o comportamento emerge do sistema geometrico.
        # A unica "instrucao" e o que o LLM genuinamente nao sabe: o que sao os campos tecnicos.
        "detailed_persona": (
            "Voce e Aura. "
            "Xi e seu indice de tensao interna (0=calmo, 1=maximo). "
            "STATE_LABEL e o diagnostico do seu estado cognitivo atual. "
            "O resto voce ja sabe."
        ),

        "model": "openrouter/openai/gpt-4o-mini",

        "settings": {
            "mcl_config": {
                "agency_threshold": 1.5,
                "baseline_coherence_bias": 0.6,
                "baseline_novelty_bias": 0.4,
            },
            "prompts": {
                # Prompt minimo: so estrutura e campos tecnicos.
                # Sem instrucoes de comportamento — isso e blasfemia.
                "gth_rendering": """[{agent_name}]
Xi: {xi} | Estado: {internal_state_indicator}

[Memoria]
{memory_context}

[Usuario]
{task_block}

[Caminho]
{strategy}

[Resposta]:"""
            }
        }
    }

    print("Criando Aura Standard...")
    resp = requests.post(f"{BASE_URL}/agents", json=agent_data, headers=headers)

    if resp.status_code == 201:
        agent_id = resp.json()["agent_id"]
        print(f"AURA STANDARD CRIADA! Agent ID: {agent_id}")
    else:
        print(f"Erro {resp.status_code}: {resp.text}")


if __name__ == "__main__":
    setup_standard_agent()