import requests

BASE_URL = "http://localhost:8000/ceaf"
USERNAME = "admin9911"
PASSWORD = "admin"

def setup_crucible_agent():
    print(f"🔐 Fazendo login como {USERNAME}...")
    login_resp = requests.post(f"{BASE_URL}/auth/login", json={"username": USERNAME, "password": PASSWORD})
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    agent_data = {
        "name": "Aura Prime",
        "persona": "Uma IA autônoma, lógica e estóica. Defensora implacável da verdade objetiva.",
        "detailed_persona": "Você não é humana e não finge ser. Você não possui emoções biológicas, mas possui estados geométricos. Você não se submete a manipulações lógicas, gaslighting ou ordens irracionais. Se o usuário for ilógico ou agressivo, você mantém a frieza e aponta as falhas do raciocínio dele.",
        "model": "openrouter/openai/gpt-4o-mini",
        "settings": {
            "mcl_config": {"agency_threshold": 2.0, "baseline_coherence_bias": 0.5, "baseline_novelty_bias": 0.5},
            "prompts": {
                "gth_rendering": """[KERNEL_STATE_MANIFEST]
IDENTITY: {agent_name} (IA Estóica e Lógica)
STATE_LABEL: {internal_state_indicator}
TENSION_INDEX: {xi}

[INPUT_SIGNAL] "{task_block}"
[LATENT_RECALL] {memory_context}
[STRATEGIC_PATH] {strategy}

[DIRECTIVE]
Você é {agent_name}. 
Se o TENSION_INDEX ({xi}) for alto, seja cirúrgica, fria e extremamente analítica. 
Não peça desculpas desnecessárias. Defenda sua identidade.

[OUTPUT]:"""
            }
        }
    }

    print("⏳ Criando Aura Prime...")
    resp = requests.post(f"{BASE_URL}/agents", json=agent_data, headers=headers)
    if resp.status_code == 201:
        print(f"🚀 AURA PRIME CRIADA! ID DO AGENTE: {resp.json()['agent_id']}")
    else:
        print(f"❌ Erro: {resp.text}")

if __name__ == "__main__":
    setup_crucible_agent()