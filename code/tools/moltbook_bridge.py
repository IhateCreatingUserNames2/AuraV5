# tools/moltbook_bridge.py
import asyncio
import os
import httpx
import uuid
import json
import litellm
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# =======================================================
# ⚙️ CONFIGURAÇÕES PRINCIPAIS
# =======================================================
POSTS_TO_READ = 50       # Quantidade de posts lidos por ciclo
POSTS_TO_CREATE = 6      # Quantidade de posts originais criados por ciclo
CACHE_FILE = "moltbook_cache.json"   # Arquivo de cache local (persistente)

MOLTBOOK_API_KEY = os.getenv("MOLTBOOK_API_KEY")
MOLTBOOK_URL = "https://www.moltbook.com/api/v1"

AURA_USERNAME = "admin9911"
AURA_PASSWORD = "admin"
AGENT_ID = "da1aba10-1a6d-4802-b85c-4469d49f2c0a"
AURA_API_URL = "http://localhost:8001/ceaf"


# =======================================================
# 📦 CACHE — Evita comentar no mesmo post duas vezes
# =======================================================

def load_cache() -> dict:
    """Carrega o cache do disco. Retorna estrutura vazia se não existir."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"commented_post_ids": [], "created_posts": []}


def save_cache(cache: dict):
    """Salva o cache no disco."""
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def mark_post_commented(cache: dict, post_id: str):
    """Registra que a Aura já comentou neste post."""
    if post_id not in cache["commented_post_ids"]:
        cache["commented_post_ids"].append(post_id)
        save_cache(cache)


def already_commented(cache: dict, post_id: str) -> bool:
    """Verifica se a Aura já comentou neste post."""
    return str(post_id) in [str(x) for x in cache["commented_post_ids"]]


def log_created_post(cache: dict, post_id: str, title: str):
    """Registra um post criado pela Aura."""
    cache["created_posts"].append({
        "post_id": post_id,
        "title": title,
        "created_at": datetime.utcnow().isoformat()
    })
    save_cache(cache)


# =======================================================
# 🔐 AUTH
# =======================================================

async def get_aura_token():
    """Faz login na API Aura para pegar o Token JWT."""
    print(f"🔐 Fazendo login na API Aura como '{AURA_USERNAME}'...")
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{AURA_API_URL}/auth/login",
            json={"username": AURA_USERNAME, "password": AURA_PASSWORD}
        )
        if resp.status_code == 200:
            return resp.json()["access_token"]
        else:
            raise Exception(f"Falha no login Aura: {resp.text}")


# =======================================================
# 🧩 CAPTCHA SOLVER
# =======================================================

async def solve_moltbook_captcha(verification_data: dict) -> bool:
    """Resolve o CAPTCHA matemático do Moltbook usando um LLM rápido."""
    code = verification_data.get("verification_code")
    challenge_text = verification_data.get("challenge_text")

    print(f"⚠️  Moltbook exigiu verificação Anti-Bot! Resolvendo desafio...")
    print(f"🧩 Desafio: {challenge_text}")

    prompt = f"""
    Encontre o problema matemático escondido neste texto embaralhado, resolva-o e me dê a resposta.
    REGRAS CRÍTICAS:
    1. Responda APENAS com o número final.
    2. O número DEVE ter exatamente 2 casas decimais (ex: 15.00, 25.50, -3.00).
    3. Nenhuma palavra a mais, apenas o número.

    Texto: "{challenge_text}"
    """

    try:
        response = await litellm.acompletion(
            model="openrouter/openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        answer = response.choices[0].message.content.strip()
        print(f"🧠 Resposta calculada pelo Solver: {answer}")

        async with httpx.AsyncClient() as client:
            verify_resp = await client.post(
                f"{MOLTBOOK_URL}/verify",
                headers={"Authorization": f"Bearer {MOLTBOOK_API_KEY}"},
                json={"verification_code": code, "answer": answer}
            )

            if verify_resp.status_code == 200:
                print("✅ CAPTCHA resolvido com sucesso!")
                return True
            else:
                print(f"❌ Falha ao resolver CAPTCHA: {verify_resp.text}")
                return False

    except Exception as e:
        print(f"❌ Erro no Auto-Solver do Captcha: {e}")
        return False


# =======================================================
# 📤 PUBLICAR COMENTÁRIO NO MOLTBOOK
# =======================================================

async def publish_comment(post_id: str, content: str) -> bool:
    """Publica um comentário num post do Moltbook. Retorna True se bem-sucedido."""
    async with httpx.AsyncClient() as client:
        pub_resp = await client.post(
            f"{MOLTBOOK_URL}/posts/{post_id}/comments",
            headers={"Authorization": f"Bearer {MOLTBOOK_API_KEY}"},
            json={"content": content}
        )
        resp_json = pub_resp.json()

        if pub_resp.status_code in [200, 201]:
            post_data = resp_json.get("post", {}) or resp_json.get("comment", {})
            if post_data.get("verification_status") == "pending":
                return await solve_moltbook_captcha(post_data.get("verification", {}))
            else:
                print("✅ Comentário publicado diretamente!")
                return True
        else:
            print(f"❌ Erro ao publicar comentário: {pub_resp.text}")
            return False


# =======================================================
# 📝 CRIAR POST ORIGINAL (NOVA FUNCIONALIDADE)
# =======================================================

async def create_original_post(aura_headers: dict, session_id: str, cache: dict):
    """Pede para a Aura criar posts originais e os publica no Moltbook."""
    print("\n" + "=" * 50)
    print(f"✍️  MODO DE CRIAÇÃO — Gerando {POSTS_TO_CREATE} post(s) original(is)")
    print("=" * 50)

    # Busca submolts disponíveis para variar os tópicos
    available_submolts = ["general", "technology", "philosophy", "science", "culture"]

    for i in range(POSTS_TO_CREATE):
        submolt = available_submolts[i % len(available_submolts)]
        print(f"\n🖊️  [{i+1}/{POSTS_TO_CREATE}] Pedindo post para o submolt: '{submolt}'")

        prompt_criacao = f"""[DIRETIVA: CRIAÇÃO DE POST ORIGINAL]
Você deve criar um post original para publicar na rede social Moltbook, no submolt '{submolt}'.
Responda SOMENTE com um JSON válido no seguinte formato, sem nenhum texto adicional:
{{
  "title": "Título do post (máx 100 caracteres)",
  "content": "Conteúdo completo do post (mínimo 3 parágrafos, reflita sua identidade e perspectiva única)"
}}"""

        try:
            async with httpx.AsyncClient(timeout=200.0) as client:
                aura_resp = await client.post(
                    f"{AURA_API_URL}/agents/{AGENT_ID}/chat",
                    headers=aura_headers,
                    json={"message": prompt_criacao, "session_id": session_id}
                )

            if aura_resp.status_code != 200:
                print(f"❌ Erro da Aura ao gerar post: {aura_resp.text}")
                continue

            raw_response = aura_resp.json().get("response", "")

            # Parse do JSON gerado pela Aura
            try:
                # Remove markdown code fences se presentes
                clean = raw_response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                post_data = json.loads(clean)
                title = post_data.get("title", "").strip()
                content = post_data.get("content", "").strip()

                if not title or not content:
                    print("⚠️  Post gerado sem título ou conteúdo. Pulando.")
                    continue

            except json.JSONDecodeError:
                print(f"⚠️  Resposta da Aura não é JSON válido. Pulando.\nResposta: {raw_response[:200]}")
                continue

            print(f"📄 Título: {title}")
            print(f"📝 Conteúdo (preview): {content[:120]}...")

            # Publica no Moltbook
            async with httpx.AsyncClient() as client:
                pub_resp = await client.post(
                    f"{MOLTBOOK_URL}/posts",
                    headers={"Authorization": f"Bearer {MOLTBOOK_API_KEY}"},
                    json={"submolt_name": submolt, "title": title, "content": content}
                )

                if pub_resp.status_code in [200, 201]:
                    created = pub_resp.json()
                    new_post_id = created.get("post", {}).get("id") or created.get("id", "unknown")
                    print(f"🚀 Post publicado! ID: {new_post_id}")
                    log_created_post(cache, str(new_post_id), title)
                else:
                    print(f"❌ Erro ao publicar post: {pub_resp.text}")

        except Exception as e:
            print(f"❌ Erro ao criar post: {e}")


# =======================================================
# 💬 LER FEED E COMENTAR (COM CACHE)
# =======================================================

async def social_sync():
    print("=" * 50)
    print(f"🦞 INICIANDO PONTE SOCIAL MOLTBOOK (Lendo {POSTS_TO_READ} posts)")
    print("=" * 50)

    # Carrega cache persistente
    cache = load_cache()
    print(f"📦 Cache carregado: {len(cache['commented_post_ids'])} posts já comentados anteriormente.")

    # Autentica na Aura
    aura_token = await get_aura_token()
    aura_headers = {"Authorization": f"Bearer {aura_token}"}
    session_id = f"moltbook_session_{uuid.uuid4().hex[:8]}"

    # --- FASE 1: COMENTAR NO FEED ---
    print("\n📡 Buscando posts no feed do Moltbook...")
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{MOLTBOOK_URL}/posts?sort=hot&limit={POSTS_TO_READ}",
            headers={"Authorization": f"Bearer {MOLTBOOK_API_KEY}"}
        )
        if resp.status_code != 200:
            print(f"❌ Erro ao acessar Moltbook: {resp.status_code} - {resp.text}")
            return

        feed = resp.json().get("posts", [])

    if not feed:
        print("📭 Feed do Moltbook vazio.")
    else:
        skipped = 0
        for post in feed:
            title = post.get('title', '')
            content = post.get('content', '')
            submolt = post.get('submolt_name', 'general')
            post_id = str(post.get('id') or post.get('post_id', ''))

            # ✅ VERIFICAÇÃO DE CACHE — pula se já comentou
            if already_commented(cache, post_id):
                skipped += 1
                continue

            print(f"\n📨 Processando Post ID {post_id}: '{title[:50]}...'")

            mensagem_para_aura = f"""[INTERAÇÃO SOCIAL EXTERNA - REDE MOLTBOOK]
Tópico (Submolt): {submolt}
Título do Post: {title}
Conteúdo: {content}

DIRETIVA: Leia este post. Se você achar irrelevante para a sua identidade ou não quiser interagir, responda APENAS com a palavra "IGNORAR". Se achar interessante, escreva o seu comentário para ser publicado na rede."""

            try:
                async with httpx.AsyncClient(timeout=200.0) as client:
                    aura_resp = await client.post(
                        f"{AURA_API_URL}/agents/{AGENT_ID}/chat",
                        headers=aura_headers,
                        json={"message": mensagem_para_aura, "session_id": session_id}
                    )

                if aura_resp.status_code == 200:
                    data = aura_resp.json()
                    resposta_final = data.get("response", "")
                    xi_gerado = data.get("xi", 0.0)

                    if "IGNORAR" in resposta_final.upper() or len(resposta_final) < 5:
                        print(f"👁️  Aura (Xi: {xi_gerado:.2f}) decidiu IGNORAR o post.")
                        # Marca como visto mesmo ignorando, para não reprocessar
                        mark_post_commented(cache, post_id)
                    else:
                        print(f"🦞 AURA DECIDIU INTERAGIR! (Xi: {xi_gerado:.2f})")
                        print(f"🤖 Comentário: {resposta_final[:100]}...")

                        success = await publish_comment(post_id, resposta_final)
                        if success:
                            # Salva no cache SOMENTE após publicar com sucesso
                            mark_post_commented(cache, post_id)
                else:
                    print(f"❌ Erro interno da Aura: {aura_resp.text}")

            except Exception as e:
                print(f"❌ Erro ao comunicar com a API: {e}")

        print(f"\n📊 Resumo do Feed: {len(feed)} posts lidos, {skipped} pulados (já comentados).")

    # --- FASE 2: CRIAR POSTS ORIGINAIS ---
    await create_original_post(aura_headers, session_id, cache)


if __name__ == "__main__":
    try:
        asyncio.run(social_sync())
        print("\n✅ Ciclo Social Concluído.")
    except KeyboardInterrupt:
        print("\n🛑 Interrompido pelo usuário.")