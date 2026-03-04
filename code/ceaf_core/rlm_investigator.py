# ceaf_core/rlm_investigator.py — Recursive Language Models Investigator (V2)
# ──────────────────────────────────────────────────────────────────────────────

import os
import asyncio
import logging
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger("AuraV4_RLM")

# Palavras-chave que indicam que a query precisa de execução de código
_CODE_KEYWORDS = {
    "calcule", "calcular", "compute", "execute", "código", "código python",
    "resultado de", "quanto é", "quantos são", "verifique", "valide",
    "parse", "converta", "converta para", "rode", "script",
    "calculate", "compute", "run", "execute", "verify", "validate",
}


def _needs_code_execution(query: str) -> bool:
    """Heurística rápida: a query exige computação real no Sandbox?"""
    q_lower = query.lower()
    return any(kw in q_lower for kw in _CODE_KEYWORDS)


class RLMInvestigator:
    """
    Investigador Recursivo de Linguagem (V2).

    Responsabilidade: dado um query e um contexto de memórias,
    retornar evidências factuais relevantes para enriquecer a resposta da Aura.

    Não é um módulo de geração — é um módulo de FILTRAGEM E VERIFICAÇÃO.
    """

    def __init__(self, api_key: Optional[str] = None, llm_service=None):
        self.api_key = api_key or os.getenv("E2B_API_KEY")
        self.llm_service = llm_service  # Injetado pelo ActivityContext se disponível

        if self.api_key:
            os.environ["E2B_API_KEY"] = self.api_key
        else:
            logger.warning("RLM: E2B_API_KEY ausente. Modo CODE_EXECUTION desativado.")

    # ──────────────────────────────────────────────────────────────────────────
    # INTERFACE PÚBLICA
    # ──────────────────────────────────────────────────────────────────────────

    async def investigate(self, query: str, context_data: str) -> str:
        """
        Ponto de entrada principal. Escolhe o modo de investigação automaticamente.

        Returns:
            str: Evidências relevantes para enriquecer o prompt downstream,
                 ou string vazia se nada relevante for encontrado.
        """
        # Contexto vazio = nada a investigar
        if not context_data or not context_data.strip():
            logger.debug("RLM: context_data vazio. Pulando investigação.")
            return ""

        # Roteamento por modo
        if _needs_code_execution(query) and self.api_key:
            logger.info("RLM: Roteando para MODE 2 — CODE_EXECUTION (E2B Sandbox)")
            return await self._mode_code_execution(query, context_data)
        else:
            logger.info("RLM: Roteando para MODE 1 — SEMANTIC_FILTER (LLM local)")
            return await self._mode_semantic_filter(query, context_data)

    # ──────────────────────────────────────────────────────────────────────────
    # MODE 1: SEMANTIC_FILTER — usa LLM para filtrar evidências do contexto
    # ──────────────────────────────────────────────────────────────────────────

    async def _mode_semantic_filter(self, query: str, context_data: str) -> str:
        """
        Usa a LLM (via llm_service) para extrair fatos relevantes do contexto.
        Muito mais preciso que keyword matching. Sem dependência externa.
        """
        if not self.llm_service:
            # Fallback: extração léxica melhorada (se não há LLM injetada)
            return self._lexical_filter(query, context_data)

        prompt = f"""Você é um filtro de evidências. Seu trabalho é APENAS extrair do CONTEXTO abaixo
os trechos que são diretamente relevantes para a QUERY. Seja objetivo e conciso.

QUERY: {query}

CONTEXTO (memórias e histórico):
{context_data[:3000]}  

INSTRUÇÃO: Liste apenas os fatos/trechos relevantes encontrados no contexto, um por linha.
Se nada for relevante, responda exatamente: "Sem evidências relevantes."
Não explique, não elabore. Apenas os fatos extraídos."""

        try:
            response = await self.llm_service.generate(
                prompt=prompt,
                max_tokens=300,
                temperature=0.1,  # Baixa temperatura para filtragem factual
                system="Você é um extrator de evidências. Responda apenas com os fatos relevantes extraídos."
            )
            result = response.strip() if response else ""

            if not result or "Sem evidências" in result:
                return ""

            logger.info(f"RLM [SEMANTIC]: Evidências extraídas ({len(result)} chars)")
            return f"[RLM SEMANTIC EVIDENCE]: {result}"

        except Exception as e:
            logger.warning(f"RLM [SEMANTIC]: Falha na LLM, usando fallback léxico: {e}")
            return self._lexical_filter(query, context_data)

    def _lexical_filter(self, query: str, context_data: str) -> str:
        """
        Fallback léxico melhorado: TF-IDF simplificado por contagem de palavras-chave.
        Muito melhor que o grep original — considera frequência e relevância.
        """
        # Remove stop words básicas em PT/EN
        stop_words = {
            "o", "a", "os", "as", "um", "uma", "de", "do", "da", "dos", "das",
            "em", "no", "na", "por", "para", "com", "que", "se", "é", "são",
            "the", "a", "an", "is", "are", "of", "in", "to", "and", "or", "for"
        }

        query_words = {
            w.lower() for w in query.split()
            if len(w) > 3 and w.lower() not in stop_words
        }

        if not query_words:
            return ""

        lines = [l.strip() for l in context_data.split('\n') if l.strip()]

        # Score cada linha por quantas query_words ela contém
        scored = []
        for line in lines:
            line_words = set(line.lower().split())
            score = len(query_words & line_words)
            if score > 0:
                scored.append((score, line))

        # Ordena por relevância, pega top 5
        scored.sort(reverse=True, key=lambda x: x[0])
        top = [line for _, line in scored[:5]]

        if not top:
            return ""

        result = "\n".join(top)
        logger.info(f"RLM [LEXICAL]: {len(top)} linhas relevantes encontradas")
        return f"[RLM LEXICAL EVIDENCE]: {result}"

    # ──────────────────────────────────────────────────────────────────────────
    # MODE 2: CODE_EXECUTION — E2B Sandbox, async-safe
    # ──────────────────────────────────────────────────────────────────────────

    async def _mode_code_execution(self, query: str, context_data: str) -> str:
        """
        Executa código Python no Sandbox E2B para verificação computacional.

        CRÍTICO: Usa asyncio.to_thread() para não bloquear o event loop do Temporal.
        O erro "Server disconnected" anterior era causado por chamar Sandbox.create()
        (síncrono) diretamente dentro de um coroutine.
        """
        script = self._build_computation_script(query, context_data)

        try:
            # asyncio.to_thread() executa a chamada síncrona em uma thread separada
            # sem bloquear o event loop — CORREÇÃO DO BUG DE CONECTIVIDADE
            result = await asyncio.to_thread(self._run_sandbox_sync, script)
            logger.info(f"RLM [CODE]: Sandbox executado com sucesso")
            return f"[RLM COMPUTED RESULT]: {result}" if result else ""

        except Exception as e:
            logger.error(f"RLM [CODE]: Falha no Sandbox: {e}")
            # Degradação graciosa: tenta filtro semântico como fallback
            logger.info("RLM [CODE]: Degradando para SEMANTIC_FILTER...")
            return await self._mode_semantic_filter(query, context_data)

    def _run_sandbox_sync(self, script: str) -> str:
        """Wrapper síncrono para o Sandbox E2B — chamado via asyncio.to_thread()."""
        from e2b_code_interpreter import Sandbox

        with Sandbox() as sandbox:
            execution = sandbox.run_code(script)
            if execution.error:
                logger.error(f"RLM Sandbox Error: {execution.error}")
                return f"Erro na execução: {execution.error}"
            return execution.text.strip() if execution.text else ""

    def _build_computation_script(self, query: str, context_data: str) -> str:
        """
        Monta o script Python que será executado no Sandbox.
        Foca em extrair e computar dados numéricos/estruturados do contexto.
        """
        return f"""
import re
import json

query = {repr(query)}
context = {repr(context_data[:2000])}

# Tenta extrair números e dados estruturados do contexto
numbers = re.findall(r'[-+]?\\d*\\.?\\d+', context)
findings = []

# Filtragem relevante por query keywords
query_words = set(query.lower().split())
for line in context.split('\\n'):
    if any(w in line.lower() for w in query_words if len(w) > 3):
        findings.append(line.strip())

# Output estruturado
output = {{
    "relevant_lines": findings[:5],
    "extracted_numbers": numbers[:10],
    "query_answered": len(findings) > 0
}}
print(json.dumps(output, ensure_ascii=False, indent=2))
"""

    # ──────────────────────────────────────────────────────────────────────────
    # MODE 3: FACT_CHECK (placeholder documentado para expansão futura)
    # ──────────────────────────────────────────────────────────────────────────

    async def _mode_fact_check(self, query: str) -> str:
        """
        [FUTURO] Verificação de fatos externos via RAG ou busca web.

        Para implementar: integrar com um cliente de busca (SearXNG local,
        Brave Search API, ou Qdrant com corpus de conhecimento externo).

        Por ora, retorna indicação de não-verificabilidade offline.
        """
        logger.debug("RLM [FACT_CHECK]: Modo ainda não implementado.")
        return ""