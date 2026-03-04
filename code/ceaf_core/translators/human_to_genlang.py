# ceaf_core/translators/human_to_genlang.py

import asyncio
import json
import re
from typing import List, Dict

from pydantic import ValidationError
from ceaf_core.genlang_types import IntentPacket, GenlangVector
from ceaf_core.utils.embedding_utils import get_embedding_client
from ceaf_core.services.llm_service import LLMService
from ceaf_core.utils.common_utils import extract_json_from_text
import logging
from ceaf_core.models import SystemPrompts, LLMConfig

logger = logging.getLogger("CEAFv3_System")


class HumanToGenlangTranslator:
    def __init__(self, prompts: SystemPrompts = None, llm_config: LLMConfig = None):
        self.embedding_client = get_embedding_client()
        self.llm_service = LLMService(config=llm_config)
        self.prompts = prompts or SystemPrompts()

    def update_prompts(self, new_prompts: SystemPrompts):
        self.prompts = new_prompts

    async def contextualize_query(self, query: str, chat_history: List[Dict[str, str]]) -> str:
        """
        🔧 VERSÃO CORRIGIDA - Anti-Alucinação

        Reescreve a query do usuário para incluir o contexto do histórico recente.
        Isso resolve referências como 'ele', 'isso', 'o anterior'.

        CORREÇÕES IMPLEMENTADAS:
        1. Prompt mais restritivo (força output limpo)
        2. Validação de saída (detecta meta-texto)
        3. Fallback robusto (volta para query original se falhar)
        4. Limite de tokens para evitar verbosidade
        """
        if not chat_history:
            return query

        # Pega os últimos 3 turnos para contexto imediato
        recent_history = chat_history[-6:]
        history_text = "\n".join(
            [f"{msg['role']}: {msg['content'][:200]}" for msg in recent_history])  # ← LIMITA CADA MSG A 200 chars

        # ═══════════════════════════════════════════════════════════════
        # 🔒 PROMPT BLINDADO - Versão Anti-Meta-Texto
        # ═══════════════════════════════════════════════════════════════
        prompt = f"""TASK: Rewrite user query using conversation context ONLY if needed.

RULES:
1. If query contains pronouns (he, she, it, that, this one) → replace with actual nouns from history
2. If query is standalone/clear → return it UNCHANGED
3. Output ONLY the rewritten text - NO explanations, NO "Contextualized:", NO analysis

HISTORY:
{history_text}

QUERY: {query}

REWRITTEN_QUERY:"""

        try:
            # 🔧 CORREÇÃO: max_tokens reduzido para evitar verbosidade
            contextualized = await self.llm_service.ainvoke(
                self.llm_service.config.fast_model,
                prompt,
                temperature=0.0,
                max_tokens=850  # ← NOVO: Limita resposta
            )

            # ═══════════════════════════════════════════════════════════════
            # 🛡️ VALIDAÇÃO DE SAÍDA - Detecta meta-texto
            # ═══════════════════════════════════════════════════════════════
            clean_query = contextualized.strip()

            # Remove prefixos comuns de meta-texto
            meta_prefixes = [
                "Pergunta Contextualizada:",
                "Rewritten Query:",
                "REWRITTEN_QUERY:",
                "A Nova Pergunta",
                "Conclusão:",
                "Ação:",
                "Output:"
            ]

            for prefix in meta_prefixes:
                if clean_query.startswith(prefix):
                    clean_query = clean_query[len(prefix):].strip()

            # 🚨 DETECÇÃO DE ALUCINAÇÃO: Se contém análise em vez de query
            hallucination_indicators = [
                "não é uma pergunta",
                "é um trecho de texto",
                "mantenha como está",
                "análise:",
                "observação:"
            ]

            is_hallucinating = any(indicator in clean_query.lower() for indicator in hallucination_indicators)

            # 🚨 DETECÇÃO DE VERBOSIDADE: Se resposta é muito maior que query original
            is_too_verbose = len(clean_query) > len(query) * 3

            # ═══════════════════════════════════════════════════════════════
            # 🔄 FALLBACK ROBUSTO
            # ═══════════════════════════════════════════════════════════════
            if is_hallucinating or is_too_verbose:
                logger.warning(
                    f"HTG Contextualizer hallucinação detectada. "
                    f"Hallucinating: {is_hallucinating}, Verbose: {is_too_verbose}. "
                    f"Revertendo para query original."
                )
                return query

            # Se passou nas validações, usa a versão contextualizada
            logger.info(f"HTG Context OK: '{query[:50]}' → '{clean_query[:50]}'")
            return clean_query

        except Exception as e:
            logger.error(f"Erro na contextualização: {e}. Usando query original.")
            return query  # ← FALLBACK SEGURO

    async def translate(self, query: str, metadata: dict, chat_history: List[Dict[str, str]] = None) -> IntentPacket:
        """
        Versão V1.3: Blindagem completa contra alucinação do HTG.
        """
        logger.info(f"--- [HTG Translator v1.3] Analisando query humana: '{query[:50]}...' ---")

        # ========================================================================
        # PASSO 1: CONTEXTUALIZAR A QUERY (COM VALIDAÇÃO ANTI-ALUCINAÇÃO)
        # ========================================================================
        refined_query = await self.contextualize_query(query, chat_history)

        # ========================================================================
        # PASSO 2: CRIAR PROMPT DE ANÁLISE (usando refined_query)
        # ========================================================================
        try:
            analysis_prompt = self.prompts.htg_analysis.format(query=refined_query)
        except KeyError:
            logger.warning("Prompt HTG mal formatado pelo usuário (KeyError). Usando concatenação simples.")
            analysis_prompt = self.prompts.htg_analysis + f"\nUser Query: {refined_query}"

        # ========================================================================
        # PASSO 3: INVOCAR LLM PARA ANÁLISE
        # ========================================================================
        model_to_use = self.llm_service.config.fast_model if hasattr(self.llm_service,
                                                                     'config') and self.llm_service.config else "gpt-3.5-turbo"

        analysis_json = None
        analysis_str = await self.llm_service.ainvoke(
            model_to_use,
            analysis_prompt,
            temperature=0.0,
            max_tokens=500  # ← NOVO: Limita análise também
        )

        # ========================================================================
        # PASSO 4: PROCESSAR RESPOSTA JSON DA LLM
        # ========================================================================
        try:
            extracted_json = extract_json_from_text(analysis_str)
            if isinstance(extracted_json, dict):
                required_keys = ["core_query", "intent_description", "emotional_tone_description", "key_entities"]
                if all(key in extracted_json for key in required_keys):
                    analysis_json = extracted_json
                else:
                    logger.warning(
                        f"HTG Translator: Invalid JSON structure (missing keys). Raw: '{analysis_str[:150]}'")
            else:
                logger.warning(
                    f"HTG Translator: Failed to extract a dictionary from LLM response. Raw: '{analysis_str[:150]}'")

        except Exception as e:
            logger.error(f"HTG Translator: Exception during JSON parsing. Error: {e}. Raw: '{analysis_str[:150]}'")

        # ========================================================================
        # PASSO 5: FALLBACK APRIMORADO
        # ========================================================================
        if not analysis_json:
            logger.error("HTG Translator: Falha na análise da LPU. Usando fallback aprimorado.")
            fallback_keywords = list(set(re.findall(r'\b\w{3,15}\b', refined_query.lower())))
            analysis_json = {
                "core_query": refined_query,
                "intent_description": "unknown_intent",
                "emotional_tone_description": "neutral",  # ← MUDADO: 'unknown' → 'neutral' (mais semântico)
                "key_entities": fallback_keywords[:5]  # ← AUMENTADO: 3 → 5 keywords
            }

        # ========================================================================
        # PASSO 6: GERAR EMBEDDINGS
        # ========================================================================
        texts_to_embed = [
                             analysis_json.get("core_query", refined_query),
                             analysis_json.get("intent_description", "unknown"),
                             analysis_json.get("emotional_tone_description", "neutral")
                         ] + analysis_json.get("key_entities", [])

        embeddings = await self.embedding_client.get_embeddings(texts_to_embed, context_type="default_query")

        # ========================================================================
        # PASSO 7: CRIAR VETORES GENLANG
        # ========================================================================
        model_name = self.embedding_client._resolve_model_name("default_query")

        query_vector = GenlangVector(
            vector=embeddings[0],
            source_text=analysis_json.get("core_query", refined_query),
            model_name=model_name
        )

        intent_vector = GenlangVector(
            vector=embeddings[1],
            source_text=analysis_json.get("intent_description", "unknown"),
            model_name=model_name
        )

        emotional_vector = GenlangVector(
            vector=embeddings[2],
            source_text=analysis_json.get("emotional_tone_description", "neutral"),
            model_name=model_name
        )

        entity_vectors = [
            GenlangVector(
                vector=emb,
                source_text=text,
                model_name=model_name
            )
            for text, emb in zip(analysis_json.get("key_entities", []), embeddings[3:])
        ]

        # ========================================================================
        # PASSO 8: PRESERVAR METADADOS (original e contextualizada)
        # ========================================================================
        metadata['original_query'] = query
        metadata['contextualized_query'] = refined_query

        # ========================================================================
        # PASSO 9: CRIAR E RETORNAR INTENT PACKET
        # ========================================================================
        intent_packet = IntentPacket(
            query_vector=query_vector,
            intent_vector=intent_vector,
            emotional_valence_vector=emotional_vector,
            entity_vectors=entity_vectors,
            metadata=metadata
        )

        logger.info(
            f"--- [HTG Translator] Análise completa. "
            f"Intenção: '{intent_vector.source_text}', "
            f"Entidades: {[e.source_text for e in entity_vectors]} ---"
        )

        return intent_packet