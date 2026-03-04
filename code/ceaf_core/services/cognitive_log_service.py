# ceaf_core/services/cognitive_log_service.py

import sqlite3
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger("CognitiveLogService")

LOG_DB_FILENAME = "cognitive_turn_history.sqlite"


class CognitiveLogService:
    """
    Armazena e consulta o histórico de pacotes Genlang de cada turno.
    [AURA V5]: Agora armazena a termodinâmica do ego (Tensão e Autenticidade).
    """

    def __init__(self, persistence_path: Path):
        if isinstance(persistence_path, str):
            persistence_path = Path(persistence_path)
        if not persistence_path.exists():
            try:
                persistence_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"📁 Pasta criada: {persistence_path}")
            except Exception as e:
                logger.error(f"❌ Falha ao criar pasta de logs: {e}")

        self.db_path = persistence_path / LOG_DB_FILENAME
        self._initialize_db()
        logger.info(f"CognitiveLogService V5 inicializado em: {self.db_path}")

    def _get_db_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _initialize_db(self):
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                # Tabela Original
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS turn_history (
                        turn_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        cognitive_state_packet TEXT NOT NULL,
                        response_packet TEXT NOT NULL,
                        mcl_guidance_json TEXT,
                        deliberation_history TEXT, 
                        intent_text TEXT,
                        final_confidence REAL,
                        agency_used INTEGER,
                        action_vector TEXT
                    )
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON turn_history (timestamp);")

                # [MIGRAÇÃO V5] Adicionando colunas de Autenticidade
                new_columns = [
                    ("tension_before", "REAL"),
                    ("tension_after", "REAL"),
                    ("authenticity_score", "REAL"),
                    # [MIGRAÇÃO V5.4] Snapshot do Ego no momento do turno
                    # Resolve o problema do referencial móvel no Dreamer:
                    # agora sabemos "quem a Aura era" quando aquele score foi calculado.
                    ("identity_vector_snapshot", "TEXT"),
                ]
                for col_name, col_type in new_columns:
                    try:
                        cursor.execute(f"ALTER TABLE turn_history ADD COLUMN {col_name} {col_type}")
                        logger.info(f"🔧 Migração V5: Coluna {col_name} adicionada ao log.")
                    except sqlite3.OperationalError:
                        pass  # Coluna já existe

                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Falha ao inicializar banco de dados do CognitiveLogService: {e}", exc_info=True)
            raise

    def log_turn(
            self,
            turn_id: str,
            session_id: str,
            cognitive_state_packet: Dict[str, Any],
            response_packet: Dict[str, Any],
            mcl_guidance: Dict[str, Any],
            action_vector: Optional[List[float]] = None,
            tension_before: float = 0.0,
            tension_after: float = 0.0,
            authenticity_score: float = 0.0,
            # [V5.4] O glyph PRÉ-síntese: quem a Aura ERA antes de responder.
            # NÃO usar o updated_glyph (pós-assimilação), pois o authenticity_score
            # foi calculado com base neste vetor, não no evoluído.
            identity_snapshot: Optional[List[float]] = None
    ):
        """Registra o turno completo com a métrica de Autenticidade V5."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                agency_used_flag = 1 if mcl_guidance.get("agency_parameters", {}).get("use_agency_simulation") else 0
                mcl_guidance_json_str = json.dumps(mcl_guidance, default=str)
                deliberation_history_list = cognitive_state_packet.get("deliberation_history", [])
                deliberation_history_json_str = json.dumps(deliberation_history_list)

                action_vector_str = json.dumps(action_vector) if action_vector else None
                identity_snapshot_str = json.dumps(identity_snapshot) if identity_snapshot else None

                cursor.execute(
                    """
                    INSERT INTO turn_history (
                        turn_id, session_id, timestamp, 
                        cognitive_state_packet, response_packet,
                        mcl_guidance_json, deliberation_history,
                        intent_text, final_confidence, agency_used,
                        action_vector, tension_before, tension_after, authenticity_score,
                        identity_vector_snapshot
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        turn_id, session_id, time.time(),
                        json.dumps(cognitive_state_packet, default=str),
                        json.dumps(response_packet, default=str),
                        mcl_guidance_json_str, deliberation_history_json_str,
                        cognitive_state_packet.get("original_intent", {}).get("query_vector", {}).get("source_text"),
                        response_packet.get("confidence_score"), agency_used_flag,
                        action_vector_str, tension_before, tension_after, authenticity_score,
                        identity_snapshot_str
                    ),
                )
                conn.commit()
                logger.debug(f"Turno '{turn_id}' logado (AuthScore: {authenticity_score:.4f})")
        except sqlite3.Error as e:
            logger.error(f"Falha ao registrar turno '{turn_id}' no log: {e}")

    def get_recent_turns(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Recupera os N turnos mais recentes para análise de forma robusta."""
        results = []
        try:
            with self._get_db_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                # <<< MUDANÇA: ATUALIZA O SELECT STATEMENT >>>
                cursor.execute(
                    "SELECT * FROM turn_history ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                )
                # <<< FIM DA MUDANÇA >>>
                rows = cursor.fetchall()

                for row in rows:
                    try:
                        row_dict = dict(row)
                        row_dict["cognitive_state_packet"] = json.loads(row_dict["cognitive_state_packet"])
                        row_dict["response_packet"] = json.loads(row_dict["response_packet"])

                        if row_dict.get("mcl_guidance_json"):
                            row_dict["mcl_guidance"] = json.loads(row_dict["mcl_guidance_json"])
                        else:
                            row_dict["mcl_guidance"] = {}

                        if row_dict.get("deliberation_history"):
                            row_dict["deliberation_history"] = json.loads(row_dict["deliberation_history"])
                        else:
                            row_dict["deliberation_history"] = []

                        # [V5.4] Deserializa o snapshot histórico do ego
                        if row_dict.get("identity_vector_snapshot"):
                            row_dict["identity_vector_snapshot"] = json.loads(row_dict["identity_vector_snapshot"])
                        else:
                            row_dict["identity_vector_snapshot"] = None

                        results.append(row_dict)
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(
                            f"Pulando turno de log corrompido (ID: {row.get('turn_id', 'N/A')}) devido a erro de parsing: {e}"
                        )
                        continue

        except sqlite3.Error as e:
            logger.error(f"Falha na consulta ao banco de dados de logs: {e}")

        return results