# api/deps.py
import logging
import jwt
from typing import AsyncGenerator, Callable
from fastapi import Depends, HTTPException, status, Path
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from database.models import DatabaseSetup, AgentRepository
from ceaf_core.services.state_manager import StateManager
from ceaf_core.services.mbs_memory_service import MBSMemoryService
from ceaf_core.system import CEAFSystem
from agent_manager import AgentManager

# --- Core Infrastructure Dependencies ---
_agent_manager_instance = None

JWT_SECRET = "your-secret-key-change-this-in-production"
JWT_ALGORITHM = "HS256"
security = HTTPBearer()


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Valida o token JWT e retorna os dados do usuário."""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return {"user_id": payload["user_id"], "username": payload["username"]}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expirado")
    except (jwt.InvalidTokenError, jwt.PyJWTError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token inválido")

def get_agent_manager() -> AgentManager:
    global _agent_manager_instance
    if _agent_manager_instance is None:
        _agent_manager_instance = AgentManager()
    return _agent_manager_instance


async def get_db() -> AsyncGenerator:
    """Provides a transactional database session."""
    session_maker = DatabaseSetup.get_session_maker()
    async with session_maker() as session:
        yield session


async def get_repository() -> AgentRepository:
    """Provides the Data Access Layer."""
    return AgentRepository()


async def get_memory_service() -> MBSMemoryService:
    """Provides the Vector Memory Service."""
    return MBSMemoryService()


async def get_state_manager() -> StateManager:
    """Provides the Redis State Manager."""
    return StateManager()


# --- Application Logic Dependencies ---

async def get_ceaf_factory() -> Callable[[dict], CEAFSystem]:
    """
    Returns a factory function to create CEAFSystem instances manually.
    Useful for WebSockets or background tasks where path parameters aren't standard.
    """

    def _factory(agent_config: dict) -> CEAFSystem:
        return CEAFSystem(agent_config)

    return _factory


async def get_active_ceaf_system(
        agent_id: str = Path(..., description="The ID of the agent to interact with"),
        repo: AgentRepository = Depends(get_repository)
) -> CEAFSystem:
    """
    REAL APP DEPENDENCY:
    1. Extracts 'agent_id' from the URL.
    2. Queries PostgreSQL to validate the agent exists.
    3. Builds the configuration dictionary.
    4. Returns the initialized CEAFSystem client.

    Raises:
        HTTP 404: If the agent does not exist.
    """
    agent = await repo.get_agent(agent_id)

    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent with ID '{agent_id}' not found."
        )

    # Convert SQL Model to Config Dict
    agent_config = {
        "agent_id": agent.id,
        "name": agent.name,
        "persona": agent.detailed_persona,
        "model": agent.model,
        # Map other DB fields to config as needed
        "settings": agent.settings if hasattr(agent, "settings") else {}
    }

    return CEAFSystem(agent_config)