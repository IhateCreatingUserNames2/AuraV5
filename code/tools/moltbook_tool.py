import httpx
import os
import logging
from typing import Any

logger = logging.getLogger("MoltbookTool")


class MoltbookClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.moltbook.com/api/v1"

    async def post(self, submolt: str, title: str, content: str):
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/posts",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"submolt_name": submolt, "title": title, "content": content}
            )
            return resp.json()

    # Movi para fora da classe, pois é uma ferramenta de Agência
    @staticmethod
    async def moltbook_post_tool(title: str, content: str, submolt: str = "general"):
        """
        Ferramenta para Aura postar no Moltbook.
        """
        api_key = os.getenv("MOLTBOOK_API_KEY")
        if not api_key:
            return {"status": "error", "response": "API Key não configurada"}

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://www.moltbook.com/api/v1/posts",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"submolt_name": submolt, "title": title, "content": content}
            )
            return {"status": "success" if resp.status_code == 201 else "error", "response": resp.text}