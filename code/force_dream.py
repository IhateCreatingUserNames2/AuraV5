# force_dream.py
import asyncio
import os
import uuid
from temporalio.client import Client

# Configuração
TEMPORAL_HOST = "localhost:7233"
TASK_QUEUE = "ceaf-cognitive-queue-v2"


async def main():
    print(f"🔌 Conectando ao Temporal em {TEMPORAL_HOST}...")
    client = await Client.connect(TEMPORAL_HOST)

    workflow_id = f"dream-force-{uuid.uuid4()}"

    print("💤 Iniciando DreamingWorkflow forçado...")

    # Dispara o workflow. Não esperamos o resultado (fire_and_forget)
    # ou esperamos (execute_workflow) dependendo da preferência.
    # O DreamingWorkflow retorna None, então podemos esperar.

    try:
        await client.execute_workflow(
            "DreamingWorkflow",
            args=[],  # O workflow não pede argumentos no run()
            id=workflow_id,
            task_queue=TASK_QUEUE
        )
        print("✅ Ciclo de Sonho concluído com sucesso!")
        print("👀 Verifique os logs da Soul Engine (VastAI) agora.")
    except Exception as e:
        print(f"❌ Erro ao executar sonho: {e}")


if __name__ == "__main__":
    asyncio.run(main())