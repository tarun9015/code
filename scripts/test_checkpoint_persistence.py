
import asyncio
from uuid import uuid4

from app.memory.checkpointer import get_checkpointer


async def main():

    thread_id = f"checkpoint-test-{uuid4()}"

    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
        }
    }

    print("\nCHECKPOINT PERSISTENCE TEST")
    print("=" * 50)

    async with get_checkpointer() as checkpointer:

        # Step 1: Prepare checkpoint data.
        checkpoint = {
            "v": 1,
            "id": str(uuid4()),
            "ts": "2026-11-04T00:00:00+00:00",
            "channel_values": {
                "messages": [
                    "Hello from Enterprise Agentic RAG"
                ],
                "response": "Checkpoint saved successfully",
            },
            "channel_versions": {
                "messages": 1,
                "response": 1,
            },
            "versions_seen": {},
        }

        # Step 2: Save checkpoint.
        saved_config = await checkpointer.aput(
            config,
            checkpoint,
            {
                "source": "update",
                "step": 1,
                "writes": None,
            },
            {},
        )

        print("Checkpoint save: SUCCESS")

        # Step 3: Retrieve saved checkpoint.
        saved_checkpoint = await checkpointer.aget(
            saved_config
        )

        if saved_checkpoint is None:
            raise RuntimeError("Checkpoint was not found")

        print("Checkpoint retrieval: SUCCESS")

        # Step 4: Verify saved data.
        channel_values = saved_checkpoint["channel_values"]

        print("\nSaved checkpoint values:")
        print(channel_values)

        assert (
            channel_values["response"]
            == "Checkpoint saved successfully"
        )

        print("\nCheckpoint persistence test passed!")


if __name__ == "__main__":
    asyncio.run(main())
