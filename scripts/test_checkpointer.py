
import asyncio

from app.memory.checkpointer import get_checkpointer


async def main():

    print("\nPOSTGRES CHECKPOINTER TEST")
    print("=" * 50)

    async with get_checkpointer() as checkpointer:

        print("PostgreSQL connection: SUCCESS")
        print("Checkpoint setup: SUCCESS")
        print(f"Checkpointer: {type(checkpointer).__name__}")

    print("\nCheckpointer connection closed successfully.")
    print("PostgreSQL checkpointer test passed!")


if __name__ == "__main__":
    asyncio.run(main())
