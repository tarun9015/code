import asyncio

from sqlalchemy import text
from app.db.session import AsyncSessionLocal


async def main():
    async with AsyncSessionLocal() as session:

        result = await session.execute(
            text("SELECT current_database(), current_user")
        )
        print("Database:", result.fetchone())

        result = await session.execute(
            text("""
                SELECT version_num
                FROM alembic_version
            """)
        )

        print("Recorded migration revisions:")
        for row in result.fetchall():
            print(row[0])


if __name__ == "__main__":
    asyncio.run(main())
