import asyncio

from sqlalchemy import text

from app.db.session import AsyncSessionLocal, close_db


async def test_connection():
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                text("SELECT current_database(), current_user")
            )

            database, user = result.fetchone()

            print("Database connection successful!")
            print("Database:", database)
            print("User:", user)

    except Exception as exc:
        print("Database connection failed:", str(exc))
        raise

    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(test_connection())
