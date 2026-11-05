
from contextlib import asynccontextmanager

from sqlalchemy.engine import make_url
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.core.config import settings


def get_checkpoint_database_url() -> str:
    """
    Convert the SQLAlchemy asyncpg URL into a
    PostgreSQL URL compatible with Psycopg.
    """

    database_url = make_url(settings.DATABASE_URL)

    database_url = database_url.set(
        drivername="postgresql"
    )

    return database_url.render_as_string(
        hide_password=False
    )


@asynccontextmanager
async def get_checkpointer():
    """
    Provide an asynchronous PostgreSQL checkpointer.

    Keeps the database connection alive while
    the graph is using the checkpointer.
    """

    database_url = get_checkpoint_database_url()

    async with AsyncPostgresSaver.from_conn_string(
        database_url
    ) as checkpointer:

        await checkpointer.setup()

        yield checkpointer
