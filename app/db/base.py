from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """
    Common declarative base for all database models.
    All SQLAlchemy models must inherit from this class.
    """
    pass
