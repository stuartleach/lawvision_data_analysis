from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, Engine


def create_engine_connection(user: str, password: str, host: str, port: str, dbname: str) -> Engine:
    """
    Create a connection to the database.

    :param user: Database user.
    :param password: Database password.
    :param host: Database host.
    :param port: Database port.
    :param dbname: Database name.
    :return: SQLAlchemy Engine.
    """
    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(connection_string)


def run_migrations():
    """
    Run Alembic migrations.
    """
    # Load the Alembic configuration
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")
