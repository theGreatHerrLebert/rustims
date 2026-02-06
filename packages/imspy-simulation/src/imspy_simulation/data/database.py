"""Simulation database access layer.

This module provides classes for reading and writing simulation data
to SQLite databases. The main class is SimulationDatabase, which handles
all interactions with the synthetic_data.db file.
"""

import logging
import os
import sqlite3
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class SimulationDatabase:
    """SQLite database interface for simulation data.

    This class manages the synthetic_data.db file that stores all
    simulation artifacts including peptides, ions, fragment ions,
    frame metadata, and acquisition settings.

    The database is automatically created if it doesn't exist, and
    indexes are created on commonly-queried columns to optimize
    lazy loading queries.

    Attributes:
        base_path: Base directory containing the database.
        database_path: Full path to the database file.
    """

    def __init__(
        self,
        database_path: str,
        database_name: str = "synthetic_data.db",
        verbose: bool = False,
    ):
        """Initialize database connection.

        Args:
            database_path: Directory to store the database.
            database_name: Name of the database file.
            verbose: If True, log verbose messages.
        """
        self._verbose = verbose
        self.base_path = database_path
        self.database_path = os.path.join(self.base_path, database_name)
        self._conn: Optional[sqlite3.Connection] = None

        self._setup()

    def _setup(self):
        """Create directory and connect to database."""
        if not os.path.exists(self.base_path):
            logger.info(f"Creating data directory: {self.base_path}")
            os.makedirs(self.base_path)

        logger.debug(f"Connecting to database: {self.database_path}")
        self._conn = sqlite3.connect(self.database_path)

    @property
    def connection(self) -> sqlite3.Connection:
        """Get the database connection.

        Returns:
            Active SQLite connection.

        Raises:
            RuntimeError: If database is closed.
        """
        if self._conn is None:
            raise RuntimeError("Database connection is closed.")
        return self._conn

    def create_table(self, table_name: str, table: pd.DataFrame) -> None:
        """Create a table from a pandas DataFrame.

        Replaces existing table if it exists. Automatically creates
        indexes for efficient lazy loading queries based on table name.

        Args:
            table_name: Name for the table.
            table: DataFrame to store.
        """
        table.to_sql(table_name, self.connection, if_exists="replace", index=False)
        self._create_indexes_for_table(table_name)

    def _create_indexes_for_table(self, table_name: str) -> None:
        """Create indexes optimized for lazy loading queries.

        Args:
            table_name: Name of the table to index.
        """
        indexes = {
            "peptides": [
                "CREATE INDEX IF NOT EXISTS idx_peptides_frame_range ON peptides(frame_occurrence_start, frame_occurrence_end)",
                "CREATE INDEX IF NOT EXISTS idx_peptides_peptide_id ON peptides(peptide_id)",
            ],
            "ions": [
                "CREATE INDEX IF NOT EXISTS idx_ions_peptide_id ON ions(peptide_id)",
            ],
            "fragment_ions": [
                "CREATE INDEX IF NOT EXISTS idx_fragment_ions_peptide_id ON fragment_ions(peptide_id)",
                "CREATE INDEX IF NOT EXISTS idx_fragment_ions_lookup ON fragment_ions(peptide_id, charge)",
            ],
        }

        if table_name in indexes:
            cursor = self.connection.cursor()
            for sql in indexes[table_name]:
                try:
                    cursor.execute(sql)
                except sqlite3.Error as e:
                    logger.warning(f"Could not create index: {e}")
            self.connection.commit()

    def append_table(self, table_name: str, table: pd.DataFrame) -> None:
        """Append data to an existing table.

        Args:
            table_name: Name of the table.
            table: DataFrame to append.
        """
        table.to_sql(table_name, self.connection, if_exists="append", index=False)

    def create_table_sql(self, sql: str) -> None:
        """Create a table using raw SQL.

        Args:
            sql: CREATE TABLE SQL statement.
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(sql)
            self.connection.commit()
        except sqlite3.Error as e:
            logger.error(f"SQL error: {e}")
            raise

    def get_table(self, table_name: str) -> pd.DataFrame:
        """Read a table as a DataFrame.

        Args:
            table_name: Name of the table.

        Returns:
            Table contents as DataFrame.
        """
        return pd.read_sql(f"SELECT * FROM {table_name}", self.connection)

    def get_frame_meta_data(self) -> pd.DataFrame:
        """Get frame metadata table.

        Returns:
            Frame metadata DataFrame.
        """
        return self.get_table("frames")

    def list_tables(self) -> List[str]:
        """List all tables in the database.

        Returns:
            List of table names.
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [row[0] for row in cursor.fetchall()]

    def list_columns(self, table_name: str) -> List[str]:
        """List columns in a table.

        Args:
            table_name: Name of the table.

        Returns:
            List of column names.

        Raises:
            ValueError: If table doesn't exist.
        """
        if table_name not in self.list_tables():
            raise ValueError(f"Table '{table_name}' does not exist.")

        cursor = self.connection.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        return [row[1] for row in cursor.fetchall()]

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.debug("Database connection closed.")

    def __enter__(self) -> "SimulationDatabase":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        return f"SimulationDatabase(path={self.database_path})"


class SimulationDatabaseDIA(SimulationDatabase):
    """Extended database interface with DIA-specific functionality.

    Adds methods for accessing DIA window settings and MS/MS info
    that are specific to Data-Independent Acquisition mode.

    Attributes:
        dia_ms_ms_info: Cached DIA MS/MS info DataFrame.
        dia_ms_ms_windows: Cached DIA MS/MS windows DataFrame.
    """

    def __init__(
        self,
        database_path: str,
        database_name: str = "synthetic_data.db",
        verbose: bool = False,
    ):
        """Initialize DIA database connection.

        Args:
            database_path: Directory containing the database.
            database_name: Name of the database file.
            verbose: If True, log verbose messages.
        """
        super().__init__(database_path, database_name, verbose)
        self.dia_ms_ms_info: Optional[pd.DataFrame] = None
        self.dia_ms_ms_windows: Optional[pd.DataFrame] = None
        self._load_dia_metadata()

    def _load_dia_metadata(self) -> None:
        """Load DIA-specific metadata tables."""
        try:
            self.dia_ms_ms_info = self.get_table("dia_ms_ms_info")
            self.dia_ms_ms_windows = self.get_table("dia_ms_ms_windows")
        except Exception as e:
            logger.warning(f"Could not load DIA metadata: {e}")

    def get_frame_to_window_group(self) -> dict:
        """Get mapping from frame IDs to window groups.

        Returns:
            Dictionary mapping frame_id -> window_group.
        """
        if self.dia_ms_ms_info is None:
            self._load_dia_metadata()

        if self.dia_ms_ms_info is None:
            return {}

        return dict(zip(self.dia_ms_ms_info.frame, self.dia_ms_ms_info.window_group))

    def get_window_group_settings(self) -> dict:
        """Get DIA window settings.

        Returns:
            Dictionary mapping (window_group, scan_start) -> (mz_mid, mz_width).
        """
        if self.dia_ms_ms_windows is None:
            self._load_dia_metadata()

        if self.dia_ms_ms_windows is None:
            return {}

        settings = {}
        for _, row in self.dia_ms_ms_windows.iterrows():
            key = (row.window_group, row.scan_start)
            value = (row.mz_mid, row.mz_width)
            settings[key] = value
        return settings

    def __repr__(self) -> str:
        return f"SimulationDatabaseDIA(path={self.database_path})"
