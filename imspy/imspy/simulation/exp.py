import sqlite3
import os

import numpy as np
import pandas as pd


class ExperimentDataHandle:
    def __init__(self,
                 path: str,
                 db_name: str = 'experiment_data.db',
                 verbose: bool = True,
                 ):
        self.verbose = verbose
        self.base_path = path
        self.db_path = os.path.join(self.base_path, db_name)
        self.raw_data_path = os.path.join(self.base_path, 'raw_data')
        self.conn = None

        self._setup()

    def _setup(self):
        # Create raw_data directory if it doesn't exist
        if not os.path.exists(self.raw_data_path):
            if self.verbose:
                print(f"Creating raw data directory: {self.raw_data_path}")
            os.makedirs(self.raw_data_path)

        # Connect to the SQLite database or create it if it doesn't exist
        if self.verbose:
            print(f"Connecting to database: {self.db_path}")
        self.conn = sqlite3.connect(self.db_path)

    def create_table(self, table_name: str, table: pd.DataFrame):
        # Create a table from a pandas DataFrame
        table.to_sql(table_name, self.conn, if_exists='replace', index=False)

    def create_table_sql(self, sql):
        # Create a table as per the provided SQL statement
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql)
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"An error occurred: {e}")

    def close(self):
        # Close the database connection
        if self.conn:
            self.conn.close()

    def get_table(self, table_name: str) -> pd.DataFrame:
        # Get a table as a pandas DataFrame
        return pd.read_sql(f"SELECT * FROM {table_name}", self.conn)


if __name__ == '__main__':

    # Example usage
    path = '/path/to/directory'
    db_name = 'experiment_data.db'
    handle = ExperimentDataHandle(path, db_name)

    # Create a table, for example
    sql_create_peptides_table = '''
    CREATE TABLE IF NOT EXISTS peptides (
        peptide_id INTEGER PRIMARY KEY,
        sequence TEXT NOT NULL,
        monoisotopic_mass REAL)
    '''
    handle.create_table(sql_create_peptides_table)

    # Close the connection when done
    handle.close()
