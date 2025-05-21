"""Utilities for exporting OpenVLM results to Datasette."""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

class DatasetteExporter:
    """Exports OpenVLM results to SQLite databases compatible with Datasette.
    
    This class facilitates the integration between OpenVLM and Datasette by providing
    utilities to convert VLM output data (e.g., analyses, extracted information from
    images) into SQLite databases that can be explored and shared using Datasette.
    """
    
    def __init__(self, db_path: Union[str, Path]):
        """Initialize the DatasetteExporter.
        
        Args:
            db_path: Path to the SQLite database file to create or use
        """
        self.db_path = Path(db_path)
        self._ensure_parent_dir()
    
    def _ensure_parent_dir(self) -> None:
        """Ensure the parent directory of the database file exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def export_results(
        self, 
        results: List[Dict[str, Any]], 
        table_name: str = "results",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Export a list of result objects to a SQLite database.
        
        Args:
            results: List of dictionaries containing the results to export
            table_name: Name of the table to create in the database
            metadata: Optional metadata to associate with the table
                (will be stored in a special Datasette metadata table)
        """
        if not results:
            raise ValueError("No results provided for export")
        
        # Connect to the database and create the table
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Determine schema from the first result
        first_item = results[0]
        columns = []
        placeholders = []
        
        for key, value in first_item.items():
            if isinstance(value, (int, float)):
                dtype = "REAL" if isinstance(value, float) else "INTEGER"
            elif isinstance(value, dict) or isinstance(value, list):
                dtype = "TEXT"  # Store JSON as text
            else:
                dtype = "TEXT"
            
            columns.append(f'"{key}" {dtype}')
            placeholders.append("?")
        
        # Create the table
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS "{table_name}" (
            {', '.join(columns)}
        )
        """
        cursor.execute(create_table_sql)
        
        # Insert the data
        insert_sql = f"""
        INSERT INTO "{table_name}" VALUES ({', '.join(placeholders)})
        """
        
        for result in results:
            # Convert any JSON objects to strings
            values = []
            for key in first_item.keys():
                value = result.get(key)
                if isinstance(value, (dict, list)):
                    values.append(json.dumps(value))
                else:
                    values.append(value)
            
            cursor.execute(insert_sql, values)
        
        # Add metadata if provided (using Datasette's metadata table approach)
        if metadata:
            cursor.execute("CREATE TABLE IF NOT EXISTS _datasette_metadata (key TEXT PRIMARY KEY, value TEXT)")
            cursor.execute(
                "INSERT OR REPLACE INTO _datasette_metadata VALUES (?, ?)",
                (f"table:{table_name}", json.dumps(metadata))
            )
        
        conn.commit()
        conn.close()
    
    def append_results(
        self, 
        results: List[Dict[str, Any]], 
        table_name: str = "results"
    ) -> None:
        """Append results to an existing table in the database.
        
        Args:
            results: List of dictionaries containing the results to append
            table_name: Name of the existing table to append to
        """
        if not results:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        if not cursor.fetchone():
            raise ValueError(f"Table '{table_name}' does not exist in the database")
        
        # Get column names from the table
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [info[1] for info in cursor.fetchall()]
        
        # Prepare placeholders for insertion
        placeholders = ", ".join(["?"] * len(columns))
        
        # Insert the data
        for result in results:
            values = []
            for col in columns:
                value = result.get(col)
                if isinstance(value, (dict, list)):
                    values.append(json.dumps(value))
                else:
                    values.append(value)
            
            cursor.execute(f"INSERT INTO {table_name} VALUES ({placeholders})", values)
        
        conn.commit()
        conn.close()
    
    def create_view(
        self, 
        view_name: str, 
        sql_query: str
    ) -> None:
        """Create a view in the database based on a SQL query.
        
        Args:
            view_name: Name of the view to create
            sql_query: SQL query that defines the view
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        create_view_sql = f"CREATE VIEW IF NOT EXISTS {view_name} AS {sql_query}"
        cursor.execute(create_view_sql)
        
        conn.commit()
        conn.close()
    
    @staticmethod
    def launch_datasette(
        db_path: Union[str, Path], 
        port: int = 8001,
        open_browser: bool = True
    ) -> None:
        """Launch Datasette to explore the database.
        
        Note: Requires datasette to be installed.
        
        Args:
            db_path: Path to the SQLite database file
            port: Port to run Datasette on
            open_browser: Whether to automatically open browser
        """
        import subprocess
        import sys
        
        try:
            open_flag = "--open" if open_browser else ""
            cmd = f"{sys.executable} -m datasette {db_path} {open_flag} --port {port}"
            subprocess.Popen(cmd, shell=True)
            print(f"Datasette launched at http://localhost:{port}")
        except Exception as e:
            print(f"Error launching Datasette: {str(e)}")
            print("Make sure Datasette is installed: pip install datasette") 