"""Tests for the Datasette export functionality."""

import os
import json
import tempfile
import sqlite3
from pathlib import Path

import pytest
from unittest.mock import patch, MagicMock

from open_vlm.integration.datasette_utils import DatasetteExporter

@pytest.fixture
def sample_results():
    """Fixture to generate sample results for testing."""
    return [
        {
            "image_id": "img_001",
            "prompt": "Describe this gear assembly",
            "response": "The image shows a gear assembly with two interlocking gears.",
            "confidence": 0.95,
            "processing_time": 1.25
        },
        {
            "image_id": "img_002",
            "prompt": "What is the diameter of the larger gear?",
            "response": "The larger gear has a diameter of approximately 5 cm.",
            "confidence": 0.87,
            "processing_time": 1.42,
            "measurements": {"diameter": 5.0, "unit": "cm"}
        }
    ]

@pytest.fixture
def sample_metadata():
    """Fixture for sample table metadata."""
    return {
        "title": "VLM Analysis Results",
        "description": "Results from analyzing engineering images with OpenVLM",
        "source": "OpenVLM test suite",
        "license": "Apache-2.0"
    }

def test_exporter_initialization():
    """Test that the exporter can be initialized properly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        exporter = DatasetteExporter(db_path)
        assert exporter.db_path == db_path
        assert not db_path.exists()  # Database should not be created until export is called

def test_export_results(sample_results):
    """Test exporting results to a SQLite database."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_export.db"
        
        # Export the results
        exporter = DatasetteExporter(db_path)
        exporter.export_results(sample_results, table_name="test_results")
        
        # Verify the database was created
        assert db_path.exists()
        
        # Connect to the database and check the contents
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if the table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test_results'")
        assert cursor.fetchone() is not None
        
        # Check if all rows were exported
        cursor.execute("SELECT COUNT(*) FROM test_results")
        count = cursor.fetchone()[0]
        assert count == len(sample_results)
        
        # Check if columns were created correctly
        cursor.execute("PRAGMA table_info(test_results)")
        columns = [info[1] for info in cursor.fetchall()]
        assert "image_id" in columns
        assert "prompt" in columns
        assert "response" in columns
        assert "confidence" in columns
        assert "processing_time" in columns
        assert "measurements" in columns
        
        # Check the data content
        cursor.execute("SELECT image_id, prompt, response, measurements FROM test_results WHERE image_id = 'img_002'")
        row = cursor.fetchone()
        assert row[0] == "img_002"
        assert row[1] == "What is the diameter of the larger gear?"
        assert row[2] == "The larger gear has a diameter of approximately 5 cm."
        assert json.loads(row[3]) == {"diameter": 5.0, "unit": "cm"}
        
        conn.close()

def test_export_with_metadata(sample_results, sample_metadata):
    """Test exporting results with metadata."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_metadata.db"
        
        # Export the results with metadata
        exporter = DatasetteExporter(db_path)
        exporter.export_results(sample_results, table_name="results_with_meta", metadata=sample_metadata)
        
        # Connect to the database and check for metadata
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if the metadata table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='_datasette_metadata'")
        assert cursor.fetchone() is not None
        
        # Check if metadata was stored correctly
        cursor.execute("SELECT value FROM _datasette_metadata WHERE key='table:results_with_meta'")
        metadata_json = cursor.fetchone()[0]
        stored_metadata = json.loads(metadata_json)
        
        assert stored_metadata["title"] == sample_metadata["title"]
        assert stored_metadata["description"] == sample_metadata["description"]
        assert stored_metadata["source"] == sample_metadata["source"]
        assert stored_metadata["license"] == sample_metadata["license"]
        
        conn.close()

def test_append_results(sample_results):
    """Test appending results to an existing table."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_append.db"
        
        # Create the initial database with the first result
        exporter = DatasetteExporter(db_path)
        exporter.export_results([sample_results[0]], table_name="append_test")
        
        # Append the second result
        exporter.append_results([sample_results[1]], table_name="append_test")
        
        # Connect to the database and check if both entries are there
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM append_test")
        count = cursor.fetchone()[0]
        assert count == 2
        
        conn.close()

def test_create_view():
    """Test creating a view in the database."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_view.db"
        
        # Create a database with some results
        exporter = DatasetteExporter(db_path)
        results = [
            {"id": 1, "category": "A", "value": 10},
            {"id": 2, "category": "B", "value": 20},
            {"id": 3, "category": "A", "value": 30}
        ]
        exporter.export_results(results, table_name="data")
        
        # Create a view
        exporter.create_view(
            view_name="category_summary",
            sql_query="SELECT category, SUM(value) as total FROM data GROUP BY category"
        )
        
        # Check if the view exists and has correct data
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='view' AND name='category_summary'")
        assert cursor.fetchone() is not None
        
        cursor.execute("SELECT category, total FROM category_summary ORDER BY category")
        results = cursor.fetchall()
        assert results[0] == ("A", 40)
        assert results[1] == ("B", 20)
        
        conn.close()

@patch("subprocess.Popen")
def test_launch_datasette(mock_popen):
    """Test the launch_datasette method."""
    db_path = "test_launch.db"
    DatasetteExporter.launch_datasette(db_path, port=8001, open_browser=True)
    
    # Check if subprocess.Popen was called with the right command
    mock_popen.assert_called_once()
    args, kwargs = mock_popen.call_args
    cmd = args[0]
    
    assert "datasette" in cmd
    assert "test_launch.db" in cmd
    assert "--open" in cmd
    assert "--port 8001" in cmd 