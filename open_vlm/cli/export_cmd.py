import click
from open_vlm.integration.datasette_utils import DatasetteExporter
from typing import Optional
import json
from pathlib import Path

@click.command()
@click.argument("results_file", type=click.Path(exists=True))
@click.option("--output", "-o", default="vlm_outputs.db", help="Output SQLite file")
@click.option("--table", "-t", default="results", help="Table name to use")
@click.option("--metadata", "-m", type=click.Path(), help="JSON file with table metadata")
@click.option("--format", "-f", type=click.Choice(["sqlite", "json", "csv"]), 
              default="sqlite", help="Output format")
def export(results_file: str, output: str, table: str, metadata: Optional[str], format: str):
    """Export VLM results to various formats with optional metadata"""
    from open_vlm.utils import load_json_or_jsonl
    
    try:
        results = load_json_or_jsonl(results_file)
        meta = {}
        if metadata:
            with open(metadata) as f:
                meta = json.load(f)
        
        if format == "sqlite":
            exporter = DatasetteExporter(output)
            exporter.export_results(results, table_name=table, metadata=meta)
        elif format == "json":
            with open(output, 'w') as f:
                json.dump(results, f)
        elif format == "csv":
            import pandas as pd
            pd.DataFrame(results).to_csv(output, index=False)
            
        click.echo(f"Successfully exported {len(results)} results to {output}")
    except Exception as e:
        click.echo(f"Error exporting results: {str(e)}", err=True)
        raise click.Abort() 