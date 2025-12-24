"""
Example Tools - Reusable tools for pydantic-collab examples

This module provides a collection of tools used across various examples:
- File operations (read/write text and JSON)
- Python code execution
- Dataset analysis
- Search and summarization utilities
"""
import json
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_ai import RunContext, Tool


# =============================================================================
# Data Models
# =============================================================================

class FileContent(BaseModel):
    """Content of a file with metadata."""
    path: str
    content: str
    size: int


class FileOperationResult(BaseModel):
    """Result of a file operation."""
    success: bool
    filepath: str
    message: str
    size_bytes: int = 0


class DataStats(BaseModel):
    """Statistics about a dataset."""
    total_records: int
    columns: list[str]
    missing_values: dict[str, int]
    summary: str


class AnalysisResult(BaseModel):
    """Results from statistical analysis."""
    metric_name: str
    value: float
    interpretation: str


# =============================================================================
# File Operations
# =============================================================================

def write_file_tool(ctx: RunContext, filepath: str, content: str) -> str:
    """Write content to a file.
    
    Args:
        filepath: Path to the file to write
        content: Content to write to the file
        
    Returns:
        Success message with file path
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return f"✓ Successfully wrote {len(content)} bytes to {filepath}"


def read_file_tool(ctx: RunContext, filepath: str) -> FileContent:
    """Read content from a file.
    
    Args:
        filepath: Path to the file to read
        
    Returns:
        FileContent object with file details
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    content = path.read_text()
    return FileContent(
        path=str(path),
        content=content,
        size=len(content)
    )


def write_text_tool(ctx: RunContext, filepath: str, content: str) -> FileOperationResult:
    """Write text content to a file.
    
    Args:
        filepath: Path to the file to write
        content: Text content to write
        
    Returns:
        FileOperationResult with operation details
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    
    return FileOperationResult(
        success=True,
        filepath=str(path),
        message=f'Successfully wrote {len(content)} characters',
        size_bytes=len(content)
    )


def read_text_tool(ctx: RunContext, filepath: str) -> str:
    """Read text content from a file.
    
    Args:
        filepath: Path to the file to read
        
    Returns:
        Text content of the file
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f'File not found: {filepath}')
    
    return path.read_text()


def modify_file_tool(ctx: RunContext, filepath: str, old_content: str, new_content: str) -> str:
    """Modify a file by replacing old content with new content.
    
    Args:
        filepath: Path to the file to modify
        old_content: Content to search for and replace
        new_content: New content to insert
        
    Returns:
        Success message
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    content = path.read_text()
    
    if old_content not in content:
        return f"⚠️  Warning: old_content not found in file. No changes made."
    
    modified = content.replace(old_content, new_content, 1)
    path.write_text(modified)
    
    return f"✓ Modified {filepath}: replaced {len(old_content)} chars with {len(new_content)} chars"


# =============================================================================
# JSON Operations
# =============================================================================

def write_json_tool(ctx: RunContext, filepath: str, data: dict[str, Any]) -> FileOperationResult:
    """Write JSON data to a file.
    
    Args:
        filepath: Path to the JSON file to write
        data: Dictionary to write as JSON
        
    Returns:
        FileOperationResult with operation details
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    json_str = json.dumps(data, indent=2)
    path.write_text(json_str)
    
    return FileOperationResult(
        success=True,
        filepath=str(path),
        message=f"Successfully wrote JSON data",
        size_bytes=len(json_str)
    )


def read_json_tool(ctx: RunContext, filepath: str) -> dict[str, Any]:
    """Read JSON data from a file.
    
    Args:
        filepath: Path to the JSON file to read
        
    Returns:
        Dictionary with the JSON data
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    return json.loads(path.read_text())


# =============================================================================
# Code Execution
# =============================================================================

def run_python_code(ctx: RunContext, filepath: str, args: str = "") -> str:
    """Run a Python script and return its output.
    
    Args:
        filepath: Path to the Python file to run
        args: Command-line arguments to pass to the script
        
    Returns:
        Combined stdout and stderr from the script execution
    """
    try:
        result = subprocess.run(
            ["uv", "run", filepath] + (args.split() if args else []),
            capture_output=True,
            text=True,
            timeout=10
        )
        
        output = []
        if result.stdout:
            output.append(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            output.append(f"STDERR:\n{result.stderr}")
        if result.returncode != 0:
            output.append(f"\n⚠️  Exit code: {result.returncode}")
        else:
            output.append("\n✓ Exit code: 0 (success)")
            
        return "\n".join(output) if output else "(no output)"
        
    except subprocess.TimeoutExpired:
        return "❌ Script execution timed out (10s limit)"
    except Exception as e:
        return f"❌ Error running script: {e}"


def run_python_script_tool(ctx: RunContext, filepath: str) -> str:
    """Run a Python script and return its output (extended timeout).
    
    Args:
        filepath: Path to the Python file to run
        
    Returns:
        Combined stdout and stderr from execution
    """
    try:
        result = subprocess.run(
            ["uv", "run", filepath],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(Path(filepath).parent)
        )
        
        output = []
        if result.stdout:
            output.append(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            output.append(f"STDERR:\n{result.stderr}")
        if result.returncode != 0:
            output.append(f"\n⚠️  Exit code: {result.returncode}")
        else:
            output.append("\n✓ Exit code: 0 (success)")
            
        return "\n".join(output) if output else "(no output)"
        
    except subprocess.TimeoutExpired:
        return "❌ Script execution timed out (30s limit)"
    except Exception as e:
        return f"❌ Error running script: {e}"


# =============================================================================
# Data Analysis
# =============================================================================

def analyze_dataset_tool(ctx: RunContext, filepath: str) -> DataStats:
    """Analyze a JSON dataset and return statistics.
    
    Args:
        filepath: Path to the JSON file containing dataset
        
    Returns:
        DataStats with dataset statistics
    """
    data = json.loads(Path(filepath).read_text())
    
    if not isinstance(data, list) or not data:
        return DataStats(
            total_records=0,
            columns=[],
            missing_values={},
            summary="Empty or invalid dataset"
        )
    
    # Get columns from first record
    columns = list(data[0].keys()) if data else []
    
    # Count missing values
    missing = {col: 0 for col in columns}
    for record in data:
        for col in columns:
            if col not in record or record[col] is None or record[col] == "":
                missing[col] += 1
    
    summary = f"Dataset contains {len(data)} records with {len(columns)} columns"
    
    return DataStats(
        total_records=len(data),
        columns=columns,
        missing_values=missing,
        summary=summary
    )


def calculate_statistics_tool(ctx: RunContext, filepath: str) -> list[AnalysisResult]:
    """Calculate statistical metrics from a dataset.
    
    Args:
        filepath: Path to the JSON file containing cleaned dataset
        
    Returns:
        List of AnalysisResult objects with computed metrics
    """
    data = json.loads(Path(filepath).read_text())
    results = []
    
    if not data or not isinstance(data, list):
        return results
    
    # Calculate total sales
    total_sales = sum(record.get('amount', 0) for record in data)
    results.append(AnalysisResult(
        metric_name="Total Sales",
        value=total_sales,
        interpretation=f"Total revenue across all transactions"
    ))
    
    # Calculate average sale
    avg_sale = total_sales / len(data) if data else 0
    results.append(AnalysisResult(
        metric_name="Average Sale",
        value=avg_sale,
        interpretation=f"Mean transaction value"
    ))
    
    # Count unique products
    products = set(record.get('product', 'Unknown') for record in data)
    results.append(AnalysisResult(
        metric_name="Unique Products",
        value=len(products),
        interpretation=f"Number of different products sold"
    ))
    
    # Count unique regions
    regions = set(record.get('region', 'Unknown') for record in data)
    results.append(AnalysisResult(
        metric_name="Unique Regions",
        value=len(regions),
        interpretation=f"Number of sales regions"
    ))
    
    return results


# =============================================================================
# Utility Tools
# =============================================================================

async def fake_search(q: str) -> str:
    """Simulated web search tool.
    
    Args:
        q: Search query
        
    Returns:
        Fake search results
    """
    return f"Search results for: {q} -> [fact1, fact2, fact3]"


async def fake_tool(q: str) -> str:
    """Generic fake tool for testing.
    
    Args:
        q: Query string
        
    Returns:
        Formatted tool response
    """
    return f"TOOL({q})"


async def summarize(text: str) -> str:
    """Simple text summarization tool.
    
    Args:
        text: Text to summarize
        
    Returns:
        Summarized text
    """
    return "Summary: " + (text[:200] + '...' if len(text) > 200 else text)


# =============================================================================
# Pre-configured Tool Objects
# =============================================================================

search_tool = Tool(fake_search, takes_ctx=False, description="Simulated web search")
generic_tool = Tool(fake_tool, takes_ctx=False, description="Generic test tool")
summarize_tool = Tool(summarize, takes_ctx=False, description="Simple summarizer")
