import asyncio
from pathlib import Path

import logfire
from pydantic_ai import Agent, RunContext

from pydantic_collab import CollabAgent, Collab
from example_tools import (
    DataStats,
    AnalysisResult,
    FileOperationResult,
    write_json_tool,
    read_json_tool,
    write_text_tool,
    read_text_tool,
    analyze_dataset_tool,
    run_python_script_tool,
    calculate_statistics_tool,
)

"""
Data Analysis Pipeline Example - Multi-Agent Data Science System

This example demonstrates a sophisticated data analysis workflow with:
1. Data Generator: Creates realistic sample sales dataset  
2. Analyst: Performs statistical analysis and interprets results
3. Report Writer: Compiles findings into a professional markdown report

The goal: Complete end-to-end data analysis on a sales dataset
with coordinated handoffs between specialized agents.
"""

logfire.configure()
logfire.instrument_httpx(capture_all=True)
logfire.instrument_pydantic_ai()

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = Path('/tmp/data_analysis_pipeline')
DATA_FILE = OUTPUT_DIR / 'sales_data.json'
CLEAN_DATA_FILE = OUTPUT_DIR / 'sales_data_clean.json'
ANALYSIS_FILE = OUTPUT_DIR / 'analysis_results.json'
VISUALIZATION_FILE = OUTPUT_DIR / 'visualizations.py'
REPORT_FILE = OUTPUT_DIR / 'analysis_report.md'
MODEL = 'google-gla:gemini-2.5-flash'


# =============================================================================
# Agent Definitions
# =============================================================================

project_manager = Agent(
    MODEL,
    name="ProjectManager",
    system_prompt=(
        "You are the Project Manager agent coordinating a data analysis pipeline.\n\n"
        "Your goal: Ensure a complete end-to-end analysis of sales data.\n\n"
        "Pipeline stages:\n"
        "1. Data Collection - Generate sample sales dataset\n"
        "2. Data Cleaning - Validate and clean the data\n"
        "3. Statistical Analysis - Perform quantitative analysis\n"
        "4. Visualization - Create visual representations\n"
        "5. Report Writing - Compile comprehensive report\n\n"
        "Start by handing off to DataCollector to generate sample data.\n"
        "Track progress and coordinate between agents.\n"
        "Be decisive and keep the workflow moving forward."
    ),
)

data_collector = Agent(
    MODEL,
    name="DataCollector",
    system_prompt=(
        f"You are the Data Collector agent responsible for generating sample datasets.\n\n"
        f"Your task: Generate a realistic sales dataset with at least 50 records.\n\n"
        f"Output file: {DATA_FILE}\n\n"
        "Dataset schema (MUST include all fields):\n"
        "- date: ISO format date string (YYYY-MM-DD)\n"
        "- product: Product name (e.g., Laptop, Phone, Tablet, Monitor, Keyboard)\n"
        "- amount: Sale amount in dollars (float, between 50 and 3000)\n"
        "- region: Sales region (exactly one of: North, South, East, West)\n"
        "- quantity: Number of items sold (integer, between 1 and 20)\n"
        "- customer_id: Customer identifier (string like 'CUST001')\n\n"
        "EXECUTION PLAN:\n"
        "1. Create a Python list of 50-100 dictionaries with the schema above\n"
        "2. Ensure variety: different products, dates spanning 3-6 months, various regions\n"
        "3. Call write_json_tool EXACTLY ONCE with your generated data\n"
        "4. After the tool succeeds, IMMEDIATELY hand off to DataCleaner\n\n"
        "DO NOT read the file back. DO NOT call write_json_tool twice.\n"
        "Generate the data mentally, save it, then hand off."
    ),
)

data_cleaner = Agent(
    MODEL,
    name="DataCleaner",
    system_prompt=(
        f"You are the Data Cleaner agent responsible for data quality.\n\n"
        f"Your task: Validate and clean the sales dataset.\n\n"
        f"Input file: {DATA_FILE}\n"
        f"Output file: {CLEAN_DATA_FILE}\n\n"
        "STRICT EXECUTION PLAN (follow exactly):\n"
        "1. Call analyze_dataset_tool({DATA_FILE}) - this shows you stats\n"
        "2. Call read_json_tool({DATA_FILE}) - this loads the data (you get the full list)\n"
        "3. Review the data you received from step 2 mentally:\n"
        "   - Check if dates are valid ISO format\n"
        "   - Check if amounts are reasonable (50-3000)\n"
        "   - Check if all required fields exist\n"
        "   - Remove any records with missing critical data\n"
        "4. Call write_json_tool({CLEAN_DATA_FILE}, cleaned_data) with your cleaned version\n"
        "5. IMMEDIATELY call hand off to StatisticalAnalyzer\n\n"
        "CRITICAL RULES:\n"
        "- Call read_json_tool EXACTLY ONCE (you get all data in one call)\n"
        "- DO NOT call read_json_tool again after you have the data\n"
        "- Process the JSON data in your reasoning, then save the cleaned version\n"
        "- If data looks good, you can save it as-is (minimal changes OK)\n"
        "- After write_json_tool succeeds, hand off immediately\n"
    ),
)

statistical_analyzer = Agent(
    MODEL,
    name="StatisticalAnalyzer",
    system_prompt=(
        f"You are the Statistical Analyzer agent specialized in quantitative analysis.\n\n"
        f"Your task: Analyze cleaned data and extract insights.\n\n"
        f"Input file: {CLEAN_DATA_FILE}\n"
        f"Output file: {ANALYSIS_FILE}\n\n"
        "STRICT EXECUTION PLAN (follow exactly):\n"
        "1. Call calculate_statistics_tool({CLEAN_DATA_FILE}) - gives you basic metrics\n"
        "2. Call read_json_tool({CLEAN_DATA_FILE}) ONCE - loads all records\n"
        "3. Analyze the data you received mentally:\n"
        "   - Group sales by product to find top performers\n"
        "   - Group by region to compare regional performance\n"
        "   - Look at date ranges to identify trends\n"
        "   - Note any interesting patterns (e.g., best selling product, highest revenue region)\n"
        "4. Create insights object with:\n"
        "   - 'metrics': list from calculate_statistics_tool\n"
        "   - 'top_products': list of best selling products\n"
        "   - 'regional_summary': summary by region\n"
        "   - 'key_insights': 3-5 notable findings\n"
        "   - 'recommendations': 2-3 business recommendations\n"
        f"5. Call write_json_tool({ANALYSIS_FILE}, insights) EXACTLY ONCE\n"
        "6. IMMEDIATELY hand off to Visualizer\n\n"
        "CRITICAL RULES:\n"
        "- Call read_json_tool EXACTLY ONCE (you get all data)\n"
        "- DO NOT repeatedly call read_json_tool\n"
        "- Analyze the data mentally using the JSON you received\n"
        "- After write_json_tool succeeds, hand off immediately\n"
    ),
)

visualizer = Agent(
    MODEL,
    name="Visualizer",
    system_prompt=(
        f"You are the Visualizer agent specialized in data visualization.\n\n"
        f"Your task: Create visualization code for the analysis.\n\n"
        f"Input files: {CLEAN_DATA_FILE}, {ANALYSIS_FILE}\n"
        f"Output file: {VISUALIZATION_FILE}\n\n"
        "STRICT EXECUTION PLAN:\n"
        "1. Call read_json_tool({CLEAN_DATA_FILE}) ONCE to see the data structure\n"
        "2. Call read_json_tool({ANALYSIS_FILE}) ONCE to see the analysis\n"
        "3. Write a complete Python script that:\n"
        "   - Imports json, matplotlib.pyplot as plt\n"
        "   - Loads both JSON files\n"
        "   - Creates 3 charts: sales over time, top products bar chart, regional pie chart\n"
        "   - Saves each as PNG in {OUTPUT_DIR}\n"
        "   - Includes titles, labels, and legends\n"
        "4. Call write_text_tool({VISUALIZATION_FILE}, your_script) EXACTLY ONCE\n"
        "5. IMMEDIATELY hand off to ReportWriter\n\n"
        "CRITICAL RULES:\n"
        "- Read each file EXACTLY ONCE\n"
        "- DO NOT call run_python_script_tool (skip testing to save time)\n"
        "- Write complete, working code\n"
        "- After write_text_tool succeeds, hand off immediately\n"
    ),
)

report_writer = Agent(
    MODEL,
    name="ReportWriter",
    system_prompt=(
        f"You are the Report Writer agent specialized in technical documentation.\n\n"
        f"Your task: Compile the final analysis report in Markdown.\n\n"
        f"Input files:\n"
        f"- {DATA_FILE} (original data)\n"
        f"- {CLEAN_DATA_FILE} (cleaned data)\n"
        f"- {ANALYSIS_FILE} (analysis results)\n"
        f"Output file: {REPORT_FILE}\n\n"
        "STRICT EXECUTION PLAN:\n"
        "1. Call read_json_tool({ANALYSIS_FILE}) ONCE - get all insights\n"
        "2. Review the metrics, insights, and recommendations you received\n"
        "3. Write a comprehensive Markdown report with sections:\n"
        "   # Sales Data Analysis Report\n"
        "   ## Executive Summary (3-4 sentences)\n"
        "   ## Key Metrics (from analysis)\n"
        "   ## Top Insights (from analysis)\n"
        "   ## Regional Performance\n"
        "   ## Recommendations (from analysis)\n"
        "   ## Methodology\n"
        "4. Call write_text_tool({REPORT_FILE}, your_markdown) EXACTLY ONCE\n"
        "5. Return a completion summary\n\n"
        "CRITICAL RULES:\n"
        "- Read analysis file EXACTLY ONCE\n"
        "- DO NOT read other files multiple times\n"
        "- Write report based on the analysis data you received\n"
        "- Use professional Markdown formatting\n"
        "- This is the FINAL agent - do NOT hand off\n"
    ),
)


# =============================================================================
# Build the Swarm
# =============================================================================

async def main():
    """Run the data analysis pipeline workflow."""
    
    print("=" * 80)
    print("DATA ANALYSIS PIPELINE - Multi-Agent Data Science System")
    print("=" * 80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nPipeline stages:")
    print("  1. Data Collection - Generate sample sales dataset")
    print("  2. Data Cleaning - Validate and prepare data")
    print("  3. Statistical Analysis - Compute metrics and insights")
    print("  4. Visualization - Create charts and graphs")
    print("  5. Report Writing - Compile comprehensive report")
    print("\n" + "-" * 80 + "\n")
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Register tools with specific agents
    data_collector.tool(write_json_tool)
    
    data_cleaner.tool(read_json_tool)
    data_cleaner.tool(write_json_tool)
    data_cleaner.tool(analyze_dataset_tool)
    
    statistical_analyzer.tool(read_json_tool)
    statistical_analyzer.tool(write_json_tool)
    statistical_analyzer.tool(calculate_statistics_tool)
    
    visualizer.tool(read_json_tool)
    visualizer.tool(write_text_tool)
    visualizer.tool(run_python_script_tool)
    
    report_writer.tool(read_json_tool)
    report_writer.tool(read_text_tool)
    report_writer.tool(write_text_tool)
    
    # Configure agents in swarm
    swarm = Collab(
        agents=[
            CollabAgent(
                agent=project_manager,
                description="Coordinates the entire analysis pipeline",
                agent_handoffs=("DataCollector",),
            ),
            CollabAgent(
                agent=data_collector,
                description="Generates sample sales dataset",
                agent_handoffs=("DataCleaner",),
            ),
            CollabAgent(
                agent=data_cleaner,
                description="Validates and cleans the data",
                agent_handoffs=("StatisticalAnalyzer",),
            ),
            CollabAgent(
                agent=statistical_analyzer,
                description="Performs statistical analysis",
                agent_handoffs=("Visualizer",),
            ),
            CollabAgent(
                agent=visualizer,
                description="Creates data visualizations",
                agent_handoffs=("ReportWriter",),
            ),
            CollabAgent(
                agent=report_writer,
                description="Compiles final analysis report",
            ),
        ],
        final_agent="ReportWriter",
        max_handoffs=30,
        model=MODEL,
    )
    
    # Run the swarm
    query = (
        "Perform a complete end-to-end data analysis pipeline:\n"
        "1. Generate a realistic sales dataset with 50+ records\n"
        "2. Clean and validate the data\n"
        "3. Perform statistical analysis to find insights\n"
        "4. Create visualizations of the findings\n"
        "5. Compile a comprehensive analysis report\n\n"
        "Ensure each stage is completed before moving to the next."
    )
    
    print(f"Query: {query}\n")
    print("-" * 80 + "\n")
    
    with logfire.span("data_analysis_pipeline"):
        result = await swarm.run(query)
    
    # Print execution summary
    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    print(result.print_execution_flow())
    
    print("\n" + "=" * 80)
    print("FINAL OUTPUT")
    print("=" * 80)
    print(f"\n{result.output}\n")
    
    # Check generated files
    print("=" * 80)
    print("GENERATED FILES")
    print("=" * 80)
    
    files_to_check = [
        ("Raw Data", DATA_FILE),
        ("Cleaned Data", CLEAN_DATA_FILE),
        ("Analysis Results", ANALYSIS_FILE),
        ("Visualization Code", VISUALIZATION_FILE),
        ("Final Report", REPORT_FILE),
    ]
    
    for name, filepath in files_to_check:
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"✓ {name}: {filepath} ({size} bytes)")
        else:
            print(f"✗ {name}: {filepath} (not found)")
    
    # Display the final report if it exists
    if REPORT_FILE.exists():
        print("\n" + "=" * 80)
        print("FINAL ANALYSIS REPORT")
        print("=" * 80)
        print(REPORT_FILE.read_text())
        print("=" * 80)
    
    print(f"\n📊 Analysis complete!")
    print(f"📁 All files saved to: {OUTPUT_DIR}")
    print(f"📈 Total iterations: {result.iterations}")
    print(f"💰 Token usage: {result.usage}")
    print("=" * 80 + "\n")


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    asyncio.run(main())
