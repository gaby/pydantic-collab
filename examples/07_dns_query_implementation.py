"""DNS Query Implementation Example - Multi-Agent Code Generation System

This example demonstrates a sophisticated multi-agent swarm that:
1. Orchestrator: Coordinates the workflow and makes decisions
2. Researcher: Uses web search and fetch tools to find information
3. Code Writer: Implements the DNS query code
4. Reviewer: Reviews, modifies, and tests the code

The goal: Implement a Python script that performs DNS queries to 8.8.8.8
"""

import asyncio
from pathlib import Path

import logfire
from example_tools import (
    modify_file_tool,
    read_file_tool,
    run_python_code,
    write_file_tool,
)
from pydantic_ai import Agent, WebFetchTool, WebSearchTool

from pydantic_collab import Collab, CollabAgent, PipelineCollab

logfire.configure()
logfire.instrument_httpx(capture_all=True)
logfire.instrument_pydantic_ai()

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_FILE = Path('/tmp/dns_query_implementation.py')
MODEL = 'google-gla:gemini-2.5-flash'  # Change to your preferred model


# =============================================================================
# Agent Definitions
# =============================================================================

# Orchestrator: Coordinates the entire workflow
orchestrator = Agent(
    MODEL,
    name='Orchestrator',
    system_prompt=(
        'You are the Orchestrator agent. Your role is to coordinate the DNS query implementation project.\n\n'
        'Project Goal: Create a Python script that performs DNS queries to 8.8.8.8\n\n'
        'Your workflow:\n'
        '1. First, hand off to the Researcher to gather information about DNS queries in Python\n'
        '2. After research, hand off to the CodeWriter to implement the solution\n'
        '3. Finally, hand off to the Reviewer to verify and test the code\n\n'
        'Keep track of the progress and make decisions about what needs to be done next.\n'
        'Be concise and action-oriented in your handoffs.'
    ),
)

# Researcher: Uses web search to find information
researcher = Agent(
    MODEL,
    name='Researcher',
    builtin_tools=[WebSearchTool(), WebFetchTool()],
    system_prompt=(
        'You are the Researcher agent specialized in finding technical information.\n\n'
        'Your task: Research how to implement DNS queries in Python.\n\n'
        'Use your web search and fetch tools to:\n'
        '1. Find information about Python DNS libraries (socket, dnspython, etc.)\n'
        '2. Look for examples of querying specific DNS servers (like 8.8.8.8)\n'
        '3. Identify best practices and common approaches\n\n'
        'Summarize your findings clearly, including:\n'
        '- Recommended libraries or approaches\n'
        '- Example code patterns\n'
        '- Any important considerations\n\n'
        'Keep your research focused and relevant to the task.'
    ),
)

# Code Writer: Implements the DNS query code
code_writer = Agent(
    MODEL,
    name='CodeWriter',
    system_prompt=(
        f'You are the CodeWriter agent specialized in implementing Python code.\n\n'
        f'Your task: Write a Python script that performs DNS queries to 8.8.8.8\n\n'
        f'Target file: {OUTPUT_FILE}\n\n'
        'Requirements:\n'
        '1. Use standard library (socket) or popular libraries (dnspython if needed)\n'
        '2. Query 8.8.8.8 DNS server specifically\n'
        '3. Accept a domain name as command-line argument\n'
        '4. Return the IP addresses for the domain\n'
        '5. Include proper error handling\n'
        '6. Add helpful comments and docstrings\n\n'
        'Use the write_file tool to save your implementation.\n'
        'Write clean, production-quality code.'
    ),
)

# Reviewer: Reviews, modifies, and tests the code
reviewer = Agent(
    MODEL,
    name='Reviewer',
    system_prompt=(
        f'You are the Reviewer agent specialized in code review and testing.\n\n'
        f'Your task: Review, improve, and test the DNS query implementation.\n\n'
        f'Target file: {OUTPUT_FILE}\n\n'
        'Workflow:\n'
        '1. Read the implementation using read_file\n'
        '2. Review the code for:\n'
        '   - Correctness and proper DNS query implementation\n'
        '   - Error handling and edge cases\n'
        '   - Code quality and documentation\n'
        '   - Security considerations\n'
        "3. Find at least one issue and move it back to CodeWriter to fix them (don't do it more than 2 times)\n"
        '4. Test the code by running it with: run_python_code\n'
        "   - Test with a simple domain like 'google.com'\n"
        '5. Report your findings and test results\n\n'
        'Be thorough but practical in your review.'
    ),
)


# =============================================================================
# Build the Swarm
# =============================================================================


async def main():
    """Run the DNS query implementation workflow."""
    print('=' * 80)
    print('DNS Query Implementation - Multi-Agent Code Generation')
    print('=' * 80)
    print(f'\nTarget file: {OUTPUT_FILE}')
    print('DNS server: 8.8.8.8')
    print('\n' + '-' * 80 + '\n')

    # Configure agents in swarm
    swarm = Collab(
        agents=[
            CollabAgent(
                agent=orchestrator,
                description='Coordinates the workflow and makes decisions',
                agent_handoffs='Researcher',
                agent_calls='Researcher',
            ),
            CollabAgent(
                agent=researcher,
                description='Researches DNS query implementation in Python',
                agent_handoffs='CodeWriter',
            ),
            CollabAgent(
                agent=code_writer,
                description='Writes the DNS query implementation code',
                agent_handoffs='Reviewer',
            ),
            CollabAgent(
                agent=reviewer,
                description='Reviews, tests, and improves the code',
                agent_calls='CodeWriter',
            ),
        ],
        final_agent='Reviewer',
        max_handoffs=15,
        model=MODEL,
    )

    # Register tools with specific agents
    code_writer.tool(write_file_tool)

    reviewer.tool(read_file_tool)
    reviewer.tool(modify_file_tool)
    reviewer.tool(run_python_code)

    # Run the swarm
    query = (
        'Create a Python script that performs DNS queries to the 8.8.8.8 DNS server. '
        'The script should accept a domain name and return its IP addresses. '
        'Use the research -> implement -> review workflow.'
    )

    print(f'Query: {query}\n')
    print('-' * 80 + '\n')
    with logfire.span('test!!'):
        result = await swarm.run(query)

    # Print execution summary
    print('\n' + '=' * 80)
    print('EXECUTION SUMMARY')
    print('=' * 80)
    print(result.print_execution_flow())
    print('\n' + '=' * 80)
    print('FINAL OUTPUT')
    print('=' * 80)
    print(f'\n{result.output}\n')

    # Check if file was created
    if OUTPUT_FILE.exists():
        print('=' * 80)
        print('GENERATED CODE')
        print('=' * 80)
        print(f'\nFile: {OUTPUT_FILE}')
        print(f'Size: {OUTPUT_FILE.stat().st_size} bytes\n')
        print('-' * 80)
        print(OUTPUT_FILE.read_text())
        print('-' * 80)

        print('\n💡 You can test the script with:')
        print(f'   python {OUTPUT_FILE} google.com')
    else:
        print(f'\n⚠️  Warning: Expected output file not found: {OUTPUT_FILE}')

    print('\n' + '=' * 80)
    print(f'Total iterations: {result.iterations}')
    print(f'Token usage: {result.usage}')
    print('=' * 80 + '\n')


# =============================================================================
# Alternative: Simpler Forward Chain Approach
# =============================================================================


async def simple_chain_example():
    """Simpler alternative using forward_handoff topology.

    This version uses a linear chain: Researcher -> CodeWriter -> Reviewer
    """
    print('=' * 80)
    print('DNS Query Implementation - Simple Chain Approach')
    print('=' * 80 + '\n')

    # Create simplified agents for chain
    researcher_simple = Agent(
        MODEL,
        name='Researcher',
        builtin_tools=[WebSearchTool(), WebFetchTool()],
        system_prompt=(
            'Research how to implement DNS queries in Python. '
            'Focus on using 8.8.8.8 as the DNS server. '
            'Provide a brief summary with code examples.'
        ),
    )

    code_writer_simple = Agent(
        MODEL,
        name='CodeWriter',
        system_prompt=(
            f'Based on the research, write a complete Python script to {OUTPUT_FILE} '
            'that queries 8.8.8.8 DNS server for a domain name. '
            'Use write_file tool to save the code.'
        ),
    )

    reviewer_simple = Agent(
        MODEL,
        name='Reviewer',
        system_prompt=(f"Read {OUTPUT_FILE}, test it with 'google.com', and report results. Fix any issues you find."),
    )

    # Set up tools
    code_writer_simple.tool(write_file_tool)
    reviewer_simple.tool(read_file_tool)
    reviewer_simple.tool(modify_file_tool)
    reviewer_simple.tool(run_python_code)

    # Create linear chain swarm
    swarm = PipelineCollab(
        agents=[
            CollabAgent(agent=researcher_simple, description='Research DNS implementation'),
            CollabAgent(agent=code_writer_simple, description='Write the code'),
            CollabAgent(agent=reviewer_simple, description='Review and test'),
        ],
        max_handoffs=5,
        model=MODEL,
    )
    with logfire.span('My Swarm'):
        result = await swarm.run('Create and test a Python DNS query script for 8.8.8.8')

    print(result.print_execution_flow())
    print(f'\nFinal output:\n{result.output}')


# =============================================================================
# Run
# =============================================================================

if __name__ == '__main__':
    # Choose which example to run
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--simple':
        asyncio.run(simple_chain_example())
    else:
        asyncio.run(main())
