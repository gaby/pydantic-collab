"""Test script to demonstrate the improved prompt builder output."""

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_collab import CollabAgent
from pydantic_collab._types import PromptBuilderContext
from pydantic_collab._utils import default_build_agent_prompt

# Create test agents
orchestrator = Agent(TestModel(), name='Orchestrator')
researcher = Agent(TestModel(), name='Researcher')
code_writer = Agent(TestModel(), name='CodeWriter')
reviewer = Agent(TestModel(), name='Reviewer')

# Test Case 1: Agent with both handoffs AND tool calls
print('=' * 80)
print('TEST CASE 1: Agent with BOTH handoffs and tool calls')
print('=' * 80)

ctx1 = PromptBuilderContext(
    agent=CollabAgent(
        agent=orchestrator, 
        description='Coordinates the workflow and delegates tasks',
        agent_calls=[],
        agent_handoffs=[]
    ),
    final_agent=True,
    can_handoff=True,
    called_as_tool=False,
    handoff_agents=[
        CollabAgent(
            agent=code_writer,
            description='Writes Python code based on requirements',
            agent_calls=[],
            agent_handoffs=[]
        ),
        CollabAgent(
            agent=reviewer,
            description='Reviews and tests code',
            agent_calls=[],
            agent_handoffs=[]
        )
    ],
    tool_agents=[
        CollabAgent(
            agent=researcher,
            description='Researches information using web search',
            agent_calls=[],
            agent_handoffs=[]
        ),
        CollabAgent(
            agent=code_writer,
            description='Writes Python code based on requirements',
            agent_calls=[],
            agent_handoffs=[]
        )
    ]
)

prompt1 = default_build_agent_prompt(ctx1)
print(prompt1)

# Test Case 2: Agent with only handoffs
print('\n' + '=' * 80)
print('TEST CASE 2: Agent with ONLY handoffs (no tool calls)')
print('=' * 80)

ctx2 = PromptBuilderContext(
    agent=CollabAgent(
        agent=researcher,
        description='Researches information using web search',
        agent_calls=[],
        agent_handoffs=[]
    ),
    final_agent=False,
    can_handoff=True,
    handoff_agents=[
        CollabAgent(
            agent=code_writer,
            description='Writes Python code based on requirements',
            agent_calls=[],
            agent_handoffs=[]
        )
    ],
    tool_agents=[],
    called_as_tool=False,
)

prompt2 = default_build_agent_prompt(ctx2)
print(prompt2)

# Test Case 3: Agent with only tool calls
print('\n' + '=' * 80)
print('TEST CASE 3: Agent with ONLY tool calls (no handoffs)')
print('=' * 80)

ctx3 = PromptBuilderContext(
    agent=CollabAgent(
        agent=orchestrator,
        description='Coordinates the workflow and delegates tasks',
        agent_calls=[],
        agent_handoffs=[]
    ),
    final_agent=True,
    can_handoff=False,
    handoff_agents=[],
    called_as_tool=False,
    tool_agents=[
        CollabAgent(
            agent=researcher,
            description='Researches information using web search',
            agent_calls=[],
            agent_handoffs=[]
        ),
        CollabAgent(
            agent=code_writer,
            description='Writes Python code based on requirements',
            agent_calls=[],
            agent_handoffs=[]
        )
    ]
)

prompt3 = default_build_agent_prompt(ctx3)
print(prompt3)

# Test Case 4: Agent with neither (final agent only)
print('\n' + '=' * 80)
print('TEST CASE 4: Agent with NO collaboration (final agent only)')
print('=' * 80)

ctx4 = PromptBuilderContext(
    agent=CollabAgent(
        agent=reviewer,
        description='Reviews and tests code',
        agent_calls=[],
        agent_handoffs=[]
    ),
    final_agent=True,
    can_handoff=False,
    handoff_agents=[],
    tool_agents=[],
    called_as_tool=False,
)

prompt4 = default_build_agent_prompt(ctx4)
print(prompt4)
