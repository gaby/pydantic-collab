"""Test fixtures for realistic handoff and data passing scenarios.

These fixtures are based on actual pydantic_ai message structures captured from real model runs.
They allow testing handoff control and data passing without requiring live model calls.
"""
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

# Fixture: Simple text response leading to handoff
SIMPLE_TEXT_RESPONSE = ModelResponse(
    parts=[TextPart(content='Data processed successfully. Ready for analysis.')],
    timestamp='2024-01-01T00:00:00Z',
)

# Fixture: Response with tool call before handoff
TOOL_CALL_RESPONSE = ModelResponse(
    parts=[
        ToolCallPart(
            tool_name='process_data',
            args={'data': [10, 20, 30]},
            tool_call_id='test_tool_001',
        )
    ],
    timestamp='2024-01-01T00:00:01Z',
)

# Fixture: Tool return that could lead to handoff
TOOL_RETURN_REQUEST = ModelRequest(
    parts=[
        ToolReturnPart(
            tool_name='process_data',
            content='{"sum": 60, "avg": 20}',
            tool_call_id='test_tool_001',
        )
    ]
)

# Fixture: Complex handoff with context
HANDOFF_WITH_CONTEXT_MESSAGES = [
    ModelRequest(
        parts=[
            SystemPromptPart(content='You are a data processor.'),
            UserPromptPart(content='Process these numbers: 5, 10, 15'),
        ]
    ),
    ModelResponse(
        parts=[
            ToolCallPart(
                tool_name='calculate_stats',
                args={'numbers': [5, 10, 15]},
                tool_call_id='stats_001',
            )
        ],
        timestamp='2024-01-01T00:00:02Z',
    ),
    ModelRequest(
        parts=[
            ToolReturnPart(
                tool_name='calculate_stats',
                content='{"mean": 10, "median": 10, "std": 4.08}',
                tool_call_id='stats_001',
            )
        ]
    ),
    ModelResponse(
        parts=[
            TextPart(content='Statistics calculated. Mean is 10, standard deviation is 4.08.')
        ],
        timestamp='2024-01-01T00:00:03Z',
    ),
]

# Fixture: Multi-step conversation before handoff
MULTI_STEP_CONVERSATION = [
    ModelRequest(
        parts=[
            SystemPromptPart(content='You research topics thoroughly.'),
            UserPromptPart(content='Tell me about machine learning'),
        ]
    ),
    ModelResponse(
        parts=[
            TextPart(content='Machine learning is a subset of AI that enables systems to learn from data.')
        ],
        timestamp='2024-01-01T00:00:04Z',
    ),
    ModelRequest(
        parts=[
            UserPromptPart(content='Can you be more specific about neural networks?')
        ]
    ),
    ModelResponse(
        parts=[
            TextPart(
                content='Neural networks are computational models inspired by biological neurons. '
                       'They consist of layers of interconnected nodes that process information.'
            )
        ],
        timestamp='2024-01-01T00:00:05Z',
    ),
]

# Fixture: Handoff data payloads
HANDOFF_DATA_BASIC = {
    'next_agent': 'Analyzer',
    'reasoning': 'Processing complete, ready for analysis',
    'query': 'Analyze the processed data',
}

HANDOFF_DATA_WITH_CONTEXT = {
    'next_agent': 'Writer',
    'reasoning': 'Research findings ready for article',
    'query': 'Write an article about the research findings',
    'include_conversation': True,
}

HANDOFF_DATA_WITHOUT_CONTEXT = {
    'next_agent': 'Finalizer',
    'reasoning': 'Data ready for final processing',
    'query': 'Finalize the report',
    'include_conversation': False,
}

# Fixture: Chain of handoffs simulation
HANDOFF_CHAIN = [
    {
        'agent': 'Intake',
        'input': 'Process customer request',
        'output': 'Request categorized as technical',
        'handoff_to': 'TechSupport',
        'reasoning': 'Technical issue detected',
    },
    {
        'agent': 'TechSupport',
        'input': 'Request categorized as technical',
        'output': 'Issue requires escalation',
        'handoff_to': 'Engineer',
        'reasoning': 'Complex technical problem',
    },
    {
        'agent': 'Engineer',
        'input': 'Issue requires escalation',
        'output': 'Solution implemented and verified',
        'handoff_to': None,  # Final agent
        'reasoning': 'Problem resolved',
    },
]

# Fixture: Parallel processing simulation (for mesh topology)
PARALLEL_AGENTS_DATA = {
    'strategist': {
        'input': 'Design a new feature',
        'tools_used': ['analyze_market', 'check_competitors'],
        'output': 'Strategic recommendations prepared',
    },
    'technologist': {
        'input': 'Design a new feature',
        'tools_used': ['check_tech_stack', 'estimate_effort'],
        'output': 'Technical feasibility assessed',
    },
    'designer': {
        'input': 'Design a new feature',
        'tools_used': ['create_mockup', 'user_research'],
        'output': 'Design mockups created',
    },
}

# Fixture: Data passing between agents with transformation
DATA_TRANSFORMATION_CHAIN = [
    {
        'stage': 'collector',
        'input_data': {'query': 'sales data'},
        'output_data': {'raw_data': [100, 200, 150, 300], 'source': 'database'},
        'transformation': 'collection',
    },
    {
        'stage': 'processor',
        'input_data': {'raw_data': [100, 200, 150, 300], 'source': 'database'},
        'output_data': {'processed': [100, 200, 150, 300], 'stats': {'sum': 750, 'count': 4}},
        'transformation': 'aggregation',
    },
    {
        'stage': 'analyzer',
        'input_data': {'processed': [100, 200, 150, 300], 'stats': {'sum': 750, 'count': 4}},
        'output_data': {'insights': 'Average sale: 187.5', 'trend': 'increasing'},
        'transformation': 'analysis',
    },
]

# Fixture: Error handling in handoff chain
ERROR_RECOVERY_SCENARIO = {
    'initial_request': 'Process invalid data: null',
    'first_agent_attempt': {
        'agent': 'Processor',
        'error': 'Cannot process null data',
        'recovery_action': 'handoff_to_error_handler',
    },
    'error_handler': {
        'agent': 'ErrorHandler',
        'input': 'Cannot process null data',
        'output': 'Data validation failed, requesting user clarification',
        'final': True,
    },
}

# Fixture: Context size control
LARGE_CONTEXT_SCENARIO = {
    'messages_count': 50,
    'include_all': False,  # Test that context can be limited
    'last_n_messages': 10,  # Only include last 10 messages in handoff
    'reasoning': 'Long conversation history should be summarized',
}

# Fixture: Tool results preservation across handoffs
TOOL_RESULTS_HANDOFF = {
    'agent1_tools': [
        {'tool': 'fetch_data', 'result': {'data': [1, 2, 3]}},
        {'tool': 'validate_data', 'result': {'valid': True}},
    ],
    'handoff_includes_tool_results': True,
    'agent2_receives': {
        'query': 'Analyze validated data',
        'context': 'Previous agent fetched and validated data',
        'tool_history': True,
    },
}

# Fixture: Conditional handoff based on result
CONDITIONAL_HANDOFF_SCENARIO = {
    'agent': 'Classifier',
    'input': 'Classify this request',
    'result': 'complex',
    'handoff_rules': {
        'simple': 'QuickProcessor',
        'moderate': 'StandardProcessor',
        'complex': 'AdvancedProcessor',
    },
    'expected_handoff': 'AdvancedProcessor',
}
