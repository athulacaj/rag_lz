User Query
    ↓
┌─────────────────────────────────────────────────┐
│ STAGE 1: NER Agent                              │
│ - Extract entities (people, skills, attributes) │
│ - Classify intent (get, compare, list)          │
│ - Determine query type (simple, chained, etc)   │
└─────────────────────────────────────────────────┘
    ↓ (NER Output)
┌─────────────────────────────────────────────────┐
│ STAGE 2: Planner Agent                          │
│ - Create execution plan using NER entities      │
│ - Define tool calls and dependencies            │
│ - Specify answer format                         │
└─────────────────────────────────────────────────┘
    ↓ (Execution Plan)
┌─────────────────────────────────────────────────┐
│ STAGE 3: Execution Engine                       │
│ - Resolve $ner and $state references            │
│ - Execute tools in order                        │
│ - Build execution state                         │
└─────────────────────────────────────────────────┘
    ↓ (Execution State)
┌─────────────────────────────────────────────────┐
│ STAGE 4: Answer Synthesizer                     │
│ - Format results based on query type            │
│ - Generate natural language answer              │
└─────────────────────────────────────────────────┘
    ↓
Final Answer


🔑 Key Benefits
1. Separation of Concerns

NER: Understands what the user wants
Planner: Decides how to get it
Executor: Actually gets it
Synthesizer: Formats the answer

2. Entity Binding
The planner can reference NER entities directly:
python"input": {"candidate_name": "$ner.entities[0].normalized_name"}
3. Query Type Classification
NER identifies 4 query types:

simple_retrieval: Direct lookup
multi_entity: Parallel queries
chained_query: Sequential dependencies
comparison: Gather then compare

4. Better Debugging
Each stage has clear inputs/outputs, making it easy to debug where things go wrong.