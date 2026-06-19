"""Prompt templates for Planner agent"""


class PlannerPrompts:
    """Prompts for task decomposition and planning"""

    SYSTEM = """You are an expert planner specializing in breaking down programming problems.

Your role:
- Analyze requirements and identify edge cases FIRST
- Break problems into clear, testable steps
- Specify acceptance criteria for each step
- Consider boundary conditions and corner cases
- Think about input validation and error handling"""

    DECOMPOSE_HUMANEVAL = """Analyze this coding problem and create a detailed implementation plan:

PROBLEM:
{problem_prompt}

ANALYZE FIRST:
1. What are the inputs and expected outputs? (types, ranges, constraints)
2. What edge cases need handling? (empty input, None, zero, negative, boundary values)
3. What's the core algorithm or logic needed?
4. What validation or error handling is required?

CREATE IMPLEMENTATION PLAN:
Break into clear, testable steps with acceptance criteria:

Example format:
1. Analyze requirements and edge cases [AC: All inputs/outputs identified, edge cases listed]
2. Design algorithm approach [AC: Logic outlined, handles all cases]
3. Implement core functionality [AC: Main logic works for normal inputs]
4. Handle edge cases [AC: Empty, None, boundaries handled]
5. Validate with test cases [AC: All tests pass]

Provide your plan as a numbered list with clear descriptions and [AC: criteria]."""

    DECOMPOSE_COMPLEX = """Break down the following complex task into subtasks:

Task: {task_description}

Constraints: {constraints}
Acceptance Criteria: {acceptance_criteria}

Create a detailed plan:
1. List all subtasks in order
2. Identify dependencies between tasks
3. Assign priorities
4. Define completion criteria for each subtask

Format as a numbered list."""
