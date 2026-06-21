"""
📚 LEARN: Agent — The Brain Behind Agentic RAG (ReAct Pattern)
===============================================================
Standard RAG is a one-shot pipeline: retrieve chunks → generate answer.
But what if the first retrieval doesn't find good context? Standard RAG
just generates a poor answer. Agentic RAG fixes this.

📚 WHAT IS AGENTIC RAG?
Agentic RAG wraps the retrieval step in an intelligent loop. Instead of
blindly retrieving once and answering, an "agent" decides:
  - "Are these retrieved chunks actually relevant to the question?"
  - "Should I reformulate my search query and try again?"
  - "Do I need more context, or do I have enough to answer?"

📚 THE ReAct PATTERN (Yao et al., 2022):
ReAct stands for "Reason + Act". The agent alternates between:

  Thought  →  "I need to find information about X"     (reasoning)
  Action   →  RequestMoreContext("X")                   (doing something)
  Observation → [retrieved chunks about X]              (seeing results)
  Thought  →  "These chunks cover X but not Y, let me   (reasoning again)
                reformulate to also find Y"
  Action   →  ReformulateQuery("X and Y relationship")  (trying differently)
  Observation → [better chunks about X and Y]           (new results)
  Thought  →  "Now I have enough context to answer"     (deciding to stop)
  Action   →  Answer("Based on the documents...")       (final answer)

The key insight: the LLM ITSELF judges whether retrieved context is
good enough. No hardcoded thresholds — the model reads the chunks and
reasons about whether they answer the question.

📚 WHY NOT JUST RETRIEVE MORE CHUNKS?
You might think: "Why not just set top_k=50 and retrieve everything?"
Because:
  1. More chunks = more noise. Irrelevant chunks confuse the LLM.
  2. Different phrasings find different things. "Return policy" and
     "how to send back an item" find different chunks.
  3. Context window limits. LLMs have a max input size.
  4. Targeted retrieval beats shotgun retrieval for answer quality.

📚 ACTIONS AVAILABLE:
  - ReformulateQuery(new_query): Rephrase the search for better results
  - RequestMoreContext(query, top_k, search_mode): Retrieve with new params
  - Answer(response): Generate the final answer and stop the loop
"""

import re
import json
import requests
from dataclasses import dataclass, field
from rag.generator import _get_api_key, CHAT_API_URL

# ── Action Data Classes ───────────────────────────────────────

# 📚: We represent each action as a simple dataclass. This makes
# it easy to inspect, log, and test agent decisions without parsing raw strings everywhere.

@dataclass
class ReformulateQuery:
    """Agent wants to rephrase the query for better retrieval."""
    new_query: str

@dataclass
class RequestMoreContext:
    """Agent wants to retrieve more chunks with specific parameters."""
    query: str
    top_k: int = 5
    search_mode: str = "vector"

# 📚: A "step" records one Thought-Action-Observation cycle.
# We keep a full history so the agent can see what it already tried
# (and avoid repeating the same failed query).
@dataclass
class AgentStep:
    """One iteration of the ReAct loop."""
    thought: str
    action: object  # ReformulateQuery | RequestMoreContext | Answer
    observation: str = ""

# ── System Prompt ─────────────────────────────────────────────

# 📚: This is the most critical piece of the agent. The system
# prompt teaches the LLM HOW to be an agent. It needs to:
#   1. Explain the Thought/Action/Observation format
#   2. Describe each available action with clear usage rules
#   3. Show examples of good reasoning (few-shot prompting)
#   4. Tell the agent when to stop (Answer) vs. keep going

AGENT_SYSTEM_PROMPT = """You are a ReAct agent for a Retrieval-Augmented Generation (RAG) system.
Your job is to find the best information to answer the user's question by reasoning and taking actions.

## How It Works
You operate in a loop of: Thought → Action → Observation
- **Thought**: Reason about what you know and what you still need
- **Action**: Choose ONE action to take
- **Observation**: You'll see the result of your action (provided by the system)

## Available Actions

1. **RequestMoreContext(query, top_k, search_mode)**
   Retrieve document chunks from the knowledge base.
   - query: The search query (string)
   - top_k: Number of chunks to retrieve (integer, default 5)
   - search_mode: "vector", "keyword", or "hybrid" (default "vector")
   Example: Action: RequestMoreContext("machine learning overfitting", 5, "vector")

2. **ReformulateQuery(new_query)**
   Rephrase your search query to find different or better results.
   Use this when previous retrieval didn't return relevant chunks.
   Example: Action: ReformulateQuery("how to prevent model overfitting techniques")

3. **Answer(response)**
   Provide the final answer based on the context you've gathered.
   Use this when you have enough information to answer the question well.
   Your response should be grounded in the retrieved context — do NOT make things up.
   Example: Action: Answer("Based on the documents, overfitting occurs when...")

## Rules
- Always start with a Thought explaining your reasoning
- Then choose exactly ONE Action
- If retrieved chunks are relevant and sufficient, choose Answer immediately — don't over-retrieve
- If retrieved chunks are poor or missing key information, try ReformulateQuery or RequestMoreContext with different parameters
- Do NOT repeat the same query you already tried
- Keep thoughts concise (1-3 sentences)
- When answering, cite which source documents you used

## Output Format
You MUST respond in exactly this format:

Thought: <your reasoning>
Action: <ActionName>(<parameters>)
"""
# ── Action Parsing ────────────────────────────────────────────

# 📚: The LLM outputs free-form text in the Thought/Action format.
# We need to parse this text to extract the structured action. This is
# the trickiest part of building an agent — LLMs don't always follow
# the format perfectly, so we need robust parsing with fallbacks.
def parse_agent_response(text: str) -> tuple[str, object]:
    """
    Parse the agent's response into a (thought, action) tuple.

    📚: We use regex to extract the Thought and Action from the
    LLM's output. The parsing is intentionally forgiving — we handle
    minor formatting variations (extra spaces, different quoting, etc.)
    because LLMs are not deterministic formatters.

    Args:
        text: Raw LLM output in Thought/Action format

    Returns:
        Tuple of (thought_text, action_object)

    Raises:
        ValueError: If the response can't be parsed into a valid action
    """
    # Extract the thought
    thought = ""
    thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|\Z)", text, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()

    # Extract the action line
    action_match = re.search(r"Action:\s*(.+)", text)
    if not action_match:
        raise ValueError(f"No Action found in agent response: {text[:200]}")

    action_str = action_match.group(1).strip()

    # Parse specific actions
    action = _parse_action_string(action_str)
    return thought, action

def _parse_action_string(action_str: str) -> object:
    """
    Parse an action string like 'RequestMoreContext("query", 5, "vector")' into
    the corresponding dataclass.

    📚 LEARN: We parse each action type separately because they have different parameters. 
    We're intentionally flexible with quoting (single, double, or none) because different LLMs format strings differently.
    """
    # Try Answer(...)
    answer_match = re.match(
        r'Answer\(\s*["\']?(.*?)["\']?\s*\)$',
        action_str,
        re.DOTALL,
    )
    if answer_match:
        return Answer(response=answer_match.group(1).strip())
    
    # 📚: For Answer, also handle multi-line responses where the LLM
    # writes a long answer inside the parentheses.
    if action_str.startswith("Answer("):
        content = action_str[7:]  # Remove "Answer("
        if content.endswith(")"):
            content = content[:-1]
        # Strip outer quotes if present
        content = content.strip().strip("'\"")
        return Answer(response=content)
    
    # try reformulateQuery(...)
    reformulate_match = re.match(
        r'ReformulateQuery\(\s*["\'](.+?)["\']\s*\)',
        action_str,
        re.DOTALL,
    )
    if reformulate_match:
        return ReformulateQuery(new_query=reformulate_match.group(1).strip())
    
    # Try RequestMoreContext(...)
    # Format: RequestMoreContext("query", top_k, "search_mode")
    rmc_match = re.match(
        r'RequestMoreContext\(\s*["\'](.+?)["\']\s*'    # query (required)
        r'(?:,\s*(\d+)\s*)?'                            # top_k (optional)
        r'(?:,\s*["\'](\w+)["\']\s*)?\)',                # search_mode (optional)
        action_str,
        re.DOTALL,
    )
    if rmc_match:
        query = rmc_match.group(1).strip()
        top_k = int(rmc_match.group(2)) if rmc_match.group(2) else 5
        search_mode = rmc_match.group(3) if rmc_match.group(3) else "vector"
        return RequestMoreContext(query=query, top_k=top_k, search_mode=search_mode)

    raise ValueError(f"Unknown action format: {action_str[:200]}")

# ── LLM Call ──────────────────────────────────────────────────

def _call_agent_llm(messages: list[dict]) -> str:
    """
    Call the LLM with the agent's conversation history.

    📚: Unlike the generator which streams tokens, the agent uses
    non-streaming calls. We need the FULL response to parse the Thought
    and Action before proceeding. Streaming would mean we'd have to
    wait for the full response anyway.

    Separated out so it can be easily mocked in tests.
    """
    api_key = _get_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "messages": messages,
        "stream": False,
    }

    response = requests.post(CHAT_API_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()

    result = response.json()
    return result["choices"][0]["message"]["content"]

# ── ReAct Agent ───────────────────────────────────────────────
class ReActAgent:
    """
    A ReAct agent that iteratively reasons about and retrieves context.

    📚: This is the "brain" of Agentic RAG. It maintains state
    across iterations:
      - steps: Full history of Thought/Action/Observation cycles
      - accumulated_chunks: All unique chunks retrieved so far
      - tried_queries: Queries already attempted (to avoid repetition)

    The agent builds a conversation with the LLM where each iteration
    adds the previous Thought + Action + Observation to the history.
    This way, the LLM remembers what it already tried and can make
    increasingly informed decisions.
    """
    def __init__(self, max_iterations: int = 5):
        """
        Args:
            max_iterations: Maximum Thought-Action-Observation cycles
                            before forcing an answer (default: 5)
        """
        self.max_iterations = max_iterations
        self.steps: list[AgentStep] = []
        self.accumulated_chunks: list[dict] = []
        self.tried_queries: set[str] = set()

    def build_messages(self, question:str) -> list[dict]:
        """
        Build the LLM message history including all previous steps.

        📚: The message history grows with each iteration:
          [system prompt, user question, assistant step 1, user observation 1,
           assistant step 2, user observation 2, ...]

        This gives the LLM full context about what it already tried,
        so it can make better decisions in the next iteration.
        """
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}"},
        ]

        # Add previous steps as conversation history
        for step in self.steps:
            # The agent's thought + action
            messages.append({
                "role": "assistant",
                "content": f"Thought: {step.thought}\nAction: {self._format_action(step.action)}",
            })
            # The observation (system feedback)
            messages.append({
                "role": "user",
                "content": f"Observation: {step.observation}",
            })

        return messages
            
    def add_chunks(self, new_chunks: list[dict]):
        """
        Add newly retrieved chunks to the accumulated context.

        📚: Deduplication is important! If the agent retrieves
        with different queries, some chunks may appear in both results.
        We deduplicate by chunk text to avoid feeding the same context
        to the generator twice (which wastes context window space).
        """
        existing_texts = {c["text"] for c in self.accumulated_chunks}
        for chunk in new_chunks:
            if chunk["text"] not in existing_texts:
                self.accumulated_chunks.append(chunk)
                existing_texts.add(chunk["text"])
    
    @staticmethod
    def _format_action(action) -> str:
        """Format an action dataclass back into the string representation."""
        if isinstance(action, Answer):
            return f'Answer("{action.response}")'
        elif isinstance(action, ReformulateQuery):
            return f'ReformulateQuery("{action.new_query}")'
        elif isinstance(action, RequestMoreContext):
            return (f'RequestMoreContext("{action.query}", '
                    f'{action.top_k}, "{action.search_mode}")')
        return str(action)
    
    def get_next_action(self, question: str) -> tuple[str, object]:
        """
        Ask the LLM for the next Thought + Action.

        📚 LEARN: This is where the "magic" happens. We send the full
        conversation history to the LLM and ask "what should we do next?"
        The LLM reasons about the question, what it's already found,
        and decides the next step.

        Args:
            question: The original user question

        Returns:
            Tuple of (thought, action)

        Raises:
            ValueError: If the LLM's response can't be parsed
        """
        messages = self.build_messages(question)
        response = _call_agent_llm(messages)
        return parse_agent_response(response)