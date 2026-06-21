"""
📚 LEARN: Agentic Pipeline — The ReAct Loop in Action
======================================================
This module wraps the standard RAG pipeline with an intelligent agent
that can iteratively improve retrieval before generating an answer.

📚 STANDARD vs AGENTIC RAG:

  Standard RAG (pipeline.py):
    Question → Retrieve once → Generate answer → Done
    Fast (1 retrieval + 1 generation), but fragile if retrieval is poor.

  Agentic RAG (this file):
    Question → Agent thinks → Retrieve → Agent evaluates → Maybe retrieve again
            → ... → Agent decides "I have enough" → Generate answer → Done
    Slower (multiple LLM calls), but self-correcting and more reliable.

📚 WHEN TO USE AGENTIC RAG:
  ✅ Complex questions that need information from multiple angles
  ✅ Ambiguous queries where the right search terms aren't obvious
  ✅ When retrieval quality varies and you need self-correction
  ❌ Simple factual lookups (overkill — standard RAG is faster)
  ❌ Latency-sensitive applications (each iteration adds ~1-2 seconds)

📚 HOW THE LOOP WORKS:
  1. Agent receives the question
  2. Agent decides: "What should I search for?"
  3. System retrieves chunks and shows them to the agent
  4. Agent evaluates: "Is this context good enough to answer?"
     - YES → Agent chooses Answer action, loop ends
     - NO  → Agent reformulates query or requests more context, loop continues
  5. If max iterations reached, agent is forced to answer with whatever context it has
"""
from rag.pipeline import RAGPipeline
from rag.retriever import Retriever
from rag.generator import generate_with_sources
from rag.agent import (
    ReActAgent,
    AgentStep,
    ReformulateQuery,
    RequestMoreContext,
    Answer,
    parse_agent_response,
)

class AgenticRAGPipeline:
    """
    RAG pipeline enhanced with a ReAct agent for iterative retrieval.

    📚 LEARN: This class orchestrates the conversation between:
      - The ReAct Agent (decides what to search for)
      - The Retriever (executes the actual search)
      - The Generator (produces the final answer)

    The agent doesn't call the retriever directly — this pipeline
    acts as the "environment" that executes the agent's actions
    and feeds observations back to it.
    """
    def __init__(
        self,
        index_path: str = "rag_index.json",
        max_iterations: int = 5,
        top_k: int = 5,
        threshold: float = 0.3,
        search_mode: str = "vector",
        use_reranker: bool = False,
    ):
        """
        Args:
            index_path: Path to the vector store index
            max_iterations: Maximum ReAct loop iterations (default: 5)
            top_k: Default number of chunks to retrieve (default: 5)
            threshold: Minimum similarity score (default: 0.3)
            search_mode: Default search mode (default: "vector")
            use_reranker: Whether to use LLM reranking (default: False)
        """
        # 📚: We create a standard RAGPipeline under the hood.
        # The agentic layer adds intelligence ON TOP of the existing
        # retrieval and generation — it doesn't replace them.
        self.rag_pipeline = RAGPipeline(
            index_path=index_path,
            top_k=top_k,
            threshold=threshold,
            search_mode=search_mode,
            use_reranker=use_reranker,
        )
        self.max_iterations = max_iterations
        self.default_top_k = top_k
        self.default_threshold = threshold
        self.default_search_mode = search_mode
        self.use_reranker = use_reranker
    
    def auto_ingest(self, sample_dir: str = "sample_data"):
        """Delegate to the underlying RAG pipeline."""
        self.rag_pipeline.auto_ingest(sample_dir)
    
    def query(self, question: str, verbose: bool = False) -> dict:
        """
        Answer a question using the ReAct agent loop.

        📚: This is the main entry point. The flow is:
          1. Create a fresh agent for this question
          2. Loop: ask agent for action → execute action → feed observation
          3. When agent chooses Answer (or hits max iterations), generate final answer
          4. Return answer + sources + metadata about the agent's reasoning

        Args:
            question: The user's question in natural language
            verbose: If True, print the full Thought/Action/Observation trace

        Returns:
            Dict with:
              - "answer": the generated text
              - "sources": list of source files used
              - "chunks_used": the accumulated context chunks
              - "iterations": number of ReAct iterations taken
              - "steps": list of AgentStep objects (for inspection)
        """
        if len(self.rag_pipeline.store) == 0:
            print("⚠️  No documents ingested yet! Run 'ingest' first.")
            return {
                "answer": "No documents indexed.",
                "sources": [],
                "chunks_used": [],
                "iterations": 0,
                "steps": [],
            }
        agent = ReActAgent(max_iterations=self.max_iterations)

        if verbose:
            print(f"\n🤖 Agentic RAG — max {self.max_iterations} iterations")
            print(f"   Question: \"{question}\"")
            print("=" * 60)
        
        # ── ReAct Loop ────────────────────────────────────────
        # 📚: Each iteration = one Thought-Action-Observation cycle.
        # The loop ends when:
        #   1. Agent chooses Answer (has enough context) — MOST COMMON
        #   2. Max iterations reached (safety net) — RARE

        for iteration in range(1, self.max_iterations + 1):
            if verbose:
                print(f"\n--- Iteration {iteration}/{self.max_iterations} ---")

            # Ask the agent what to do next
            try:
                thought, action = agent.get_next_action(question)
            except (ValueError, Exception) as e:
                # 📚: If the LLM's response can't be parsed, we
                # fall back to answering with whatever context we have.
                # This prevents the agent from crashing on malformed output.
                if verbose:
                    print(f"  ⚠️  Agent parse error: {e}")
                    print("  ↳ Falling back to answer with current context")
                break
            
            if verbose:
                print(f"  💭 Thought: {thought}")
                print(f"  🎯 Action:  {agent._format_action(action)}")
            
            # Execute the action and get an observation
            observation = self._execute_action(
                action, agent, question, verbose
            )

            # Record this step
            step = AgentStep(thought=thought, action=action, observation=observation)
            agent.steps.append(step)

            # If the agent chose to Answer, we're done
            if isinstance(action, Answer):
                if verbose:
                    print(f"\n✅ Agent answered after {iteration} iteration(s)")
                break
        else:
            # 📚: This 'else' clause runs if the for-loop completes
            # without 'break' — meaning we hit max iterations without
            # the agent choosing to Answer. We force-answer now.
            if verbose:
                print(f"\n⏰ Max iterations ({self.max_iterations}) reached — forcing answer")
        
        # ── Generate Final Answer ─────────────────────────────
        # 📚: The agent's Answer action contains a brief response,
        # but we generate a proper answer using the full accumulated context
        # and the standard generator. This gives us streaming, source
        # attribution, and the quality of the tuned system prompt.
        if agent.accumulated_chunks:
            if verbose:
                print(f"\n📚 Using {len(agent.accumulated_chunks)} accumulated chunks from "
                      f"{len(agent.tried_queries)} unique queries")

            print("\n💬 Answer:")
            result = generate_with_sources(
                question, agent.accumulated_chunks, stream=True
            )
        else:
            # No chunks retrieved at all — generate a "no info" response
            result = {
                "answer": "I couldn't find relevant information in the documents to answer this question.",
                "sources": [],
                "chunks_used": [],
            }
        
        # Add agent metadata
        result["iterations"] = len(agent.steps)
        result["steps"] = agent.steps

        return result
    
def _execute_action(
        self, action, agent: ReActAgent, question: str, verbose: bool
    ) -> str:
        """
        Execute an agent action and return the observation string.

        📚: This method is the "environment" that the agent
        interacts with. It translates the agent's high-level decisions
        (ReformulateQuery, RequestMoreContext) into actual retriever calls.

        The observation string is fed back to the agent so it can
        reason about the results in the next iteration.
        """
        if isinstance(action, Answer):
            return "Answer provided. Stopping."
        if isinstance(action, ReformulateQuery):
            # 📚: ReformulateQuery doesn't retrieve — it just records the new query. 
            # The agent will likely follow up with RequestMoreContext using this reformulated query.
            # However, for convenience, we also retrieve immediately.
            return self._do_retrieval(
                action.new_query, self.default_top_k,
                self.default_search_mode, agent, verbose
            )

        if isinstance(action, RequestMoreContext):
            return self._do_retrieval(
                action.query, action.top_k,
                action.search_mode, agent, verbose
            )
        return "Unknown action. Please use RequestMoreContext, ReformulateQuery, or Answer."

def _do_retrieval(
        self,
        query: str,
        top_k: int,
        search_mode: str,
        agent: ReActAgent,
        verbose: bool,
    ) -> str:
        """
        Execute a retrieval and return a formatted observation.

        📚: We track which queries have been tried to help the agent avoid repeating the same search. 
        The observation includes chunk count and a preview of each chunk so the agent can judge 
        relevance without seeing full documents.
        """
        agent.tried_queries.add(query)

        retriever = Retriever(
            self.rag_pipeline.store,
            top_k=top_k,
            threshold=self.default_threshold,
            search_mode=search_mode,
            use_reranker=self.use_reranker
        )
        results = retriever.retrieve(query)

        # Add new chunks to the agent's accumulated context
        agent.add_chunks(results)

        # Build observation string for the agent
        if not results:
            observation = (
                f"No relevant chunks found for query: \"{query}\" "
                f"(search_mode={search_mode}, top_k={top_k}). "
                "Try reformulating with different terms."
            )
        else:
            chunk_summaries = []
            for i,r in enumerate(results):
                source = r["metadata"].get("source", "unknown")
                score = r.get("score", 0.0)
                preview = r["text"][:150].replace("\n", " ")
                chunk_summaries.append(
                    f"  [{i+1}] (score: {score:.3f}, source: {source}) {preview}..."
                )
            observation = (
                f"Retrieved {len(results)} chunks for \"{query}\" "
                f"(search_mode={search_mode}):\n"
                + "\n".join(chunk_summaries)
            )
        if verbose:
            print(f"  👁️  Observation: {observation[:300]}{'...' if len(observation) > 300 else ''}")

        return observation

def list_documents(self) -> list[str]:
        """List all documents that have been ingested."""
        return self.rag_pipeline.list_documents()

