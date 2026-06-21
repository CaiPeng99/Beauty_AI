"""
📚 LEARN: CLI — Command-Line Interface
========================================
This is how users interact with the RAG system from the terminal.
We use Python's built-in argparse module (no external dependencies).

Usage:
  python cli.py ingest --source <file_or_url>   Ingest a document
  python cli.py query "your question"            Ask a question
  python cli.py list                             List indexed documents
"""

import argparse
import sys
from rag.pipeline import RAGPipeline

def main():
    parser = argparse.ArgumentParser(
        description="🔍 RAG From Scratch — Ask questions about your documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py ingest --source sample_data/sample.md
  python cli.py ingest --source https://example.com
  python cli.py query "What is machine learning?"
  python cli.py query "What is overfitting?" --verbose
  python cli.py list
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── Ingest command ─────────────────────────────────────────
    '''
    add_parser("ingest", ...)：创建一个专门处理 ingest 子命令的解析器。返回的 ingest_parser 可以继续添加参数。
    add_argument()：为这个子命令添加参数。
        "--source", "-s"：同时支持长选项和短选项。-s 是 --source 的别名。
        required=True：该参数必须提供，否则 argparse 会报错。
        type=int：将输入的字符串转换为整数类型（如果转换失败则报错）。
        default=500：如果用户未提供该选项，则使用默认值 500。
        help：帮助信息中显示的说明。
    参数	        是否必需	                说明
    --source 或 -s	✅ 必需	            指定要摄入的文档路径（本地文件）或 URL。
    --chunk-size	❌ 可选（默认 500）	分块的目标字符数。
    --overlap	    ❌ 可选（默认 100）	相邻块之间的重叠字符数。
    '''
    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest a document (PDF, text, or URL)"
    )
    ingest_parser.add_argument(
        "--source", "-s",
        required=True,
        help="Path to a file or URL to ingest",
    )
    ingest_parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Target chunk size in characters (default: 500)",
    )
    ingest_parser.add_argument(
        "--overlap",
        type=int,
        default=100,
        help="Overlap between chunks in characters (default: 100)",
    )

    # ── Query command ──────────────────────────────────────────
    '''
    位置参数 "question"：没有 -- 前缀的参数是位置参数，用户必须按顺序提供。例如 query "What is AI?"，字符串 "What is AI?" 会赋值给 args.question。
    choices=[...]：限定参数的值只能是指定列表中的某一个，否则报错。
    action="store_true"：布尔标志。如果命令行中出现 --rerank，则 args.rerank = True；否则为 False。常用于开关选项。
    '''
    query_parser = subparsers.add_parser(
        "query", help="Ask a question about your documents"
    )
    query_parser.add_argument(
        "question",
        help="Your question in natural language",
    )
    query_parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )
    query_parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.3,
        help="Minimum similarity score (default: 0.3)",
    )
    query_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show retrieved chunks before the answer",
    )
    query_parser.add_argument(
        "--search-mode", "-m",
        choices=["vector", "keyword", "hybrid"],
        default="vector",
        help="Search mode: vector (default), keyword (BM25), or hybrid (both)",
    )
    query_parser.add_argument(
        "--rerank",
        action="store_true",
        help="Rerank results with LLM for higher precision (slower)",
    )

    # ── List command ───────────────────────────────────────────
    subparsers.add_parser("list", help="List all indexed documents")

    # ── Agentic Query command ─────────────────────────────────
    agentic_parser = subparsers.add_parser(
        "agentic-query", help="Ask a question using the ReAct agent (iterative retrieval)"
    )
    agentic_parser.add_argument(
        "question",
        help="Your question in natural language",
    )
    agentic_parser.add_argument(
        "--max-iterations", "-i",
        type=int,
        default=5,
        help="Maximum ReAct loop iterations (default: 5)",
    )
    agentic_parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Default number of chunks to retrieve per iteration (default: 5)",
    )
    agentic_parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.3,
        help="Minimum similarity score (default: 0.3)",
    )
    agentic_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show the full Thought/Action/Observation reasoning trace",
    )
    agentic_parser.add_argument(
        "--search-mode", "-m",
        choices=["vector", "keyword", "hybrid"],
        default="vector",
        help="Default search mode (default: vector). The agent may switch modes during iterations.",
    )
    agentic_parser.add_argument(
        "--rerank",
        action="store_true",
        help="Rerank results with LLM for higher precision (slower)",
    )

    # ── Eval command ───────────────────────────────────────────
    eval_parser = subparsers.add_parser(
        "eval", help="Run evaluation against test questions"
    )
    eval_parser.add_argument(
        "--test-file",
        default="eval/test_questions.json",
        help="Path to test questions JSON (default: eval/test_questions.json)",
    )
    eval_parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )
    eval_parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.3,
        help="Minimum similarity score (default: 0.3)",
    )
    eval_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed results per question",
    )
    eval_parser.add_argument(
        "--search-mode", "-m",
        choices=["vector", "keyword", "hybrid"],
        default="vector",
        help="Search mode: vector (default), keyword (BM25), or hybrid (both)",
    )
    eval_parser.add_argument(
        "--rerank",
        action="store_true",
        help="Rerank results with LLM for higher precision (slower)",
    )

    # ── Parse and execute ──────────────────────────────────────
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "ingest":
        pipeline = RAGPipeline(
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
        )
        pipeline.ingest(args.source)

    elif args.command == "query":
        pipeline = RAGPipeline(
            top_k=args.top_k,
            threshold=args.threshold,
            search_mode=args.search_mode,
            use_reranker=args.rerank,
        )
        # Auto-ingest sample documents on first run
        pipeline.auto_ingest()
        result = pipeline.query(args.question, verbose=args.verbose)

        # Show sources
        if result["sources"]:
            print(f"\n📚 Sources: {', '.join(result['sources'])}")

    elif args.command == "list":
        pipeline = RAGPipeline()
        # Auto-ingest sample documents on first run
        pipeline.auto_ingest()
        sources = pipeline.list_documents()
        if sources:
            print("📚 Indexed documents:")
            for s in sources:
                print(f"  • {s}")
        else:
            print("No documents indexed yet. Run 'ingest' first.")

    elif args.command == "agentic-query":
        from rag.agentic_pipeline import AgenticRAGPipeline

        pipeline = AgenticRAGPipeline(
            max_iterations=args.max_iterations,
            top_k=args.top_k,
            threshold=args.threshold,
            search_mode=args.search_mode,
            use_reranker=args.rerank,
        )
        # Auto-ingest sample documents on first run
        pipeline.auto_ingest()# auto_ingest()方法 没有返回值（返回 None），也没有删除 self 或重置任何标志。
        # 它仅仅 修改对象内部状态（往 self.store 中添加一些文档）。
        # 调用完成后，pipeline 对象已经拥有了索引数据，接下来调用 query() 自然可以正常工作。
        result = pipeline.query(args.question, verbose=args.verbose)

        # Show sources
        if result["sources"]:
            print(f"\n📚 Sources: {', '.join(result['sources'])}")
        print(f"🔄 Iterations: {result['iterations']}")

    elif args.command == "eval":
        from rag.evaluator import load_test_questions, run_evaluation, save_results, print_scorecard

        # Auto-ingest if needed
        pipeline = RAGPipeline()
        pipeline.auto_ingest() 

        # Load test questions
        print(f"📋 Loading test questions from: {args.test_file}")
        questions = load_test_questions(args.test_file)
        print(f"   → {len(questions)} questions loaded\n")

        # Run evaluation
        results = run_evaluation(
            questions,
            top_k=args.top_k,
            threshold=args.threshold,
            verbose=args.verbose,
            search_mode=args.search_mode,
            use_reranker=args.rerank,
        )

        # Print scorecard
        print_scorecard(results)

        # Save results
        filepath = save_results(results)
        print(f"\n💾 Results saved to: {filepath}")


if __name__ == "__main__":
    main()


