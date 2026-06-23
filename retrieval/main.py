import sys

from retrieval.llm import get_llm
from retrieval.retriever import Retriever, METHODS


def main():
    llm_name = "deepseek"
    embed_name = "sentence-transformer"
    method = "vector"

    for arg in sys.argv:
        if arg.startswith("--llm="):
            llm_name = arg.split("=")[1]
        if arg.startswith("--embed="):
            embed_name = arg.split("=")[1]
        if arg.startswith("--method="):
            method = arg.split("=")[1]

    print(
        f"LLM: {llm_name} | "
        f"Embeddings: {embed_name} | "
        f"Method: {method}\n"
    )

    if method not in METHODS:
        print(
            f"Unknown method: {method}\n"
            f"Available: {METHODS}"
        )
        return

    llm = get_llm(llm_name)
    retriever = Retriever(
        llm=llm,
        embedding_name=embed_name,
        method=method,
    )

    if "--temporal" in sys.argv:
        _temporal_mode(retriever)
    else:
        _interactive_mode(retriever, method)


def _interactive_mode(
    retriever: Retriever,
    method: str,
):
    print(
        "Ask questions about Reddit discussions."
    )
    print(
        "Commands: /quit, /temporal, "
        "/subreddit <name>\n"
    )

    while True:
        question = input("You: ").strip()

        if not question:
            continue

        if question == "/quit":
            break

        if question == "/temporal":
            question = input(
                "Temporal question: "
            ).strip()
            if not question:
                continue

            print("\nAnalyzing across "
                  "time windows...\n")

            result = (
                retriever.ask_temporal_comparison(
                    question
                )
            )
            print(f"Answer:\n{result['answer']}")
            print()
            continue

        if question.startswith("/subreddit"):
            parts = question.split(maxsplit=1)
            if len(parts) < 2:
                print(
                    "Usage: /subreddit "
                    "<name> <question>"
                )
                continue

            rest = parts[1].strip()
            sub_parts = rest.split(maxsplit=1)
            if len(sub_parts) < 2:
                print(
                    "Usage: /subreddit "
                    "<name> <question>"
                )
                continue

            subreddit = sub_parts[0]
            q = sub_parts[1]

            print(
                f"\nSearching r/{subreddit}...\n"
            )
            result = retriever.ask_by_subreddit(
                question=q,
                subreddit=subreddit,
            )
            print(f"Answer:\n{result['answer']}")
            _print_sources(result["sources"])
            print()
            continue

        print(
            f"\nSearching ({method})...\n"
        )
        result = retriever.ask(question)
        print(f"Answer:\n{result['answer']}")
        _print_sources(result["sources"])
        print()


def _temporal_mode(retriever: Retriever):
    print(
        "Temporal comparison mode.\n"
        "Ask questions to compare across "
        "W1/W2/W3.\n"
    )

    while True:
        question = input("You: ").strip()

        if not question or question == "/quit":
            break

        print(
            "\nComparing across time windows...\n"
        )
        result = (
            retriever.ask_temporal_comparison(
                question
            )
        )
        print(f"Answer:\n{result['answer']}")
        print()


def _print_sources(sources: list[dict]):
    if not sources:
        return

    print("\nSources:")
    seen = set()
    for s in sources:
        if s["type"] == "post":
            url = s.get("url", "")
            if url and url not in seen:
                seen.add(url)
                print(
                    f"  - r/{s.get('subreddit', '')} "
                    f"({s.get('time_window', '')}) "
                    f"{url}"
                )


if __name__ == "__main__":
    main()
