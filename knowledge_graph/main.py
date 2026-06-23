import os
import sys

from dotenv import load_dotenv

load_dotenv()


def main():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

    if not uri or not user or not password:
        print(
            "Missing Neo4j credentials.\n"
            "Set NEO4J_URI, NEO4J_USER, "
            "NEO4J_PASSWORD in .env\n"
            "\n"
            "Example (AuraDB free tier):\n"
            "  NEO4J_URI=neo4j+s://xxxx."
            "databases.neo4j.io\n"
            "  NEO4J_USER=neo4j\n"
            "  NEO4J_PASSWORD=your_password\n"
            "\n"
            "Example (local Docker):\n"
            "  NEO4J_URI=bolt://localhost:7687\n"
            "  NEO4J_USER=neo4j\n"
            "  NEO4J_PASSWORD=password"
        )
        return

    from knowledge_graph.builder import (
        GraphBuilder,
    )

    print(f"Connecting to Neo4j at {uri}...")

    builder = GraphBuilder(uri, user, password)

    try:
        if "--clear" in sys.argv:
            builder.clear()

        if "--stats" in sys.argv:
            builder.stats()
        else:
            builder.build()
    finally:
        builder.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
