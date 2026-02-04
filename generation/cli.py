import json

from .config import GenerationConfig
from .models import GenerateRequest
from .service import GenerationService


def main() -> None:
    config = GenerationConfig.from_env()
    service = GenerationService(config=config)

    print("Generation CLI (Ollama + Retrieval)")
    print("Type a question, or 'exit' to quit.")

    while True:
        query = input("\n> ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        request = GenerateRequest(query=query)
        response = service.generate(request)
        print(json.dumps(response.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
