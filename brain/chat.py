"""Chat with Aria's LangGraph brain in the terminal. Type 'quit' to stop."""
import uuid

from graph import ask, build_graph, open_checkpointer
from memory import load_facts

graph = build_graph(open_checkpointer())

user_id = input("Your user name (e.g. seema): ").strip().lower() or "guest"
facts = load_facts(user_id)
print("Aria remembers:", facts if facts else "nothing yet")

session_id = f"{user_id}-{uuid.uuid4().hex[:8]}"  # a new conversation each time you start
print("Type your question (or 'quit').\n")

while True:
    question = input("You:  ").strip()
    if question.lower() in {"quit", "exit", "q"}:
        break
    if not question:
        continue
    try:
        result = ask(graph, question, user_id, session_id)
    except Exception as e:
        print("Aria: Sorry, I can't reach the AI service right now.", f"({e})\n")
        continue
    print("Aria:", result["answer"])
    if result["tools"]:
        print("      tools used:", ", ".join(result["tools"]))
    print()
