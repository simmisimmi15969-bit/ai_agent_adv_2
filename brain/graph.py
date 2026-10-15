"""Aria's brain as a LangGraph flow: remember -> agent <-> tools -> check -> save."""
import json
import re
import sqlite3
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from memory import DB_PATH, load_facts, save_facts
from models import get_model
from tools import TOOLS

CHATS_DB = DB_PATH.parent / "chats.db"
MAX_HISTORY = 20  # how many recent messages the model sees

INSTRUCTION = """You are Aria, a friendly AI assistant for company policies.
Decide which source to use:
- Live data (like current weather): call get_weather. Do NOT call search_documents for these.
- Sales leads (projects to bid on, companies to approach, new AI/web projects to learn from,
  "today's leads", "any blockchain projects?"): call get_leads. Pick kind and service from the
  user's words; leave them empty if not mentioned. Do NOT call search_documents for these.
  Present at most 3 leads, best first: score out of 10, title, budget in dollars if known,
  and the short reason. Put each link on its own line at the end, and say the links are in the chat.
  If there are no leads, say so and suggest running the Lead Finder.
- Questions about the company (policies, HR rules, leave, work from home, IT): call search_documents
  with a short, focused query. Always write that query in English, because the documents are in English.
  For follow-up questions, include the topic from earlier in the conversation in the query.
- Reply in the same language as the user's latest message: Hindi (in Devanagari) if they write in Hindi,
  English if they write in English.
- Answer ONLY from passages that actually contain the answer, and cite them exactly as
  '(source: file, page N)' with the file name unchanged.
- If none of the passages answer the question, start with
  'Not found in your documents, answering from general knowledge:' and answer from your own knowledge.
- If a tool returns an error, tell the user politely. Never invent facts about the company documents.
- Keep answers short and clear: they are spoken aloud by your avatar."""

CHECK_PROMPT = """You are a strict fact checker.
Passages from the company documents:
{passages}

Answer given to the user:
{answer}

Is every company-specific fact in the answer supported by the passages?
An answer that says the information was not found in the documents also counts as supported.
Reply with exactly YES, or NO followed by one short sentence saying what is unsupported."""

EXTRACT_PROMPT = """From the user's message below, extract lasting personal facts about the user
(for example name, role, team, department, city, preferences).
Return only a JSON object with short lowercase keys and values written exactly as the user wrote them
(for example {{"name": "Seema"}}), or {{}} if there are none.

Message: {message}"""

# Only spend an AI call on fact extraction when the user talks about themselves.
SELF_INFO = re.compile(r"\b(my|i am|i'm|im|call me|i work|i prefer|mera|meri|main)\b|मेरा|मेरी|मैं|मुझे", re.I)


class State(TypedDict):
    messages: Annotated[list, add_messages]  # the conversation; add_messages appends instead of replacing
    user_id: str
    facts: dict      # long-term memory about this user
    feedback: str    # why the checker rejected the last answer ("" if accepted)
    retries: int     # how many times the checker sent us back this turn


agent_model = get_model(TOOLS)  # can call tools
helper_model = get_model()      # plain model for checking and fact extraction


def this_turn(messages: list) -> list:
    """Messages from the user's latest question to the end."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i:]
    return messages


def recent(messages: list) -> list:
    """The last MAX_HISTORY messages, always starting at a user message (so no tool result is cut off)."""
    starts = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    for i in starts:
        if len(messages) - i <= MAX_HISTORY:
            return messages[i:]
    return messages[starts[-1]:] if starts else messages


# ---------------- nodes ----------------

def remember(state: State) -> dict:
    return {"facts": load_facts(state["user_id"]), "feedback": "", "retries": 0}


def agent(state: State) -> dict:
    system = INSTRUCTION
    if state["facts"]:
        system += "\n\nWhat you remember about this user from earlier chats: " + json.dumps(state["facts"], ensure_ascii=False)
        system += "\nUse it naturally (for example their name), without repeating it every time."
    if state["feedback"]:
        system += ("\n\nYour previous answer was rejected by the fact checker: " + state["feedback"] +
                   "\nSearch again with different words, or say it is not in the documents.")
    reply = agent_model.invoke([SystemMessage(system)] + recent(state["messages"]))
    return {"messages": [reply]}


def check(state: State) -> dict:
    turn = this_turn(state["messages"])
    passages = [m.content for m in turn if isinstance(m, ToolMessage) and m.name == "search_documents"]
    if not passages or state["retries"] >= 1:
        return {"feedback": ""}  # nothing to check, or already retried once
    answer = state["messages"][-1]
    verdict = helper_model.invoke(CHECK_PROMPT.format(passages="\n\n".join(passages), answer=answer.text)).text.strip()
    if verdict.upper().startswith("YES"):
        return {"feedback": ""}
    # Remove the rejected answer and go back to the agent with the checker's reason.
    return {"feedback": verdict, "retries": state["retries"] + 1, "messages": [RemoveMessage(id=answer.id)]}


def save(state: State) -> dict:
    turn = this_turn(state["messages"])
    question = turn[0].text
    facts = {"language": "Hindi" if re.search(r"[ऀ-ॿ]", question) else "English"}
    searches = [call["args"].get("query") for m in turn if isinstance(m, AIMessage)
                for call in m.tool_calls if call["name"] == "search_documents"]
    if searches:
        facts["last_topic"] = searches[-1]
    if SELF_INFO.search(question):
        reply = helper_model.invoke(EXTRACT_PROMPT.format(message=question)).text
        found = re.search(r"\{.*\}", reply, re.S)
        try:
            facts.update(json.loads(found.group()) if found else {})
        except json.JSONDecodeError:
            pass  # the model returned something that isn't JSON: skip it
    save_facts(state["user_id"], facts)
    return {"facts": {**state["facts"], **facts}}


# ---------------- edges ----------------

def after_agent(state: State) -> str:
    return "tools" if state["messages"][-1].tool_calls else "check"


def after_check(state: State) -> str:
    return "agent" if state["feedback"] else "save"


def build_graph(checkpointer=None):
    graph = StateGraph(State)
    graph.add_node("remember", remember)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("check", check)
    graph.add_node("save", save)

    graph.add_edge(START, "remember")
    graph.add_edge("remember", "agent")
    graph.add_conditional_edges("agent", after_agent, ["tools", "check"])
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges("check", after_check, ["agent", "save"])
    graph.add_edge("save", END)
    return graph.compile(checkpointer=checkpointer)


def open_checkpointer() -> SqliteSaver:
    """Short-term memory: every conversation is saved in memory/chats.db."""
    CHATS_DB.parent.mkdir(exist_ok=True)
    return SqliteSaver(sqlite3.connect(CHATS_DB, check_same_thread=False))


def ask(graph, text: str, user_id: str, session_id: str) -> dict:
    """Send one message and return Aria's answer plus the tools she used."""
    config = {"configurable": {"thread_id": session_id}}
    result = graph.invoke({"messages": [HumanMessage(text)], "user_id": user_id}, config)
    turn = this_turn(result["messages"])
    tools = [call["name"] for m in turn if isinstance(m, AIMessage) for call in m.tool_calls]
    return {"answer": result["messages"][-1].text, "tools": tools}


if __name__ == "__main__":
    graph = build_graph(open_checkpointer())
    print(graph.get_graph().draw_mermaid())

    for question in ["Hi, my name is Seema. How many days can I work from home?",
                     "and what do I need to do before that?"]:
        print("\nYou: ", question)
        result = ask(graph, question, user_id="seema_test", session_id="test_session")
        print("Aria:", result["answer"])
        print("      tools used:", result["tools"])
    print("\nRemembered about seema_test:", load_facts("seema_test"))
