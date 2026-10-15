import json
import sys
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "brain"))

from graph import ask as ask_brain, build_graph, open_checkpointer, this_turn
from memory import load_facts, save_facts

PAGE = Path(__file__).parent / "index.html"
AVATARS = {"glb": Path(__file__).parent / "avatar.glb", "vrm": Path(__file__).parent / "avatar.vrm"}

graph = build_graph(open_checkpointer())  # memory saved on disk
# Which live step the page shows while each tool runs.
TOOL_STEPS = {"search_documents": "search", "get_weather": "weather", "get_leads": "leads"}
app = FastAPI()


class Question(BaseModel):
    text: str
    session_id: str | None = None
    user_id: str = "guest"


@app.get("/")
def index():
    return FileResponse(PAGE, headers={"Cache-Control": "no-cache"})


@app.get("/avatar.{ext}")
def avatar(ext: str):
    path = AVATARS.get(ext)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Put your model at face/avatar.glb or face/avatar.vrm")
    return FileResponse(path, media_type="model/gltf-binary")


class Person(BaseModel):
    name: str


@app.post("/hello")
def hello(person: Person):
    """Look up a user by name. New users get their name saved; returning users get their facts back."""
    name = person.name.strip()
    user_id = name.lower()
    facts = load_facts(user_id)
    returning = bool(facts)
    if not returning:
        save_facts(user_id, {"name": name})
    return {"user_id": user_id, "returning": returning, "facts": load_facts(user_id)}


@app.post("/ask")
async def ask(question: Question):
    session_id = question.session_id or uuid.uuid4().hex
    try:
        result = await run_in_threadpool(ask_brain, graph, question.text, question.user_id, session_id)
    except Exception as e:
        return {
            "answer": "Sorry, I can't reach the AI service right now. Please try again in a minute.",
            "error": str(e),
            "session_id": session_id,
            "tools": [],
        }
    return {"answer": result["answer"], "session_id": session_id, "tools": result["tools"]}


@app.post("/ask/stream")
def ask_stream(question: Question):
    """Like /ask, but sends one JSON line per brain step so the face can react live."""
    session_id = question.session_id or uuid.uuid4().hex
    config = {"configurable": {"thread_id": session_id}}

    def events():
        try:
            start = {"messages": [HumanMessage(question.text)], "user_id": question.user_id}
            for update in graph.stream(start, config, stream_mode="updates"):
                for node, values in update.items():
                    if node == "remember" and values.get("facts"):
                        yield json.dumps({"type": "step", "step": "remember"}) + "\n"
                    elif node == "agent":
                        calls = values["messages"][-1].tool_calls
                        if calls:
                            step = TOOL_STEPS.get(calls[0]["name"], "search")
                        else:
                            step = "check"
                        yield json.dumps({"type": "step", "step": step}) + "\n"
                    elif node == "check" and values.get("feedback"):
                        yield json.dumps({"type": "step", "step": "retry"}) + "\n"
            messages = graph.get_state(config).values["messages"]
            tools = [c["name"] for m in this_turn(messages) for c in getattr(m, "tool_calls", [])]
            answer = {"answer": messages[-1].text, "session_id": session_id, "tools": tools}
        except Exception as e:
            answer = {"answer": "Sorry, I can't reach the AI service right now. Please try again in a minute.",
                      "error": str(e), "session_id": session_id, "tools": []}
        yield json.dumps({"type": "answer", **answer}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8090)
