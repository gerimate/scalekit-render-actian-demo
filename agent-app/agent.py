"""
LangGraph agent graph.

Graph topology:
    START → recall → agent ─┬─ tools → agent (loop)
                             └─ remember → END

- recall_node:   Query this user's VectorAI collection for relevant past turns.
- agent_node:    Call the LLM (with Scalekit-authenticated tools bound) using
                 recalled memories in the system prompt.
- ToolNode:      Execute any tool calls the LLM returned.
- remember_node: Write the latest user+assistant turn back to VectorAI.

The graph is rebuilt per-request (because the tool list is per-user and
fetched fresh each time).  There is no shared mutable state between requests.
"""

from __future__ import annotations

import logging
import os
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from memory import recall_memories, remember_turn

log = logging.getLogger(__name__)

LLM_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    user_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    recalled_memories: list[str]


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

def recall_node(state: AgentState) -> dict:
    """Fetch relevant memories from this user's VectorAI collection."""
    user_id = state["user_id"]
    messages = state["messages"]

    # Use the latest human message as the recall query.
    query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            query = msg.content
            break

    memories = recall_memories(user_id, query, k=5)
    log.debug("Recalled %d memories for %s", len(memories), user_id)
    return {"recalled_memories": memories}


def make_agent_node(llm_with_tools):
    """Return a closure over the bound LLM so the node is a plain function."""

    def agent_node(state: AgentState) -> dict:
        memories = state.get("recalled_memories", [])
        if memories:
            memory_block = "\n---\n".join(memories)
            memory_context = (
                "Relevant memories from past conversations with this user:\n"
                + memory_block
            )
        else:
            memory_context = "No prior memories for this user."

        system = SystemMessage(
            content=(
                "You are a helpful personal assistant. "
                "You have access to the user's prior conversation memories and "
                "external tools authenticated via Scalekit on their behalf.\n\n"
                + memory_context
            )
        )
        response = llm_with_tools.invoke([system] + list(state["messages"]))
        return {"messages": [response]}

    return agent_node


def remember_node(state: AgentState) -> dict:
    """
    Persist the most recent user+assistant exchange to VectorAI DB.

    This runs once per graph execution, after the tool loop has finished
    and the LLM has produced its final response.
    """
    user_id = state["user_id"]
    messages = state["messages"]

    human_text = ""
    ai_text = ""

    for msg in reversed(messages):
        if not ai_text and isinstance(msg, AIMessage) and not msg.tool_calls:
            ai_text = msg.content
        if not human_text and isinstance(msg, HumanMessage):
            human_text = msg.content
        if human_text and ai_text:
            break

    if human_text and ai_text:
        ids = remember_turn(user_id, human_text, ai_text)
        log.debug("Persisted turn for %s → ids %s", user_id, ids)
    else:
        log.warning("remember_node: could not find a complete turn to persist")

    return {}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "remember"


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def build_agent_graph(tools: list, llm: ChatAnthropic | None = None):
    """
    Compile and return the LangGraph graph.

    Args:
        tools: LangChain-compatible tool objects (from Scalekit MCP or direct
               adapter).  Pass an empty list to run memory-only (no external
               tool calls).
        llm:   Optional pre-built ChatOpenAI instance.  Defaults to
               gpt-4o-mini.
    """
    if llm is None:
        llm = ChatAnthropic(model=LLM_MODEL, temperature=0)

    llm_with_tools = llm.bind_tools(tools) if tools else llm

    builder = StateGraph(AgentState)

    builder.add_node("recall", recall_node)
    builder.add_node("agent", make_agent_node(llm_with_tools))
    builder.add_node("remember", remember_node)

    if tools:
        builder.add_node("tools", ToolNode(tools))
        builder.add_conditional_edges(
            "agent",
            _should_continue,
            {"tools": "tools", "remember": "remember"},
        )
        builder.add_edge("tools", "agent")
    else:
        builder.add_edge("agent", "remember")

    builder.add_edge(START, "recall")
    builder.add_edge("recall", "agent")
    builder.add_edge("remember", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# High-level run helper used by main.py
# ---------------------------------------------------------------------------

async def run_agent(
    user_id: str,
    message: str,
    tools: list,
) -> dict:
    """
    Run one turn of the agent graph for the given user.

    Returns a dict with:
        response       (str)  — the final assistant message
        memories_used  (int)  — number of memories injected into context
    """
    graph = build_agent_graph(tools)

    initial_state: AgentState = {
        "user_id": user_id,
        "messages": [HumanMessage(content=message)],
        "recalled_memories": [],
    }

    final_state = await graph.ainvoke(initial_state)

    response_text = ""
    for msg in reversed(final_state["messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            response_text = msg.content
            break

    return {
        "response": response_text,
        "memories_used": len(final_state.get("recalled_memories", [])),
    }
