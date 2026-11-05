
from langgraph.graph import StateGraph, START, END

from app.graph.state import AgentState

from app.graph.nodes.supervisor import supervisor_node
from app.graph.nodes.rag_agent import rag_agent_node
from app.graph.nodes.web_search_agent import web_search_agent_node
from app.graph.nodes.email_agent import email_agent_node
from app.graph.nodes.conversational_agent import (
    conversational_agent_node,
)


def route_to_agent(state: AgentState) -> str:
    """
    Route the request based on the Supervisor's decision.
    """

    next_agent = state.get("next_agent")

    valid_agents = {
        "rag",
        "web_search",
        "email",
        "conversational",
    }

    if next_agent not in valid_agents:
        raise ValueError(
            f"Invalid supervisor routing decision: {next_agent}"
        )

    return next_agent


def build_graph(checkpointer=None):
    """
    Build and compile the Enterprise Agentic RAG graph.

    Args:
        checkpointer: Optional LangGraph checkpointer.
                      Use AsyncPostgresSaver for persistence.
    """

    # Step 1: Initialize StateGraph.
    workflow = StateGraph(AgentState)

    # Step 2: Register all agent nodes.
    workflow.add_node(
        "supervisor",
        supervisor_node,
    )

    workflow.add_node(
        "rag",
        rag_agent_node,
    )

    workflow.add_node(
        "web_search",
        web_search_agent_node,
    )

    workflow.add_node(
        "email",
        email_agent_node,
    )

    workflow.add_node(
        "conversational",
        conversational_agent_node,
    )

    # Step 3: Connect START to Supervisor.
    workflow.add_edge(
        START,
        "supervisor",
    )

    # Step 4: Add conditional routing.
    workflow.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "rag": "rag",
            "web_search": "web_search",
            "email": "email",
            "conversational": "conversational",
        },
    )

    # Step 5: Connect every agent to END.
    workflow.add_edge("rag", END)
    workflow.add_edge("web_search", END)
    workflow.add_edge("email", END)
    workflow.add_edge("conversational", END)

    # Step 6: Compile graph with PostgreSQL persistence.
    return workflow.compile(
        checkpointer=checkpointer,
    )
