from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model
from typing import List, TypedDict
from langchain.agents import create_agent

from langgraph.graph import StateGraph, START, END
from langsmith import Client
import json

load_dotenv("/home/david/Git/JobScraper/.env")

class State(TypedDict):
    submission_email: str


client = Client()
writing_model = init_chat_model("gemini-2.5-flash", model_provider="google-genai", temperature=0.85)


def output_node(state: MessagesState) -> State:
    """Generate full cover letter"""

    job_description = state["messages"][-1].content

    prompt = client.pull_prompt("jobscraper_submission_email_prompt")
    prompt = prompt.invoke({"job_description": job_description})
    response = writing_model.invoke(prompt).content

    return State(submission_email=response)


# Build workflow
builder = StateGraph(MessagesState)

# Add nodes
builder.add_node("output_node", output_node)


builder.add_edge(START, "output_node")
builder.add_edge("output_node", END)


graph = builder.compile()