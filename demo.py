"""Runs one task through the agent. Pass --ollama to use a local model."""
import sys
from agent import run_agent
from benchmark_tasks import wavelength
from llm import FaultyPlanner, OllamaLLM

t = wavelength(28)
llm = OllamaLLM() if "--ollama" in sys.argv else FaultyPlanner(t, 0.35, 2)
r = run_agent(t["q"], llm)
print("Task:", t["q"])
for step, value in r.trace:
    print(" ", step, "->", value)
print("Answer:", r.answer, "| correct:", t["truth"])
