#!/bin/bash
# ---
# Script to start the JobScraper backend and frontend dev servers.
# (Using xterm to bypass terminal library conflicts)
# ---

# 1. Define the main project directory
PROJECT_DIR="/home/david/Git/JobScraper"

wget -qO- https://astral.sh/uv/install.sh | sh

echo "Navigating to project directory: $PROJECT_DIR"
cd "$PROJECT_DIR" || { echo "Error: Project directory not found at $PROJECT_DIR"; exit 1; }

# ---
# 2. Spawn the Backend Process
# ---
echo "Activating venv..."; 
source .venv/bin/activate; 
echo "Syncing dependencies with uv..."; 
uv sync; 
echo "Starting LangGraph dev server..."; 
uv run langgraph dev; 
