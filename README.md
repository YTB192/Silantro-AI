# AI Prompt Director

AI Prompt Director is a locally running AI tool that helps you improve image and video prompts.

It uses Ollama to run the AI model locally on your computer. No API key or paid AI service is required.

## Requirements

Before running the application, install:

1. **Python** — https://www.python.org/downloads/
2. **Ollama** — https://ollama.com/

## Installation

### 1. Download the project

Download and extract the AI Prompt Director ZIP file to a folder on your computer.

### 2. Install the AI model

Open PowerShell and run:

```powershell
ollama pull qwen3.5:9b
```

Wait for the model download to finish.

### 3. Install the Python dependencies

In PowerShell, navigate to the extracted project folder.

Run:

```powershell
py -m pip install -r requirements.txt
```

### 4. Start AI Prompt Director

In the same project folder, run:

```powershell
py -m streamlit run app.py
```

The application should open in your web browser.

If it does not open automatically, visit:

http://localhost:8501

## Important

- Ollama must be installed and running.
- The `qwen3.5:9b` model must be downloaded before using the application.
- AI Prompt Director runs locally on your computer.
- Keep the PowerShell window open while using the application.