# Argo API Bridge

This project provides a compatibility layer that transforms OpenAI-style API requests into Argonne National Lab's Argo API format. It supports chat completions, text completions, and embeddings endpoints.

## Setup

### 1. Create Conda Environment

First, create a new conda environment with Python 3.12:

```bash
conda create -n argo_bridge python=3.12
conda activate argo_bridge
```

### 2. Install Requirements

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

### 3. Run the Server

Start the server with default settings (port 7285):

```bash
python argo_bridge.py
```

## Configuration Options

The server supports the following command-line arguments:

- `--username`: Set the username for API requests (default: 'APS')
- `--port`: Set the port number for the server (default: 7285)
- `--dlog`: Enable debug-level logging (when set, logging is at DEBUG level; by default it is INFO level)

Example with custom settings:

```bash
python argo_bridge.py --username custom_user --port 8000 --dlog
```

## Endpoints

The API exposes the following endpoints:

- **Chat Completions**: `/chat/completions` (POST)
- **Text (Legacy) Completions**: `/completions` (POST)
- **Embeddings**: `/embeddings` (POST)

## Supported Models

The server accepts both Argo and OpenAI model identifiers.

### Chat and Completion Models

- GPT-3.5: (`gpt35`, `gpt-3.5`)
- GPT-3.5 Large: (`gpt35large`)
- GPT-4: (`gpt4`, `gpt-4`)
- GPT-4 Large: (`gpt4large`)
- GPT-4 Turbo: (`gpt4turbo`, `gpt-4-turbo`)
- GPT-4o: (`gpt4o`, `gpt-4o`, `gpt-4o-mini`)
- GPT-o1 Preview: (`gpto1preview`, `o1-preview`)
- GPT-o1 Mini: (`gpto1mini`, `o1-mini`, `o1mini`)
- GPT-o3 Mini: (`gpto3mini`, `o3-mini`, `o3mini`)

### Embedding Models

- v3small: (`text-embedding-3-small`, `v3small`)
- v3large: (`text-embedding-3-large`, `v3large`)
- ada002: (`text-embedding-ada-002`, `ada002`)

## Production Deployment
For personal use, the development server should be plenty, but if you wish to scale up gunicorn is included in the requirements. 

`gunicorn --workers 4 --bind localhost:7285 argo_bridge:app`


## Testing

Run the test suite using unittest:

```bash
python -m unittest test_server.py
