import os
import datetime
import requests
from flask import Flask, request, jsonify, Response
import time
import json
import logging
import argparse

app = Flask(__name__)

# Model names are different between OpenAI and Argo API
MODEL_MAPPING = {
    'gpt35': 'gpt35',
    'gpt-3.5': 'gpt35',

    'gpt-3.5-turbo': 'gpt35large',
    'gpt35large': 'gpt35large',

    'gpt4': 'gpt4',
    'gpt-4': 'gpt4',

    'gpt4large': 'gpt4large',

    'gpt4turbo': 'gpt4turbo',
    'gpt-4-turbo': 'gpt4turbo',

    'gpt-4o': 'gpt4o',
    'gpt4o': 'gpt4o',
    'gpt-4o-mini': 'gpt4o',

    'gpto1preview': 'gpto1preview',
    'o1-preview': 'gpto1preview',
}
EMBEDDING_MODEL_MAPPING = {
    'text-embedding-3-small': 'v3small',
    'v3small': 'v3small',

    'text-embedding-3-large': 'v3large',
    'v3large': 'v3large',

    'text-embedding-ada-002': 'ada002',
    'ada002': 'ada002',
}

DEFAULT_MODEL = "gpt4o"
ANL_USER = "APS"
ANL_LLM_URL = 'https://apps.inside.anl.gov/argoapi/api/v1/resource/chat/'
ANL_EMBED_URL = 'https://apps.inside.anl.gov/argoapi/api/v1/resource/embed/'
ANL_DEBUG_FP = 'log_bridge.log'

"""
=================================
    Chat Endpoint
=================================
"""

@app.route('/chat/completions', methods=['POST'])
@app.route('/v1/chat/completions', methods=['POST']) #LMStudio Compatibility
def chat_completions():
    logging.info("Received chat completions request")

    data = request.get_json()
    model_base = data.get("model", DEFAULT_MODEL)
    is_streaming = data.get("stream", False)
    temperature = data.get("temperature", 0.1)
    stop = data.get("stop", [])

    if model_base not in MODEL_MAPPING:
        return jsonify({"error": {
            "message": f"Model '{model_base}' not supported."
        }}), 400
    
    model = MODEL_MAPPING[model_base]

    logging.debug(f"Received request: {data}")

    req_obj = {
        "user": ANL_USER,
        "model": model,
        "messages": data['messages'],
        "system": "",
        "stop": stop,
        "temperature": temperature
    }

    logging.debug(f"Argo Request {req_obj}")

    response = requests.post(ANL_LLM_URL, json=req_obj)
    if not response.ok:
        logging.error(f"Internal API error: {response.status_code} {response.reason}")
        return jsonify({"error": {
            "message": f"Internal API error: {response.status_code} {response.reason}"
        }}), 500

    json_response = response.json()
    text = json_response.get("response", "")

    logging.debug(f"Response Text {text}")

    if is_streaming: 
        return Response(_stream_chat_response(text, model), mimetype='text/event-stream')
    else:
        return jsonify(_static_chat_response(text, model_base))

def _stream_chat_response(text, model):
    chunk = {
        "id": 'argo',
        "object": "chat.completion.chunk",
        "created": int(datetime.datetime.now().timestamp()),
        "model": model,
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [{
            "index": 0,
            "delta": {'role': 'assistant', 'content': text, 'refusal': None},
            "logprobs": None,
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    time.sleep(0.8)

    yield "data: [DONE]\n\n"


def _static_chat_response(text, model):
    return {
        "id": "argo",
        "object": "chat.completion",
        "created": int(datetime.datetime.now().timestamp()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text,
            },
            "logprobs": None,
            "finish_reason": "stop"
        }]
    }


"""
=================================
    Completions Endpoint
=================================
"""


@app.route('/completions', methods=['POST'])
@app.route('/v1/completions', methods=['POST']) #LMStudio Compatibility
def completions():
    logging.info("Received completions request")
    data = request.get_json()
    prompt = data.get("prompt", "")
    stop = data.get("stop", [])
    temperature = data.get("temperature", 0.1)
    model_base = data.get("model", DEFAULT_MODEL)
    is_streaming = data.get("stream", False)

    if model_base not in MODEL_MAPPING:
        return jsonify({"error": {
            "message": f"Model '{model_base}' not supported."
        }}), 400
    
    model = MODEL_MAPPING[model_base]

    logging.debug(f"Received request: {data}")

    req_obj = {
        "user": ANL_USER,
        "model": model,
        "prompt": [data['prompt']],
        "system": "",
        "stop": stop,
        "temperature": temperature
    }

    logging.debug(f"Argo Request {req_obj}")

    response = requests.post(ANL_LLM_URL, json=req_obj)
    if not response.ok:
        logging.error(f"Internal API error: {response.status_code} {response.reason}")
        return jsonify({"error": {
            "message": f"Internal API error: {response.status_code} {response.reason}"
        }}), 500

    json_response = response.json()
    text = json_response.get("response", "")
    logging.debug(f"Response Text {text}")

    if is_streaming:
        return Response(_stream_completions_response(text, model), mimetype='text/event-stream')
    else:
        return jsonify(_static_completions_response(text, model_base))


def _static_completions_response(text, model):
    return {
        "id": "argo",
        "object": "text_completion",
        "created": int(datetime.datetime.now().timestamp()),
        "model": model,
        "choices": [{
            "text": text,
            "logprobs": None,
            "finish_reason": "stop"
        }]
    }

def _stream_completions_response(text, model):
    chunk = {
        "id": 'abc',
        "object": "text_completion",
        "created": int(datetime.datetime.now().timestamp()),
        "model": model,
        "choices": [{
            "text": text,
            'index': 0,
            "logprobs": None,
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


"""
=================================
    Embeddings Endpoint
=================================
"""
@app.route('/embeddings', methods=['POST'])
@app.route('/v1/embeddings', methods=['POST'])
def embeddings():
    logging.info("Recieved embeddings request")
    data = request.get_json()
    model_base = data.get("model", "v3small")
    
    # Check if the model is supported
    if model_base not in EMBEDDING_MODEL_MAPPING:
        return jsonify({"error": {
            "message": f"Embedding model '{model_base}' not supported."
        }}), 400
    model = EMBEDDING_MODEL_MAPPING[model_base]
    
    input_data = data.get("input", [])
    if isinstance(input_data, str):
        input_data = [input_data]
    
    embedding_vectors = _get_embeddings_from_argo(input_data, model)
    
    response_data = {
        "object": "list",
        "data": [],
        "model": model_base,
        "usage": {
            "prompt_tokens": 0,  # We don't have token counts from Argo
            "total_tokens": 0
        }
    }
    
    for i, embedding in enumerate(embedding_vectors):
        response_data["data"].append({
            "object": "embedding",
            "embedding": embedding,
            "index": i
        })
    
    return jsonify(response_data)
    

def _get_embeddings_from_argo(texts, model):
    BATCH_SIZE = 16
    all_embeddings = []
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        
        payload = {
            "user": ANL_USER,
            "model": model,
            "prompt": batch_texts
        }
        
        logging.debug(f"Sending embedding request for batch {i // BATCH_SIZE + 1}: {payload}")
        
        response = requests.post(ANL_EMBED_URL, json=payload)
        
        if not response.ok:
            logging.error(f"Embedding API error: {response.status_code} {response.reason}")
            raise Exception(f"Embedding API error: {response.status_code} {response.reason}")
        
        embedding_response = response.json()
        batch_embeddings = [item["embedding"] for item in embedding_response.get("data", [])]
        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings

"""
=================================
    CLI Functions
=================================
"""

def parse_args():
    parser = argparse.ArgumentParser(description='Run the Flask server.')
    parser.add_argument('--username', type=str, default='APS', help='Username for the API requests')
    parser.add_argument('--port', type=int, default=7285, help='Port number to run the server on')
    parser.add_argument('--dlog', action='store_true', help='Enable debug-level logging')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    debug_enabled = args.dlog
    logging.basicConfig(
        filename=ANL_DEBUG_FP, 
        level=logging.DEBUG if debug_enabled else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info(f'Starting server with debug mode: {debug_enabled}')
    print(f'Starting server... | Port {args.port} | User {args.username} | Debug: {debug_enabled}')
    app.run(host='localhost', port=args.port, debug=debug_enabled)