import os
import datetime
import requests
from flask import Flask, request, jsonify, Response
import time
import json
import logging
import argparse
from flask_cors import CORS  # Add this import
import httpx

app = Flask(__name__)
CORS(app, 
     origins="*",
     allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
     expose_headers=["Content-Type", "Authorization"],
     methods=["POST", "OPTIONS"],
     supports_credentials=True)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response


# Model names are different between OpenAI and Argo API
MODEL_MAPPING = {
    'gpt35': 'gpt35',
    'gpt-3.5': 'gpt35',

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

    'o1-mini': 'gpto1mini',
    'gpto1mini': 'gpto1mini',
    'o1mini': 'gpto1mini',

    'o3-mini': 'gpto3mini',
    'o3mini': 'gpto3mini',
    'gpto3mini': 'gpto3mini',
}

EMBEDDING_MODEL_MAPPING = {
    'text-embedding-3-small': 'v3small',
    'v3small': 'v3small',

    'text-embedding-3-large': 'v3large',
    'v3large': 'v3large',

    'text-embedding-ada-002': 'ada002',
    'ada002': 'ada002',
}

# URL mapping for different models
URL_MAPPING = {
    # Production URLs
    'prod': {
        'chat': 'https://apps.inside.anl.gov/argoapi/api/v1/resource/chat/',
        'embed': 'https://apps.inside.anl.gov/argoapi/api/v1/resource/embed/'
    },
    # Development URLs
    'dev': {
        'chat': 'https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/',
        'embed': 'https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/embed/'
    }
}

# Define which models use which environment
MODEL_ENV = {
    # Models using production environment
    'gpt35': 'prod',
    'gpt35large': 'prod',
    'gpt4': 'prod',
    'gpt4large': 'prod',
    'gpt4turbo': 'prod',
    'gpt4o': 'prod',
    'gpto1preview': 'prod',
    
    # Models using development environment
    'gpto3mini': 'dev',
    'gpto1mini': 'dev'
}

# Default embedding environment
EMBED_ENV = 'prod'

DEFAULT_MODEL = "gpt4o"
ANL_USER = "APS"
ANL_STREAM_URL = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/streamchat/"
ANL_DEBUG_FP = 'log_bridge.log'

def get_api_url(model, endpoint_type):
    """
    Determine the correct API URL based on model and endpoint type
    
    Args:
        model (str): The model identifier
        endpoint_type (str): Either 'chat' or 'embed'
    
    Returns:
        str: The appropriate API URL
    """
    # For embedding models, use the default embedding environment
    if model in EMBEDDING_MODEL_MAPPING.values():
        env = EMBED_ENV
    else:
        # For chat models, look up the environment or default to prod
        env = MODEL_ENV.get(model, 'prod')
    
    # Return the URL from the mapping
    return URL_MAPPING[env][endpoint_type]

"""
=================================
    Chat Endpoint
=================================
"""

@app.route('/chat/completions', methods=['POST'])
@app.route('/api/chat/completions', methods=['POST'])
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

    if is_streaming:
        return Response(_stream_chat_response(model, req_obj), mimetype='text/event-stream')
    else:
        response = requests.post(get_api_url(model, 'chat'), json=req_obj)
        if not response.ok:
            logging.error(f"Internal API error: {response.status_code} {response.reason}")
            return jsonify({"error": {
                "message": f"Internal API error: {response.status_code} {response.reason}"
            }}), 500

        json_response = response.json()
        text = json_response.get("response", "")
        logging.debug(f"Response Text {text}")
        return jsonify(_static_chat_response(text, model_base))

    
def _stream_chat_response(model, req_obj):
    begin_chunk = {
        "id": 'abc',
        "object": "chat.completion.chunk",
        "created": int(datetime.datetime.now().timestamp()),
        "model": model,
        "choices": [{
                    "index": 0,
                    "delta": {'role': 'assistant', 'content':''},
                    "logprobs": None,
                    "finish_reason": None
                }]                   
    }
    yield f"data: {json.dumps(begin_chunk)}\n\n"


    with httpx.stream("POST", ANL_STREAM_URL, json=req_obj, timeout=300.0)  as response:
        for chunk in response.iter_bytes():
            if chunk:
                text = chunk.decode(errors="replace")
                logging.debug(f'Text Chunk Recieved: {chunk}')
                response_chunk = {
                    "id": 'abc',
                    "object": "chat.completion.chunk",
                    "created": int(datetime.datetime.now().timestamp()),
                    "model": model,
                    "choices": [{
                                "index": 0,
                                "delta": {'content': text},
                                "logprobs": None,
                                "finish_reason": None
                            }]                   
                }
                yield f"data: {json.dumps(response_chunk)}\n\n"
    
    end_chunk = {
        "id": 'argo',
        "object": "chat.completion.chunk",
        "created": int(datetime.datetime.now().timestamp()),
        "model": model,
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [{
            "index": 0,
            "delta": {},
            "logprobs": None,
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(end_chunk)}\n\n"
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
@app.route('/v1/completions', methods=['POST', 'OPTIONS']) #LMStudio Compatibility
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

    response = requests.post(get_api_url(model, 'chat'), json=req_obj)
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
        
        response = requests.post(get_api_url(model, 'embed'), json=payload)
        
        if not response.ok:
            logging.error(f"Embedding API error: {response.status_code} {response.reason}")
            raise Exception(f"Embedding API error: {response.status_code} {response.reason}")
        
        embedding_response = response.json()
        batch_embeddings = embedding_response.get("embedding", [])
        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings

def check_argo_connection():
    """
    Check connection to Argo API endpoints and report status.
    
    Returns:
        bool: True if all connections successful, False otherwise
    """
    all_successful = True
    
    # Print for end-user
    print("\nTesting Argo API connections...")
    logging.info("Testing Argo API connections")
    
    for env in ['prod', 'dev']:
        for endpoint_type in ['chat', 'embed']:
            url = URL_MAPPING[env][endpoint_type]
            try:
                response = requests.head(url, timeout=5)
                msg = f"✓ Connection to {env} {endpoint_type} endpoint ({url}) available"
                print(msg)
                logging.info(msg)
            except requests.exceptions.RequestException as e:
                msg = f"✗ Connection to {env} {endpoint_type} endpoint ({url}) failed: {str(e)}"
                print(msg)
                logging.error(msg)
                all_successful = False
    
    if all_successful:
        print("✓ All Argo API connections successful!")
        logging.info("All Argo API connections successful")
    else:
        print("⚠ Some Argo API connections failed. The server may not function correctly.")
        logging.warning("Some Argo API connections failed")
    
    return all_successful


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

    check_argo_connection()

    app.run(host='localhost', port=args.port, debug=debug_enabled)
