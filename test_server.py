import unittest
from unittest.mock import patch, MagicMock
import json
from io import BytesIO

# Import the server module with absolute import
from argo_bridge import (
    app, parse_args, MODEL_MAPPING, EMBEDDING_MODEL_MAPPING, DEFAULT_MODEL,
    _static_chat_response, _stream_chat_response,
    _static_completions_response, _stream_completions_response,
    _get_embeddings_from_argo
)


class TestServerConfig(unittest.TestCase):
    """Test the server configuration and constants"""
    
    def test_model_mappings(self):
        """Test that model mappings are correctly defined"""
        self.assertEqual(MODEL_MAPPING['gpt-4o'], 'gpt4o')
        self.assertEqual(MODEL_MAPPING['gpt35'], 'gpt35')
        self.assertEqual(EMBEDDING_MODEL_MAPPING['text-embedding-3-small'], 'v3small')
    
    def test_default_model(self):
        """Test the default model is set correctly"""
        self.assertEqual(DEFAULT_MODEL, "gpt4o")


class TestChatEndpoint(unittest.TestCase):
    """Test the chat completions endpoint"""
    
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
    
    @patch('argo_bridge.requests.post')
    def test_chat_completions_success(self, mock_post):
        """Test successful chat completion request"""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"response": "This is a test response"}
        mock_post.return_value = mock_response
        
        # Create test request
        test_data = {
            "model": "gpt4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.5
        }
        
        # Send request
        response = self.app.post('/chat/completions', 
                                 data=json.dumps(test_data),
                                 content_type='application/json')
        
        # Assert response
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["choices"][0]["message"]["content"], "This is a test response")
        
        # Verify API was called correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args[1]["json"]
        self.assertEqual(call_args["model"], "gpt4o")
        self.assertEqual(call_args["messages"], [{"role": "user", "content": "Hello"}])
        self.assertEqual(call_args["temperature"], 0.5)
    
    @patch('argo_bridge.requests.post')
    def test_chat_unsupported_model(self, mock_post):
        """Test chat completion with unsupported model"""
        test_data = {
            "model": "unsupported-model",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        response = self.app.post('/chat/completions', 
                                 data=json.dumps(test_data),
                                 content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn("not supported", data["error"]["message"])
        mock_post.assert_not_called()
    
    @patch('argo_bridge.requests.post')
    def test_chat_api_failure(self, mock_post):
        """Test handling of API failures in chat endpoint"""
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 500
        mock_response.reason = "Internal Server Error"
        mock_post.return_value = mock_response
        
        test_data = {
            "model": "gpt4o",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        response = self.app.post('/chat/completions', 
                                 data=json.dumps(test_data),
                                 content_type='application/json')
        
        self.assertEqual(response.status_code, 500)
        data = json.loads(response.data)
        self.assertIn("Internal API error", data["error"]["message"])
    
    @patch('argo_bridge.httpx.stream')
    def test_chat_streaming(self, mock_stream):
        """Test streaming response from chat endpoint"""
        # Mock the context manager that httpx.stream returns
        mock_response = MagicMock()
        
        # Create mock chunks that will be yielded
        mock_chunks = [b"First chunk", b"Second chunk"]
        mock_response.iter_bytes.return_value = mock_chunks
        
        # Set up the context manager mock to return our response mock
        mock_stream_cm = MagicMock()
        mock_stream_cm.__enter__.return_value = mock_response
        mock_stream.return_value = mock_stream_cm
        
        test_data = {
            "model": "gpt4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True
        }
        
        response = self.app.post('/chat/completions',
                                data=json.dumps(test_data),
                                content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type.split(';')[0], 'text/event-stream')
        
        # Read response data
        data = response.data.decode('utf-8')
        
        # Check for the beginning chunk
        self.assertIn('data: {"id": "abc", "object": "chat.completion.chunk"', data)
        self.assertIn('"delta": {"role": "assistant", "content": ""}', data)
        
        # Check for the content chunks
        for chunk_content in mock_chunks:
            self.assertIn(f'"delta": {{"content": "{chunk_content.decode()}', data)
        
        # Check for the ending chunk
        self.assertIn('"delta": {}', data)
        self.assertIn('"finish_reason": "stop"', data)
        
        # Check for the [DONE] marker
        self.assertIn('data: [DONE]', data)
        
        # Verify the API was called correctly
        mock_stream.assert_called_once()
        call_args = mock_stream.call_args
        self.assertEqual(call_args[0][0], 'POST')
        self.assertEqual(call_args[0][1], 'https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/streamchat/')
        self.assertEqual(call_args[1]['json']['model'], 'gpt4o')
        self.assertEqual(call_args[1]['json']['messages'], [{"role": "user", "content": "Hello"}])

class TestCompletionsEndpoint(unittest.TestCase):
    """Test the completions endpoint"""
    
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
    
    @patch('argo_bridge.requests.post')
    def test_completions_success(self, mock_post):
        """Test successful completions request"""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"response": "This is a completion"}
        mock_post.return_value = mock_response
        
        test_data = {
            "model": "gpt4o",
            "prompt": "Complete this sentence",
            "temperature": 0.3
        }
        
        response = self.app.post('/completions', 
                                 data=json.dumps(test_data),
                                 content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["choices"][0]["text"], "This is a completion")
        
        # Verify API was called correctly
        call_args = mock_post.call_args[1]["json"]
        self.assertEqual(call_args["model"], "gpt4o")
        self.assertEqual(call_args["prompt"], ["Complete this sentence"])


class TestEmbeddingsEndpoint(unittest.TestCase):
    """Test the embeddings endpoint"""
    
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
    
    @patch('argo_bridge._get_embeddings_from_argo')
    def test_embeddings_success(self, mock_get_embeddings):
        """Test successful embeddings request"""
        # Mock embedding vectors
        mock_get_embeddings.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        test_data = {
            "model": "text-embedding-3-small",
            "input": ["Test text 1", "Test text 2"]
        }
        
        response = self.app.post('/embeddings', 
                                 data=json.dumps(test_data),
                                 content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(len(data["data"]), 2)
        self.assertEqual(data["data"][0]["embedding"], [0.1, 0.2, 0.3])
        self.assertEqual(data["data"][1]["embedding"], [0.4, 0.5, 0.6])
    
    def test_embeddings_unsupported_model(self):
        """Test embeddings with unsupported model"""
        test_data = {
            "model": "unsupported-embedding-model",
            "input": ["Test text"]
        }
        
        response = self.app.post('/embeddings', 
                                 data=json.dumps(test_data),
                                 content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn("not supported", data["error"]["message"])
    
    @patch('argo_bridge.requests.post')
    def test_get_embeddings_from_argo(self, mock_post):
        """Test the _get_embeddings_from_argo helper function"""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "embedding": [
                 [0.1, 0.2, 0.3],
                 [0.4, 0.5, 0.6]
            ]
        }
        mock_post.return_value = mock_response
        
        result = _get_embeddings_from_argo(["Test1", "Test2"], "v3small")
        
        self.assertEqual(result, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_post.assert_called_once()


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions separately"""
    
    def test_static_chat_response(self):
        """Test _static_chat_response function"""
        result = _static_chat_response("Test response", "gpt4")
        
        self.assertEqual(result["object"], "chat.completion")
        self.assertEqual(result["model"], "gpt4")
        self.assertEqual(result["choices"][0]["message"]["content"], "Test response")
        self.assertEqual(result["choices"][0]["finish_reason"], "stop")
    
    def test_static_completions_response(self):
        """Test _static_completions_response function"""
        result = _static_completions_response("Test completion", "gpt35")
        
        self.assertEqual(result["object"], "text_completion")
        self.assertEqual(result["model"], "gpt35")
        self.assertEqual(result["choices"][0]["text"], "Test completion")
    
    def test_parse_args(self):
        """Test argument parsing function"""
        with patch('sys.argv', ['argo_bridge.py', '--username', 'testuser', '--port', '8080']):
            args = parse_args()
            self.assertEqual(args.username, 'testuser')
            self.assertEqual(args.port, 8080)
            self.assertFalse(args.dlog)


if __name__ == '__main__':
    unittest.main()