# Continue

URL: https://github.com/continuedev/continue

1. Click 3 dot menu in the top left. 
2. Under Models click add model
3. In the config.json merge this into the models and embeddings keys: 

```json
 "models": [
    {
      "model": "gpt4o",
      "title": "local",
      "apiKey": "aps",
      "provider": "openai",
      "apiBase": "http://localhost:7285/"
    }
  ],
   "embeddingsProvider": {
      "provider": "openai",
      "model": "text-embedding-3-large", 
      "apiBase": "http://localhost:7285/",
      "apiKey": "aps"
  },
```

# Cline

URL: https://github.com/cline/cline

1. Click the cog wheel in the top right
2. Under each mode select "LM Studio" as the API provider
3. Set the base url to http://localhost:7285
4. Enter your preferred model into the Model ID (o3-mini is a decent starting point)

# Open-Webui

URL: https://github.com/open-webui/open-webui

1. Once set up, click the user profile in the top right
2. Navigate to the "Connections" in user settings
3. Click the + to add a new connection
4. In the URL put in `http://localhost:7285`
5. In key put in APS
6. Leave prefix ID blank
7. In model IDs add any model ids you wish to use.