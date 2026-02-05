import sys
import json

print("Checking environment...")
try:
    import requests
    print("Requests library found.")
except ImportError:
    print("Requests library NOT found.")
    sys.exit(1)

print("Checking Ollama connection...")
try:
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "gemma3:270m",
        "prompt": "Hello",
        "stream": False
    }
    response = requests.post(url, json=payload, timeout=10)
    if response.status_code == 200:
        print("Ollama connection successful.")
        print("Response:", response.json().get('response'))
    else:
        print(f"Ollama failed with status: {response.status_code}")
except Exception as e:
    print(f"Ollama connection failed: {e}")
