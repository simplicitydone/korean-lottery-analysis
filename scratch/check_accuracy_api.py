import requests

url = "http://localhost:5000/api/accuracy?mode=LOTTO"
try:
    r = requests.get(url, timeout=5)
    print(r.status_code)
    print(r.json()[:1]) # Just check the first one
except Exception as e:
    print(e)
