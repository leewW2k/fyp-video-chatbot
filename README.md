## Microsoft Azure OpenAI Python Template

### Create a Virtual Environment

Run in the root folder:

```python -m venv .venv```

```.venv\Scripts\activate```

### Installing Required Packages

```pip install -r requirements.txt```

### Handling Environment Variables

Rename [.env-template](.env-template) to ```.env```

### Starting FastAPI Server

Run ```uvicorn main:app --port 8080 --reload```
