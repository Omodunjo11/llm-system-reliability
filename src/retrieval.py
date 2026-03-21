import json

def load_docs():
    with open("data/sample_docs.json") as f:
        return json.load(f)

def simple_search(query, docs):
    results = []

    for doc in docs:
        if query.lower() in doc["text"].lower():
            results.append(doc)

    return results