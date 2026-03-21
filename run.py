from src.retrieval import load_docs, simple_search
from src.generation import generate
from src.abstention import compute_confidence, should_abstain
from src.evaluation import evaluate

query = input("Enter your query: ")

docs = load_docs()
results = simple_search(query, docs)

confidence = compute_confidence(results)

print("\n=== SYSTEM DEBUG ===")
print("Retrieved docs:", results)
print("Confidence:", confidence)

if should_abstain(confidence):
    print("\n=== RESULT ===")
    print("ABSTAIN: Not enough reliable information to answer.")
else:
    answer = generate(query, results)
    score = evaluate(answer, results)

    print("\n=== ANSWER ===")
    print(answer)

    print("\n=== EVALUATION ===")
    print("Faithfulness score:", score)
    