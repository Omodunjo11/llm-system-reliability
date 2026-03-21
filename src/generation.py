def generate(query, context):
    if not context:
        return "No relevant information found."

    combined_context = " ".join([c["text"] for c in context])

    return f"Answer based on context:\n{combined_context}"
    