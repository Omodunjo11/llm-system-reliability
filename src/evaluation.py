def evaluate(answer, context):
    if not context:
        return 0.0

    context_text = " ".join([c["text"] for c in context])

    answer_words = answer.lower().split()
    context_words = context_text.lower().split()

    overlap = sum(1 for word in answer_words if word in context_words)

    score = overlap / max(len(answer_words), 1)

    return round(score, 2)
    