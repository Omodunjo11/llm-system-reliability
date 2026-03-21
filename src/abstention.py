def compute_confidence(context):
    # Simple heuristic: more context = higher confidence
    if not context:
        return 0.0
    
    return min(1.0, len(context) * 0.4)


def should_abstain(confidence, threshold=0.5):
    return confidence < threshold
    