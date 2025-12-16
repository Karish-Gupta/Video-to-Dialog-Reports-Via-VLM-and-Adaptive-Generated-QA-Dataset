import json

def calculate_averages(data):
    totals = {
        "Factual Accuracy": 0,
        "Completeness": 0,
        "Visual Enrichment": 0,
        "Total Score": 0,
    }

    # Extract only the "score" dictionaries
    score_entries = []
    for key, item in data.items():
        if "score" in item:  # safety check
            score_entries.append(item["score"])

    count = len(score_entries)

    for score in score_entries:
        totals["Factual Accuracy"] += score.get("Factual Accuracy", 0)
        totals["Completeness"] += score.get("Completeness", 0)
        totals["Visual Enrichment"] += score.get("Visual Enrichment", 0)
        totals["Total Score"] += score.get("Total Score", 0)

    # Avoid division by zero
    if count == 0:
        return {k: 0 for k in totals}

    # Compute averages
    averages = {key: totals[key] / count for key in totals}
    return averages

