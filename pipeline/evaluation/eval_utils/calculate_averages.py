import json

def calculate_averages(data):
    totals = {
        "Factual Accuracy": 0,
        "Completeness": 0,
        "Visual Enrichment": 0,
        "Clarity": 0,
        "Total Score": 0,
    }

    # Extract only the "score" dictionaries
    score_entries = []
    for key, item in data.items():
        if "score" in item:  # safety check
            score_entries.append(item["score"])

    count = len(score_entries)

    for score in score_entries:
        totals["Factual Accuracy"] += score["Factual Accuracy"]
        totals["Completeness"] += score["Completeness"]
        totals["Visual Enrichment"] += score["Visual Enrichment"]
        totals["Clarity"] += score["Clarity"]
        totals["Total Score"] += score["Total Score"]

    # Compute averages
    averages = {key: totals[key] / count for key in totals}
    return averages

