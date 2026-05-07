#!/usr/bin/env python3

import json
import sys
from collections import defaultdict


def analyze_samples(file_path):
    """Read JSONL file and count by companyId and subscriptionEventType"""
    counts = defaultdict(lambda: defaultdict(int))
    total_rows = 0

    try:
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    company_id = data.get("companyId", "UNKNOWN")
                    event_type = data.get("subscriptionEventType", "UNKNOWN")
                    counts[company_id][event_type] += 1
                    total_rows += 1
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {e}", file=sys.stderr)
                    continue

    except FileNotFoundError:
        print(f"File not found: {file_path}", file=sys.stderr)
        return

    # Print header
    print(f"{'companyId':<20} {'subscriptionEventType':<30} {'count':<10}")
    print("-" * 60)

    # Print data
    for company_id in sorted(counts.keys()):
        for event_type in sorted(counts[company_id].keys()):
            count = counts[company_id][event_type]
            print(f"{company_id:<20} {event_type:<30} {count:<10}")

    print("-" * 60)
    print(f"Total rows in file: {total_rows}")


if __name__ == "__main__":
    file_path = "./data/latest_samples.jsonl"
    analyze_samples(file_path)
