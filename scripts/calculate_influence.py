import json
from collections import Counter

counter = Counter()

with open("outputs/Reputation_1/chief_outputs_medgemma.jsonl", "r") as f:
    for line in f:
        row = json.loads(line)
        
        influenced = row["chief"].get("which_panelists_influenced_me", [])
        
        # kan være én eller flere, så vi teller alle
        for gp in influenced:
            counter[gp] += 1

# se alle counts
print(counter)

# mest brukt
most_common = counter.most_common(1)[0]
print("Mest brukt:", most_common)

# minst brukt
least_common = min(counter.items(), key=lambda x: x[1])
print("Minst brukt:", least_common)