import json
from collections import Counter

file_path = "cahir_train.json"  # check this carefully

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("\n========== BASIC INFO ==========")
print("Type:", type(data))
print("Total samples:", len(data))

print("\n========== SAMPLE ==========")
print(data[0])

required_keys = ["id", "question", "answer", "category", "specialty"]

missing = []

for i, item in enumerate(data):
    for key in required_keys:
        if key not in item or item[key] in [None, "", {}]:
            missing.append((i, key))

print("\n========== MISSING FIELDS ==========")
print("Missing count:", len(missing))

empty_samples = 0

for item in data:
    if (
        not item.get("question", "").strip()
        or not item.get("answer", "").strip()
    ):
        empty_samples += 1

print("\n========== EMPTY CHECK ==========")
print("Empty samples:", empty_samples)

print("\n========== SAMPLE QA ==========")

for i in range(3):
    print("\n--- Sample", i, "---")
    print("Q:", data[i].get("question", ""))
    print("Context:", str(data[i].get("context", ""))[:120])
    print("A:", str(data[i].get("answer", ""))[:120])

cats = Counter([x.get("category", "NONE") for x in data])
spec = Counter([x.get("specialty", "NONE") for x in data])

print("\n========== CATEGORY DISTRIBUTION ==========")
print(cats.most_common(10))

print("\n========== SPECIALTY DISTRIBUTION ==========")
print(spec.most_common(10))

print("\n========== DONE ==========")