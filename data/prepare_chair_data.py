import json

input_file = "train.json"
output_file = "cahir_train.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

processed = []

for item in data:
    processed.append({
        "id": item.get("instance_id"),
        "category": item.get("Category", ""),
        "specialty": item.get("Specialty", ""),
        "age": item.get("Age", ""),
        "sex": item.get("Sex", ""),
        "source": item.get("dataset_source", ""),
        "question": item.get("Question", ""),
        "answer": item.get("Expert_Answer", ""),
        "question_type": item.get("QuestionTypeTag", {}).get("QuestionType_Tag", "")
    })

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(processed, f, ensure_ascii=False, indent=2)

print("Created:", output_file)
print("Samples:", len(processed))