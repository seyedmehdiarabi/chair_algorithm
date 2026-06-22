import json
file_path = "train.json"

with open(
    file_path,
    "r",
    encoding="utf-8"
)as f:
    data = json.load(f)

print("Type:" , type(data))

print("Legth:" , len(data))

print("\n First item:")
print(data[0])