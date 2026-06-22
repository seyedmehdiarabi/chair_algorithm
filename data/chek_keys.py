import json


with open(
    "train.json",
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


print("Number of samples:", len(data))

print("\nFirst item keys:")

print(data[0].keys())


print("\nFirst item full:")

print(data[0])