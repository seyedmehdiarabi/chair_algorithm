import json
import sys
from pathlib import Path

def check_dataset(file_path):
    path = Path(file_path)
    if not path.exists():
        print(f"❌ File not found: {file_path}")
        print("Please check the path and file name.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and "data" in data:
        items = data["data"]
        found = False
        for article in items:
            title = article.get("title", "")
            if "توس" in title:
                print(f"✅ Found article: {title}")
                found = True
            for paragraph in article.get("paragraphs", []):
                context = paragraph.get("context", "")
                if "توس" in context:
                    print(f"✅ Found 'توس' in context: {context[:200]}...")
                    found = True
                for qas in paragraph.get("qas", []):
                    question = qas.get("question", "")
                    if "توس" in question:
                        print(f"✅ Found 'توس' in question: {question}")
                        found = True
        if not found:
            print("❌ No 'توس' found in the dataset!")
    else:
        print("❌ Unknown dataset format")
        print(f"First 500 chars: {str(data)[:500]}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "data/PQuAD-main/train.json"
        print(f"Using default path: {file_path}")
    check_dataset(file_path)