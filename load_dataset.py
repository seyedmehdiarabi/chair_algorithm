from datasets import load_dataset
import os
import json

# مسیر پروژه
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# بارگذاری دیتاست
print("در حال بارگذاری دیتاست PersianMedQA...")
dataset = load_dataset("MohammadJRanjbar/PersianMedQA")

# تبدیل به فرمت JSON سفارشی
def convert_to_qa_format(examples):
    """تبدیل داده به فرمت سوال-پاسخ"""
    formatted_data = []
    for i in range(len(examples['question'])):
        question = examples['question'][i]
        options = examples['options'][i]
        answer_idx = examples['answer_idx'][i]
        specialty = examples['specialty'][i]
        
        # ساخت پاسخ از گزینه صحیح
        answer = options[answer_idx] if answer_idx < len(options) else ""
        
        # ساخت سوال کامل (با گزینه‌ها)
        full_question = question + "\n" + "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(options)])
        
        formatted_data.append({
            "question": full_question,
            "answer": answer,
            "category": specialty
        })
    return formatted_data

# تبدیل هر بخش
train_data = convert_to_qa_format(dataset['train'])
val_data = convert_to_qa_format(dataset['validation'])
test_data = convert_to_qa_format(dataset['test'])

# ذخیره به صورت JSON
with open(os.path.join(DATA_DIR, "persianmedqa_train.json"), "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open(os.path.join(DATA_DIR, "persianmedqa_val.json"), "w", encoding="utf-8") as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)

with open(os.path.join(DATA_DIR, "persianmedqa_test.json"), "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

print(f"✅ دیتاست در پوشه {DATA_DIR} ذخیره شد")
print(f"تعداد نمونه‌های آموزش: {len(train_data)}")
print(f"تعداد نمونه‌های اعتبارسنجی: {len(val_data)}")
print(f"تعداد نمونه‌های تست: {len(test_data)}")

# نمونه از داده‌های ذخیره شده
print("\nنمونه اول:")
print(json.dumps(train_data[0], ensure_ascii=False, indent=2))