import json
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional
from utils.error_handler import DataError, raise_errors

logger = logging.getLogger(__name__)

class DatasetLoader:
    """Load datasets from various formats with automatic field detection"""

    SUPPORTED_FORMATS = ['.json', '.csv', '.xlsx', '.xls', '.parquet', '.tsv']

    @classmethod
    @raise_errors
    def load(cls, file_path: str, **kwargs) -> List[Dict[str, Any]]:
        """Load dataset from file"""
        path = Path(file_path)

        if not path.exists():
            raise DataError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in cls.SUPPORTED_FORMATS:
            raise DataError(f"Unsupported format: {ext}. Supported: {cls.SUPPORTED_FORMATS}")

        logger.info(f"Loading dataset from {file_path}")

        if ext == '.json':
            return cls._load_json(path, **kwargs)
        elif ext in ['.csv', '.tsv']:
            return cls._load_csv(path, **kwargs)
        elif ext in ['.xlsx', '.xls']:
            return cls._load_excel(path, **kwargs)
        elif ext == '.parquet':
            return cls._load_parquet(path, **kwargs)
        else:
            raise DataError(f"Unhandled format: {ext}")

    @staticmethod
    def _load_json(path: Path, **kwargs) -> List[Dict]:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # If data is a dict, try to find the actual list
        if isinstance(data, dict):
            for key in ['data', 'items', 'documents', 'results', 'records']:
                if key in data:
                    data = data[key]
                    break
            else:
                # Convert dict to list of items
                data = [{'key': k, 'value': v} for k, v in data.items()]

        if not isinstance(data, list):
            raise DataError("JSON root must be a list or object containing a list")

        return DatasetLoader._normalize_fields(data)

    @staticmethod
    def _load_csv(path: Path, **kwargs) -> List[Dict]:
        sep = kwargs.get('sep', '\t' if path.suffix == '.tsv' else ',')
        df = pd.read_csv(path, sep=sep, encoding='utf-8')
        return DatasetLoader._df_to_list(df)

    @staticmethod
    def _load_excel(path: Path, **kwargs) -> List[Dict]:
        df = pd.read_excel(path, engine='openpyxl')
        return DatasetLoader._df_to_list(df)

    @staticmethod
    def _load_parquet(path: Path, **kwargs) -> List[Dict]:
        df = pd.read_parquet(path)
        return DatasetLoader._df_to_list(df)

    @staticmethod
    def _df_to_list(df: pd.DataFrame) -> List[Dict]:
        """Convert DataFrame to list of dicts with standardized fields"""
        columns = df.columns.tolist()

        question_col = None
        answer_col = None
        category_col = None

        for col in columns:
            col_lower = col.lower().strip()
            if any(k in col_lower for k in ['question', 'query', 'text', 'title', 'content']):
                if question_col is None:
                    question_col = col
            elif any(k in col_lower for k in ['answer', 'response', 'output', 'label']):
                if answer_col is None:
                    answer_col = col
            elif any(k in col_lower for k in ['category', 'class', 'type', 'label']):
                if category_col is None:
                    category_col = col

        if question_col is None and len(columns) > 0:
            question_col = columns[0]
        if answer_col is None and len(columns) > 1:
            answer_col = columns[1]

        results = []
        for _, row in df.iterrows():
            item = {}
            if question_col:
                item['question'] = str(row[question_col]) if pd.notna(row[question_col]) else ""
            if answer_col:
                item['answer'] = str(row[answer_col]) if pd.notna(row[answer_col]) else ""
            if category_col:
                item['category'] = str(row[category_col]) if pd.notna(row[category_col]) else ""

            for col in columns:
                if col not in [question_col, answer_col, category_col]:
                    val = row[col]
                    if pd.notna(val):
                        item[col] = str(val) if not isinstance(val, (int, float, bool)) else val
            results.append(item)

        return results

    @staticmethod
    def _normalize_fields(data: List[Dict]) -> List[Dict]:
        """
        Normalize field names: ensure 'question', 'answer', 'category' exist.
        Supports PQuAD structure (with 'paragraphs' and 'qas') automatically.
        """
        normalized = []

        for item in data:
            # Check for PQuAD structure: contains 'paragraphs' and optionally 'title'
            if "paragraphs" in item and isinstance(item["paragraphs"], list):
                title = item.get("title", "")
                for paragraph in item.get("paragraphs", []):
                    context = paragraph.get("context", "")
                    for qas in paragraph.get("qas", []):
                        new_item = {}
                        new_item["question"] = qas.get("question", "")
                        new_item["context"] = context
                        new_item["title"] = title

                        # ✅ استخراج صحیح پاسخ از answers
                        answers = qas.get("answers", [])
                        if answers and len(answers) > 0:
                            # answers[0] یک دیکشنری با کلید 'text' است
                            new_item["answer"] = answers[0].get("text", "").strip()
                        else:
                            new_item["answer"] = ""

                        new_item["category"] = title
                        new_item["id"] = qas.get("id", "")
                        new_item["is_impossible"] = qas.get("is_impossible", False)
                        normalized.append(new_item)
                continue  # پایان پردازش این آیتم PQuAD

            # --- Fallback: standard item (non-PQuAD) ---
            new_item = {}
            for key, value in item.items():
                key_lower = key.lower().strip()
                if any(k in key_lower for k in ['question', 'query', 'text', 'title']):
                    new_item['question'] = str(value) if value else ""
                elif any(k in key_lower for k in ['answer', 'response', 'output']):
                    new_item['answer'] = str(value) if value else ""
                elif any(k in key_lower for k in ['category', 'class', 'label']):
                    new_item['category'] = str(value) if value else ""
                else:
                    new_item[key] = value

            if 'question' not in new_item:
                new_item['question'] = str(item.get('content', item.get('text', '')))
            if 'answer' not in new_item:
                new_item['answer'] = str(item.get('response', item.get('output', '')))
            if 'category' not in new_item:
                new_item['category'] = str(item.get('type', item.get('label', 'general')))

            normalized.append(new_item)

        return normalized