import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from typing import Optional, Tuple, List, Dict, Any
import logging
from utils.error_handler import handle_errors, ModelError

logger = logging.getLogger(__name__)


class PersianQAE:
    """
    Extractive Question Answering برای زبان فارسی با استفاده از مدل
    m3hrdadfi/bert-fa-base-uncased-squad (fine-tuned روی PQuAD و PersianQA)
    """

    def __init__(
        self,
        model_name: str = "m3hrdadfi/bert-fa-base-uncased-squad",
        device: Optional[str] = None,
        max_length: int = 512,
        stride: int = 128,
        batch_size: int = 16
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.stride = stride
        self.batch_size = batch_size

        try:
            logger.info(f"Loading QA model: {model_name} on {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            self._available = True
            logger.info("✅ QA model loaded successfully.")
        except Exception as e:
            logger.warning(f"⚠️ QA model could not be loaded: {e}")
            logger.warning("QA features will be disabled. Use --no-qa to suppress this warning.")
            self._available = False
            self.tokenizer = None
            self.model = None

    def is_available(self) -> bool:
        return self._available

    @handle_errors
    def extract_answer(self, question: str, context: str) -> Tuple[str, float]:
        if not self._available:
            return "", 0.0
        if not question or not context:
            return "", 0.0

        inputs = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation="only_second",
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # اگر چندین پنجره وجود داشته باشد، بهترین را انتخاب می‌کنیم
        # (برای سادگی، از پنجره اول استفاده می‌کنیم)
        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        best_start = torch.argmax(start_logits)
        best_end = torch.argmax(end_logits)

        if best_start > best_end:
            best_end = best_start + 1

        input_ids = inputs["input_ids"][0]
        answer_tokens = input_ids[best_start:best_end + 1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)

        score = (torch.max(start_logits).item() + torch.max(end_logits).item()) / 2

        return answer.strip(), float(score)

    @handle_errors
    def extract_from_candidates(self, question: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not self._available or not candidates:
            return candidates

        candidates = candidates[:top_k]
        for item in candidates:
            context = item.get('context', '')
            if not context:
                # اگر context موجود نبود، از ترکیب سوال و پاسخ استفاده می‌کنیم (اما دقیق نیست)
                context = item.get('question', '') + " " + item.get('answer', '')
            if context:
                answer, score = self.extract_answer(question, context)
                item['extracted_answer'] = answer
                item['qa_score'] = score
            else:
                item['extracted_answer'] = ""
                item['qa_score'] = 0.0

        candidates.sort(key=lambda x: x.get('qa_score', 0), reverse=True)
        return candidates


# نمونه برای استفاده (در صورت موفقیت بارگذاری)
try:
    default_qa = PersianQAE()
except Exception as e:
    import logging
    logging.warning(f"QA model could not be loaded: {e}. QA features will be disabled.")
    default_qa = None