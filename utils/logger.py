import json
import os
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ResultLogger:
    """Log search results to JSON and text files"""
    
    def __init__(self, output_dir="results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session file
        now = datetime.now()
        self.session_id = now.strftime("%Y%m%d_%H%M%S")
        self.session_file = self.output_dir / f"session_{self.session_id}.json"
        self.text_file = self.output_dir / f"session_{self.session_id}.txt"
        
        self.session_data = {
            "session_info": {
                "timestamp": now.isoformat(),
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "session_id": self.session_id
            },
            "queries": []
        }
        
        logger.info(f"Result logger initialized: {self.session_file}")
    
    def log_query(self, query: str, results: dict):
        """Log a single query and its results"""
        entry = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "results": {}
        }
        
        # Convert any numpy/torch types to Python native
        def convert_to_serializable(obj):
            if hasattr(obj, 'item'):  # numpy/torch scalar
                return obj.item()
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            return obj
        
        for method, res_list in results.items():
            entry["results"][method] = convert_to_serializable(res_list)
        
        self.session_data["queries"].append(entry)
        self._save()
    
    def _save(self):
        """Save session data to JSON and generate text report"""
        with open(self.session_file, 'w', encoding='utf-8') as f:
            json.dump(self.session_data, f, ensure_ascii=False, indent=2)
        
        # Generate text report
        self._save_text_report()
    
    def _save_text_report(self):
        """Generate human-readable text report"""
        with open(self.text_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("SEARCH RESULTS REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Session: {self.session_id}\n")
            f.write(f"Date: {self.session_data['session_info']['date']}\n")
            f.write(f"Time: {self.session_data['session_info']['time']}\n")
            f.write(f"Total queries: {len(self.session_data['queries'])}\n\n")
            
            for i, q_entry in enumerate(self.session_data['queries'], 1):
                f.write(f"\n--- Query {i}: {q_entry['query']} ---\n")
                for method, results in q_entry['results'].items():
                    f.write(f"\n  Method: {method.upper()}\n")
                    for j, r in enumerate(results[:5], 1):
                        score = r.get('score', r.get('fusion_score', r.get('final_score', 0)))
                        f.write(f"    {j}. Score: {score:.4f} | Q: {r.get('question', '')[:60]}...\n")

_result_logger = None

def get_result_logger():
    global _result_logger
    if _result_logger is None:
        _result_logger = ResultLogger()
    return _result_logger