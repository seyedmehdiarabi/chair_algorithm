#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from colorama import init, Fore, Style

# فعال‌سازی رنگ در ترمینال
init(autoreset=True)

def print_header(text, char='='):
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{char*80}")
    print(f"{Fore.YELLOW}{Style.BRIGHT}{text.center(80)}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{char*80}{Style.RESET_ALL}\n")

def print_subheader(text):
    print(f"{Fore.MAGENTA}{Style.BRIGHT}{text}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{'-'*len(text)}{Style.RESET_ALL}")

def view_results(file_path=None):
    if file_path is None:
        results_dir = Path("results")
        if not results_dir.exists():
            print(f"{Fore.RED}❌ No results directory found!{Style.RESET_ALL}")
            return
        sessions = list(results_dir.rglob("session_*.json"))
        if not sessions:
            print(f"{Fore.RED}❌ No session files found!{Style.RESET_ALL}")
            return
        sessions.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        file_path = sessions[0]

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        info = data.get("session_info", {})
        print_header(f"📊 SESSION REPORT — {info.get('date', '')} {info.get('time', '')}")
        print(f"{Fore.WHITE}📅 Date: {Fore.GREEN}{info.get('date', 'N/A')}")
        print(f"{Fore.WHITE}⏰ Time: {Fore.GREEN}{info.get('time', 'N/A')}")
        print(f"{Fore.WHITE}📝 Total Queries: {Fore.GREEN}{len(data.get('queries', []))}")
        print(f"{Fore.WHITE}📁 File: {Fore.CYAN}{file_path}")

        for i, q_entry in enumerate(data.get('queries', []), 1):
            print_subheader(f"\n🔍 Query {i}: {q_entry['query']}")
            results = q_entry.get('results', {})
            
            # نمایش هر روش
            for method_name, method_results in results.items():
                if not method_results:
                    continue
                
                # انتخاب آیکون و رنگ
                emoji = {
                    'bm25': '📚', 'semantic': '🧠',
                    'hybrid_rrf': '🔄', 'hybrid_weighted': '⚖️'
                }.get(method_name, '📌')
                
                print(f"\n{Fore.YELLOW}{Style.BRIGHT}{emoji} {method_name.upper().replace('_', ' ')}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}{'─'*70}{Style.RESET_ALL}")
                
                for j, item in enumerate(method_results[:5], 1):
                    score = item.get('score', item.get('fusion_score', item.get('final_score', 0)))
                    if score > 0.7:
                        score_color = Fore.GREEN
                    elif score > 0.4:
                        score_color = Fore.YELLOW
                    else:
                        score_color = Fore.RED
                    
                    q = item.get('question', 'N/A')[:60]
                    a = item.get('answer', 'N/A')[:60]
                    cat = item.get('category', 'N/A')
                    
                    print(f"  {Fore.WHITE}{j}. {Fore.CYAN}Q:{Fore.WHITE} {q}...")
                    print(f"     {Fore.CYAN}A:{Fore.WHITE} {a}...")
                    print(f"     {Fore.CYAN}Cat:{Fore.WHITE} {cat}  {Fore.CYAN}Score:{score_color} {score:.4f}")
                    
                    # نمایش امتیازات اضافی
                    extra = []
                    if 'bm25_score' in item:
                        extra.append(f"BM25: {item['bm25_score']:.3f}")
                    if 'semantic_score' in item:
                        extra.append(f"Sem: {item['semantic_score']:.3f}")
                    if 'rerank_score' in item:
                        extra.append(f"Rerank: {item['rerank_score']:.3f}")
                    if extra:
                        print(f"     {Fore.MAGENTA}📈 {', '.join(extra)}")
                    print()

        print_header("✅ END OF REPORT")

    except FileNotFoundError:
        print(f"{Fore.RED}❌ File not found: {file_path}{Style.RESET_ALL}")
    except json.JSONDecodeError:
        print(f"{Fore.RED}❌ Invalid JSON file: {file_path}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")

def list_sessions():
    results_dir = Path("results")
    if not results_dir.exists():
        print(f"{Fore.RED}❌ No results directory found!{Style.RESET_ALL}")
        return
    sessions = list(results_dir.rglob("session_*.json"))
    if not sessions:
        print(f"{Fore.RED}❌ No session files found!{Style.RESET_ALL}")
        return
    sessions.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    print_header("📂 AVAILABLE SESSIONS")
    for i, file in enumerate(sessions, 1):
        try:
            with open(file, "r") as f:
                data = json.load(f)
                info = data.get('session_info', {})
                date = info.get('date', 'N/A')
                time = info.get('time', 'N/A')
                queries = len(data.get('queries', []))
                print(f"  {Fore.GREEN}{i:2d}.{Fore.WHITE} {date} {time}  —  {Fore.CYAN}{queries} queries{Fore.WHITE}  ({file.name})")
        except:
            print(f"  {Fore.RED}{i:2d}.{Fore.WHITE} {file.name} (corrupted)")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--list", "-l"]:
            list_sessions()
        else:
            view_results(sys.argv[1])
    else:
        view_results()