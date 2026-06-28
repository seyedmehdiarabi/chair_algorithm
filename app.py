import streamlit as st
import json
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from pathlib import Path
import numpy as np
import re

# ========== تنظیمات صفحه ==========
st.set_page_config(
    page_title="🔍 Intelligent Retrieval Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== استایل CSS سفارشی ==========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    * {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0 0.5rem 0;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    .query-box {
        background: linear-gradient(135deg, #f0f4ff 0%, #e8edf5 100%);
        padding: 1.2rem 2rem;
        border-radius: 16px;
        border-left: 6px solid #667eea;
        margin: 1rem 0 2rem 0;
        font-size: 1.2rem;
        font-weight: 500;
        color: #1a1a2e;
        box-shadow: 0 2px 12px rgba(102,126,234,0.15);
    }
    .best-answer-card {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin: 1rem 0 2rem 0;
        color: white;
        box-shadow: 0 4px 20px rgba(16,185,129,0.3);
    }
    .best-answer-card .label {
        font-size: 0.9rem;
        font-weight: 500;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .best-answer-card .answer {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .best-answer-card .meta {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .result-card {
        background: white;
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        margin: 0.8rem 0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border-left: 5px solid #667eea;
        transition: all 0.2s ease;
        height: 100%;
    }
    .result-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(102,126,234,0.18);
    }
    .result-card .rank {
        font-weight: 700;
        color: #667eea;
        font-size: 1.1rem;
    }
    .result-card .question {
        font-weight: 600;
        color: #1a1a2e;
        margin: 0.3rem 0;
    }
    .result-card .answer {
        color: #4a4a6a;
        font-size: 0.95rem;
        margin: 0.3rem 0;
    }
    .result-card .extracted-answer {
        background: #ecfdf5;
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        margin: 0.5rem 0;
        border-left: 4px solid #10b981;
        font-weight: 500;
        color: #065f46;
    }
    .result-card .score {
        font-weight: 700;
        margin-top: 0.5rem;
    }
    .score-high { color: #27ae60; }
    .score-medium { color: #f39c12; }
    .score-low { color: #e74c3c; }
    .category-tag {
        display: inline-block;
        background: #eef2ff;
        color: #4f46e5;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem 1rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(102,126,234,0.3);
    }
    .metric-box h3 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    .metric-box p {
        margin: 0.2rem 0 0;
        font-size: 0.85rem;
        opacity: 0.9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #f8fafc;
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        color: #475569;
        transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: #667eea !important;
        color: white !important;
        box-shadow: 0 2px 10px rgba(102,126,234,0.3);
    }
    .sidebar-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.8rem;
        margin-top: 3rem;
        border-top: 1px solid #e2e8f0;
        padding-top: 1.5rem;
    }
    @media (max-width: 640px) {
        .main-header { font-size: 2rem; }
        .query-box { font-size: 1rem; padding: 0.8rem 1.2rem; }
        .result-card { padding: 0.8rem 1rem; }
        .best-answer-card .answer { font-size: 1.5rem; }
    }
</style>
""", unsafe_allow_html=True)

# ========== توابع کمکی ==========
def load_sessions():
    results_dir = Path("results")
    if not results_dir.exists():
        return []
    sessions = []
    for file in results_dir.rglob("session_*.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                sessions.append({
                    "file": str(file),
                    "data": data,
                    "timestamp": data.get("session_info", {}).get("timestamp", ""),
                    "date": data.get("session_info", {}).get("date", ""),
                    "time": data.get("session_info", {}).get("time", ""),
                    "queries": len(data.get("queries", []))
                })
        except Exception:
            continue
    sessions.sort(key=lambda x: x["timestamp"], reverse=True)
    return sessions

def get_score(item):
    return item.get("score", item.get("fusion_score", item.get("final_score", 0)))

def get_score_class(score):
    if score > 0.7:
        return "score-high"
    elif score > 0.4:
        return "score-medium"
    else:
        return "score-low"

def truncate(text, length=80):
    if not text:
        return ""
    return text[:length] + ("..." if len(text) > length else "")

def create_comparison_chart(results_dict, query):
    methods = []
    scores = []
    # skip 'best_answer' if present
    for method_name, method_results in results_dict.items():
        if method_name == "best_answer":
            continue
        if method_results:
            avg_score = np.mean([get_score(r) for r in method_results[:5]])
            methods.append(method_name.upper().replace("_", " "))
            scores.append(avg_score)
    if not methods:
        return go.Figure()
    
    colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe']
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=methods,
        y=scores,
        text=[f"{s:.4f}" for s in scores],
        textposition='auto',
        marker_color=colors[:len(methods)],
        marker_line_color='rgba(0,0,0,0.08)',
        marker_line_width=1,
        opacity=0.9
    ))
    fig.update_layout(
        title=f"📊 Average Score per Method — '{query[:40]}'",
        xaxis_title="Method",
        yaxis_title="Average Score",
        height=400,
        template="plotly_white",
        font=dict(family="Inter, sans-serif", size=14),
        hovermode='x',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=60, b=40)
    )
    return fig

def create_distribution_chart(results_dict):
    all_scores = []
    for method_name, method_results in results_dict.items():
        if method_name == "best_answer":
            continue
        if method_results:
            scores = [get_score(r) for r in method_results[:10]]
            all_scores.extend(scores)
    if not all_scores:
        return go.Figure()
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=all_scores,
        nbinsx=20,
        marker_color='#667eea',
        opacity=0.8,
        name='Scores'
    ))
    fig.update_layout(
        title="📈 Score Distribution (Top-10 per method)",
        xaxis_title="Score",
        yaxis_title="Frequency",
        height=300,
        template="plotly_white",
        font=dict(family="Inter, sans-serif", size=14),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=50, b=30)
    )
    return fig

def display_results_cards(results, method_name, emoji):
    if not results:
        st.warning(f"No results for {method_name}")
        return
    cols = st.columns(3)
    for i, item in enumerate(results[:6]):
        with cols[i % 3]:
            score = get_score(item)
            score_class = get_score_class(score)
            category = item.get("category", "N/A")
            
            # نمایش پاسخ استخراج شده (اگر وجود داشته باشد)
            extracted = item.get("extracted_answer", "")
            qa_score = item.get("qa_score", None)
            
            st.markdown(f"""
            <div class="result-card">
                <div class="rank">#{i+1}</div>
                <div class="question">❓ {truncate(item.get('question', 'N/A'), 60)}</div>
                <div class="answer">💬 {truncate(item.get('answer', 'N/A'), 70)}</div>
                {f'<div class="extracted-answer">✅ <b>پاسخ دقیق:</b> {truncate(extracted, 80)}' + (f' (امتیاز QA: {qa_score:.3f})' if qa_score is not None else '') + '</div>' if extracted else ''}
                <div style="margin-top:0.5rem;">
                    <span class="category-tag">{category}</span>
                    <span class="score {score_class}" style="float:right;">Score: {score:.4f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ========== اپ اصلی ==========
def main():
    st.markdown('<div class="main-header">🔍 Intelligent Retrieval System</div>', unsafe_allow_html=True)

    # ===== سایدبار =====
    with st.sidebar:
        st.markdown('<div class="sidebar-header">📂 Session Selector</div>', unsafe_allow_html=True)
        sessions = load_sessions()
        if not sessions:
            st.warning("No sessions found. Please run `main.py` first!")
            return

        session_options = [f"{s['date']} {s['time']} — {s['queries']} queries" for s in sessions]
        selected_idx = st.selectbox(
            "Select Session",
            range(len(session_options)),
            format_func=lambda x: session_options[x]
        )
        selected_session = sessions[selected_idx]

        st.markdown("---")
        st.markdown("### 📊 Session Info")
        st.markdown(f"**Date:** {selected_session['date']}")
        st.markdown(f"**Time:** {selected_session['time']}")
        st.markdown(f"**Total Queries:** {selected_session['queries']}")
        st.markdown(f"**File:** `{os.path.basename(selected_session['file'])}`")

        if st.button("🔄 Refresh Sessions", use_container_width=True):
            st.rerun()

    # ===== محتوای اصلی =====
    data = selected_session['data']
    queries = data.get('queries', [])
    if not queries:
        st.warning("No queries in this session")
        return

    # انتخاب کوئری
    query_texts = [q['query'] for q in queries]
    selected_query_idx = st.selectbox(
        "🔍 Select Query",
        range(len(query_texts)),
        format_func=lambda x: f"Query {x+1}: {query_texts[x][:60]}..."
    )

    query_data = queries[selected_query_idx]
    query = query_data['query']
    results = query_data['results']

    # نمایش کوئری
    st.markdown(f'<div class="query-box">🔍 <b>Query:</b> {query}</div>', unsafe_allow_html=True)

    # ===== تب‌ها =====
    tabs = st.tabs(["📊 Overview", "📚 BM25", "🧠 Semantic", "🔄 Hybrid RRF", "⚖️ Weighted Hybrid"])

    with tabs[0]:
        # ---- نمایش بهترین پاسخ (Best Answer) ----
        best_answer = results.get("best_answer", [])
        if best_answer and isinstance(best_answer, list) and len(best_answer) > 0:
            best = best_answer[0]
            if best.get('answer') and best['answer'] != 'پاسخی یافت نشد.':
                st.markdown(f"""
                <div class="best-answer-card">
                    <div class="label">🏆 Best Answer</div>
                    <div class="answer">{best['answer']}</div>
                    <div class="meta">
                        Method: {best['method'].upper().replace('_', ' ')} &nbsp;|&nbsp;
                        Score: {best['score']:.4f} &nbsp;|&nbsp;
                        Category: {best['category']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ---- نمودارها ----
        col1, col2 = st.columns([2, 1])
        with col1:
            fig1 = create_comparison_chart(results, query)
            if fig1.data:
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.info("No data for comparison chart.")
        with col2:
            fig2 = create_distribution_chart(results)
            if fig2.data:
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No data for distribution chart.")

        # ---- متریک‌ها ----
        c1, c2, c3, c4 = st.columns(4)
        # محاسبه تعداد کل نتایج (به جز best_answer)
        total_results = sum(len(r) for method_name, r in results.items() if method_name != "best_answer")
        all_scores = [get_score(r) for method_name, method_results in results.items() 
                      if method_name != "best_answer" for r in method_results[:5]]
        avg_score = np.mean(all_scores) if all_scores else 0
        categories = set()
        for method_name, method_results in results.items():
            if method_name == "best_answer":
                continue
            for r in method_results[:5]:
                cat = r.get("category", "N/A")
                if cat and cat != "N/A":
                    categories.add(cat)
        best_method = max([m for m in results.keys() if m != "best_answer"], 
                          key=lambda x: np.mean([get_score(r) for r in results.get(x, [])[:5]]) if results.get(x) else 0)

        with c1:
            st.markdown(f"""
            <div class="metric-box">
                <h3>{total_results}</h3>
                <p>Total Results</p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-box">
                <h3>{avg_score:.3f}</h3>
                <p>Avg Score</p>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-box">
                <h3>{len(categories)}</h3>
                <p>Categories</p>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-box">
                <h3>{best_method.upper().replace('_', ' ')}</h3>
                <p>Best Method</p>
            </div>
            """, unsafe_allow_html=True)

        # ---- بخش پاسخ استخراج‌شده با QA (اگر موجود باشد) ----
        extracted_found = False
        for method_name, method_results in results.items():
            if method_name == "best_answer":
                continue
            for r in method_results[:5]:
                if r.get("extracted_answer"):
                    extracted_found = True
                    break
            if extracted_found:
                break
        
        if extracted_found:
            st.markdown("### ✅ Extracted Answer (QA)")
            best_extracted_method = None
            best_extracted_score = -1
            best_extracted_item = None
            for method_name, method_results in results.items():
                if method_name == "best_answer":
                    continue
                for r in method_results[:5]:
                    if r.get("extracted_answer") and r.get("qa_score", 0) > best_extracted_score:
                        best_extracted_score = r.get("qa_score", 0)
                        best_extracted_method = method_name
                        best_extracted_item = r
            if best_extracted_item:
                st.success(f"**{best_extracted_method.upper().replace('_', ' ')}** — پاسخ دقیق: **{best_extracted_item['extracted_answer']}** (امتیاز QA: {best_extracted_score:.3f})")
                with st.expander("مشاهده جزئیات بیشتر"):
                    st.json(best_extracted_item)

    # نمایش نتایج هر روش در تب‌های مربوطه
    method_emojis = {
        'bm25': '📚',
        'semantic': '🧠',
        'hybrid_rrf': '🔄',
        'hybrid_weighted': '⚖️'
    }
    for i, (key, emoji) in enumerate(method_emojis.items(), 1):
        with tabs[i]:
            if key in results:
                display_results_cards(results[key], key, emoji)
            else:
                st.warning(f"No results for {key.upper().replace('_', ' ')}")

    # دکمه دانلود
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        json_str = json.dumps(query_data, ensure_ascii=False, indent=2)
        st.download_button(
            label="📥 Download Results (JSON)",
            data=json_str,
            file_name=f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

    st.markdown('<div class="footer">🔍 Intelligent Retrieval System — Built with ❤️ using Streamlit</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()