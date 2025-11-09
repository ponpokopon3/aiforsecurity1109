#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Streamlit frontend for main2.py (cleaned header)

from pathlib import Path
import streamlit as st
import json
import re

try:
    import main as backend
except Exception as e:
    st.error("バックエンド main2.py の読み込みに失敗しました。先に main2.py が正しく動くか確認してください。")
    st.exception(e)
    raise

st.set_page_config(page_title="AgentRAG - Web UI", layout="wide")

# タイトルをコンパクトに表示
st.markdown("# 🛡️ AgentRAG — セキュリティ規則チェッカー")
st.markdown("*統一基準対応 RAGチャットボット (Streamlit UI)*")
st.markdown("---")  # 区切り線

# 小さめフォントとコンパクト表示のための簡易 CSS + スクロール改善 + タイトル修正
st.markdown(
    """
    <style>
    * { font-size:13px !important; }
    .stButton>button { padding:4px 8px !important; font-size:13px !important; }
    textarea { font-size:12px !important; }
    
    /* タイトルエリアの修正 */
    .main .block-container {
        padding-top: 3rem !important;  /* タイトル用のスペースを確保 */
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-bottom: 5rem !important;
        max-width: none !important;
        overflow-y: visible !important;
    }
    
    /* タイトル（h1）の表示改善 */
    .main h1 {
        margin-top: 0 !important;
        margin-bottom: 0.5rem !important;
        padding-top: 0 !important;
        font-size: 1.8rem !important;  /* フォントサイズを少し小さく */
        line-height: 1.2 !important;
    }
    
    /* サブタイトル（em）のスタイル */
    .main em {
        font-size: 0.9rem !important;
        color: #666 !important;
        display: block !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* 区切り線の調整 */
    .main hr {
        margin: 0.5rem 0 1rem 0 !important;
    }
    
    /* 確実なスクロール設定 */
    html, body, #root {
        overflow-y: auto !important;
        height: 100% !important;
    }
    
    .main {
        overflow-y: auto !important;
        height: 100vh !important;
        padding-top: 0 !important;  /* メインエリアの上部パディングをリセット */
    }
    
    /* selectboxのドロップダウンリスト改善 */
    .stSelectbox div[data-baseweb="select"] > div {
        max-height: 300px !important; 
        overflow-y: auto !important;
    }
    
    /* selectbox の選択肢リストのスクロール */
    div[data-baseweb="popover"] {
        max-height: 400px !important;
        overflow-y: auto !important;
    }
    
    /* ヘッダー部分の調整 */
    header[data-testid="stHeader"] {
        height: 2.5rem !important;  /* ヘッダー高さを調整 */
    }
    
    /* サイドバーとの間隔調整 */
    .css-1d391kg {  /* サイドバーのクラス */
        padding-top: 1rem !important;
    }
    
    /* レスポンシブ対応 */
    @media (max-width: 768px) {
        .main h1 {
            font-size: 1.5rem !important;
        }
        .main .block-container {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
    }
    
    /* 小さな画面での調整 */
    @media (max-width: 480px) {
        .main h1 {
            font-size: 1.3rem !important;
        }
        * {
            font-size: 12px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# キャッシュ付き初期化
@st.cache_resource
def get_vectordb():
    docs = backend.load_spec_documents(backend.SPEC_DIR)
    return backend.init_chroma(docs)

@st.cache_resource
def get_rules():
    return backend.load_rules_from_dir(backend.RULE_DIR)

@st.cache_resource
def get_llm():
    return backend.make_chat_model()

vectordb = None
llm = None
rules = []
try:
    vectordb = get_vectordb()
    llm = get_llm()
    rules = get_rules()
except Exception as e:
    st.warning("ベクトルDB や LLM の初期化で警告が出ました。OpenAIキーや依存が正しく設定されているか確認してください。")
    st.exception(e)

# サイドバー: ページ切替と共通オプション
with st.sidebar:
    st.markdown("### 📋 メニュー")
    st.write(f"**ルール数**: {len(rules):,}件")
    
    page = st.radio("ページ選択", ["🔍 ルールチェック", "💬 RAG 質問"], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### ⚙️ 設定")
    topk = st.slider("参照ドキュメント数", 1, 10, backend.TOP_K, help="RAG検索で参照するドキュメントの数")
    
    st.markdown("---")
    st.markdown("### 📁 対応ファイル形式")
    st.markdown("""
    **ドキュメント読み込み対応:**
    - 📄 PDF (.pdf)
    - 📝 Word (.docx)
    - 📊 Excel (.xlsx)
    - 📈 PowerPoint (.pptx)
    - 📋 Markdown (.md)
    - 📄 Text (.txt)
    
    **ルール定義:**
    - 📋 JSON (.json)
    """)
    
    st.markdown("---")
    st.caption("💡 `specification/` フォルダにファイルを配置してください")

# ルール一覧を取得（プレビュー付き選択肢を作成）
def create_rule_preview(rule):
    """ルールの選択肢用プレビューテキストを作成"""
    rule_id = rule.get('id', '')
    title = rule.get('title', '')
    rule_type = rule.get('type', '')
    content = rule.get('content', '')
    
    # 内容の先頭部分を取得（改行削除、短めに）
    preview_content = content.replace('\n', ' ').replace('\r', '').strip()
    if len(preview_content) > 50:  # 80文字から50文字に短縮
        preview_content = preview_content[:50] + "..."
    
    # 選択肢テキストを構築（よりコンパクトに）
    choice_text = f"{rule_id}"
    if rule_type:
        choice_text += f" [{rule_type}]"
    
    # タイトルがIDと違う場合のみ追加
    if title and title != rule_id:
        # タイトルも短縮
        short_title = title[:30] + "..." if len(title) > 30 else title
        choice_text += f" {short_title}"
    
    # 内容プレビュー
    if preview_content:
        choice_text += f" | {preview_content}"
    
    return choice_text

rule_choices = {}
for r in rules:
    preview_text = create_rule_preview(r)
    rule_choices[preview_text] = r

if "ルールチェック" in page:
    st.header("🔍 ルールチェック")
    
    # 検索機能を追加
    col1, col2 = st.columns([3, 1])
    with col1:
        search_term = st.text_input("ルール検索（ID、種別、内容で検索）", placeholder="例: 責任者, 遵守事項, 2.1.1")
    with col2:
        st.write("") # 空白でレイアウト調整
        show_all = st.checkbox("全件表示", help="チェックすると検索結果の全件を表示します（重い場合があります）")
    
    # 表示数の設定
    max_display_items = 500 if show_all else 100
    
    # 検索でフィルタリング（表示数制限を追加）
    filtered_choices = {}
    
    if search_term:
        count = 0
        for preview_text, rule in rule_choices.items():
            if count >= max_display_items:
                break
            if (search_term.lower() in preview_text.lower() or
                search_term.lower() in rule.get('content', '').lower() or
                search_term.lower() in rule.get('id', '').lower() or
                search_term.lower() in rule.get('type', '').lower()):
                filtered_choices[preview_text] = rule
                count += 1
        
        if filtered_choices:
            total_matches = sum(1 for preview_text, rule in rule_choices.items() 
                              if (search_term.lower() in preview_text.lower() or
                                  search_term.lower() in rule.get('content', '').lower() or
                                  search_term.lower() in rule.get('id', '').lower() or
                                  search_term.lower() in rule.get('type', '').lower()))
            
            if total_matches > max_display_items:
                st.info(f"🔍 検索結果: {total_matches}件中 上位{len(filtered_choices)}件を表示")
                if not show_all:
                    st.caption("より多く表示するには「全件表示」をチェックするか、検索語を具体化してください")
            else:
                st.success(f"🔍 検索結果: {len(filtered_choices)}件のルールが見つかりました")
        else:
            st.warning("🔍 検索条件に一致するルールが見つかりませんでした")
    else:
        # 検索なしの場合は最初のN件のみ表示
        count = 0
        for preview_text, rule in rule_choices.items():
            if count >= max_display_items:
                break
            filtered_choices[preview_text] = rule
            count += 1
        
        if len(rule_choices) > max_display_items:
            st.info(f"📋 全{len(rule_choices)}件中 上位{max_display_items}件を表示")
            st.caption("検索機能または「全件表示」チェックで他のルールも表示できます")
    
    # ルール選択（スクロール可能にするためにコンテナで囲む）
    with st.container():
        choices = ["(選択してください)"] + list(filtered_choices.keys())
        sel = st.selectbox(
            "評価するルールを選択", 
            choices, 
            help="ルールID、種別、内容のプレビューが表示されます",
            key="rule_selector"
        )
    
    st.caption("⚙️ ルールを選択して 'チェック実行' を押すと評価が始まります。")

    if sel == "(選択してください)":
        if search_term:
            st.info("上記の検索結果からルールを選択してください")
        else:
            st.info("ルールを選択してください（上部の検索ボックスで絞り込み可能）")
    else:
        r = filtered_choices[sel]
        
        # ルール情報の表示（コンパクトに整理）
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"**ID**: `{r.get('id')}`")
            if r.get('type'):
                st.markdown(f"**種別**: {r.get('type')}")
            if r.get('source_file'):
                st.markdown(f"**ソース**: {r.get('source_file')}")
        
        with col2:
            st.markdown(f"**タイトル**: {r.get('title')}")
            if r.get('path'):
                st.markdown(f"**階層**: `{r.get('path')}`")
        
        # 内容表示（詳細情報も含む）
        content = r.get('content', '')
        if content:
            st.markdown("**内容:**")
            if len(content) > 500:
                # 長い場合は常に expandable に
                with st.expander(f"内容を表示（{len(content)}文字）"):
                    st.text(content)
                # 短縮版も表示
                st.text(content[:200] + "..." if len(content) > 200 else content)
            else:
                st.text(content)

        if st.button("チェック実行"):
            try:
                docs = backend.retrieve_related_docs(vectordb, r.get('content') or r.get('title') or "", k=topk)
                st.write(f"取得ドキュメント: {len(docs)} チャンク（上位 {topk}）")
                st.info("要約中...")
                summary = backend.agent_a_summarize(llm, r.get('content') or '', docs)
                st.success("要約完了")
                # Streamlit 上では要約のプレビューは不要

                st.info("評価中...")
                b_result = backend.agent_b_check(llm, summary, r.get('raw', {}), docs)
                st.success("評価完了")
                st.subheader("判定（Agent B）")
                b_text = backend.format_b_result(b_result)
                # 表示用に改行や余分な空行を整形
                def _normalize_display(text: str) -> str:
                    if not text:
                        return ""
                    t = text.replace('\r\n', '\n').replace('\r', '\n')
                    t = re.sub(r"\n{3,}", "\n\n", t)
                    lines = [ln.rstrip() for ln in t.split('\n')]
                    while lines and lines[0].strip() == "":
                        lines.pop(0)
                    while lines and lines[-1].strip() == "":
                        lines.pop()
                    out_lines = []
                    prev_blank = False
                    for ln in lines:
                        if ln.strip() == "":
                            if not prev_blank:
                                out_lines.append("")
                            prev_blank = True
                        else:
                            out_lines.append(ln.lstrip())
                            prev_blank = False
                    return "\n".join(out_lines)

                b_text_clean = _normalize_display(b_text)
                # Markdown で整形表示: 判定、詳細、根拠一覧（各抜粋は expander で展開）
                res_symbol = b_result.get("result") or b_result.get("status") or "△"
                st.markdown(f"**判定: {res_symbol}**")

                # 詳細説明があれば表示
                details = b_result.get("details") or b_result.get("detail") or b_result.get("notes")
                if details:
                    st.markdown("**説明:**")
                    st.text(details if isinstance(details, str) else json.dumps(details, ensure_ascii=False, indent=2))

                # 根拠を表示
                evs = b_result.get("evidence_normalized") or []
                if evs:
                    st.markdown("**根拠 (参照文書と抜粋):**")
                    for i, e in enumerate(evs, 1):
                        src = e.get("source") or "(unknown)"
                        excerpt = e.get("excerpt") or ""
                        with st.expander(f"{i}. {src}"):
                            ex = excerpt.replace("\r\n", "\n").replace("\r", "\n").strip()
                            st.text(ex)
                else:
                    st.info("(根拠情報はありません)")

                with st.expander("（参考）整形済みテキスト（生）"):
                    st.text(b_text_clean)

            except Exception as e:
                st.error("評価に失敗しました。ログを確認してください。")
                st.exception(e)

elif "RAG" in page:
    st.header("💬 RAG 質問 (システム情報に関する QA)")
    st.caption("📁 PDF, Word, Excel, PowerPoint, Markdown, テキストファイルから情報を検索できます")
    
    q = st.text_input("質問を入力してください", placeholder="例: ウイルス対策の要件は？ / Excel形式の要件は？")
    if st.button("質問実行"):
        if not q:
            st.warning("質問を入力してください")
        else:
            try:
                docs = backend.retrieve_related_docs(vectordb, q, k=topk)
                st.write(f"🔍 {len(docs)}件の関連ドキュメントを検索しました")
                
                # 参照したファイルの種類を表示
                file_types_found = set()
                for d in docs:
                    file_type = d.metadata.get('file_type', 'unknown')
                    file_types_found.add(file_type)
                
                if file_types_found:
                    type_emojis = {'.pdf': '📄', '.docx': '📝', '.xlsx': '📊', '.pptx': '📈', '.md': '📋', '.txt': '📄'}
                    type_str = " ".join([f"{type_emojis.get(ft, '📄')}{ft}" for ft in sorted(file_types_found)])
                    st.caption(f"参照ファイル形式: {type_str}")
                
                context = "\n\n".join([f"[src:{d.metadata.get('source')}]\n{d.page_content}" for d in docs])
                system = "あなたはシステム情報の検索アシスタントです。ユーザの質問に、関連するドキュメントを参照して簡潔に答えてください。\n\n重要: 回答は必ず日本語で行ってください。"
                messages = [backend.SystemMessage(content=system), backend.HumanMessage(content=f"質問: {q}\n\n参照文書:\n{context}" )]
                resp = llm(messages)
                # 出力は小さく表示
                st.markdown("**回答:**")
                st.text(resp.content)
                
                # 参照元ファイルの詳細表示
                with st.expander("🔗 参照元ファイル詳細"):
                    for i, d in enumerate(docs, 1):
                        source = d.metadata.get('source', 'unknown')
                        file_type = d.metadata.get('file_type', 'unknown')
                        chunk_id = d.metadata.get('chunk', 0)
                        st.text(f"{i}. {Path(source).name} ({file_type}, chunk {chunk_id})")
                        st.text(f"   内容: {d.page_content[:100]}...")
                        st.text("")
                        
            except Exception as e:
                st.error("QA 実行でエラーが発生しました")
                st.exception(e)

st.caption("この UI は Streamlit を利用しています。バックエンドである main.py を大きく変更せずにフロントエンドを提供します。")
