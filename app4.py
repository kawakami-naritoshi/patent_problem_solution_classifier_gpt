import streamlit as st
import pandas as pd
import time
from datetime import datetime
import io
import re
from openai import OpenAI

# ページ設定
st.set_page_config(
    page_title="課題分類・解決手段分類あてはめアプリ",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# レート制限設定
RATE_LIMIT_DELAY = 0.1  # GPT-4o-mini用
GPT4_DELAY = 3.0        # GPT-4.1系用

# 利用可能なモデル一覧
AVAILABLE_MODELS = {
    "gpt-4o-mini": {"name": "GPT-4o-mini", "delay": RATE_LIMIT_DELAY, "cost": "低"},
    "gpt-4.1":     {"name": "GPT-4.1",      "delay": GPT4_DELAY,     "cost": "高"}
}
PRECISION_REPROCESS_MODEL = "gpt-4.1"

# セッション状態の初期化
if 'invalid_rows_data' not in st.session_state:
    st.session_state.invalid_rows_data = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'final_invalid' not in st.session_state:
    st.session_state.final_invalid = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'manual_correction_complete' not in st.session_state:
    st.session_state.manual_correction_complete = False
if 'need_manual_correction' not in st.session_state:
    st.session_state.need_manual_correction = False
if 'problem_def_used' not in st.session_state:
    st.session_state.problem_def_used = None
if 'solution_def_used' not in st.session_state:
    st.session_state.solution_def_used = None

# ヘッダー
st.title("🔬 課題分類・解決手段分類あてはめアプリ")
st.subheader("-AI駆動型自動分類あてはめアプリ")

with st.expander("🔧 最新機能", expanded=False):
    st.markdown("""
    - **シンプル分類処理**: GPT-4o-miniで効率的な一括処理
    - **GPT-4.1精密再処理**: 問題分類のみ自動修正
    - **コスト最適化**: 必要最小限の高性能モデル使用
    - **詳細ログ**: 使用モデル記録
    """)

# 有効カテゴリ抽出
def extract_valid_categories(def_text: str) -> list:
    lines = def_text.strip().split('\n')
    return [m.group(1) for line in lines if (m := re.match(r'\[([^\]]+)\]', line))]

# 分類結果検証
def validate_classification_results(df: pd.DataFrame, problems: list, solutions: list) -> dict:
    invalid = {'problem': [], 'solution': []}
    # 無効な回答のパターン
    invalid_patterns = ['該当なし', 'N/A', 'その他', 'None', 'なし', '不明', '該当無し', 'NA']
    
    for idx, row in df.iterrows():
        p = str(row.get('課題分類', '')).strip()
        # 課題分類の検証
        if (not p or 
            p.startswith(('エラー:', '分類エラー:')) or 
            p not in problems or
            any(pattern in p for pattern in invalid_patterns)):
            invalid['problem'].append(idx)
            
        s = str(row.get('解決手段分類', '')).strip()
        # 解決手段分類の検証
        if (not s or 
            s.startswith(('エラー:', '分類エラー:')) or 
            s not in solutions or
            any(pattern in s for pattern in invalid_patterns)):
            invalid['solution'].append(idx)
    return invalid

# 分類処理 with retry
def generate_classification_with_retry(text, def_text, kind, client, model="gpt-4o-mini", retries=3):
    delay = AVAILABLE_MODELS[model]['delay']
    # カテゴリ名のリストを抽出
    categories = extract_valid_categories(def_text)
    categories_list = ", ".join(categories)
    
    prompt = (
        f"##Task: Classify the following {kind} into EXACTLY ONE of these categories.\n"
        f"Categories: {def_text}\n\n"
        f"Input text: {text}\n\n"
        f"CRITICAL RULES:\n"
        f"1. You MUST output ONLY the category name from this list: [{categories_list}]\n"
        f"2. NEVER output '該当なし', 'N/A', 'その他', 'None', or any similar terms\n"
        f"3. If uncertain, choose the MOST SIMILAR category based on semantic meaning\n"
        f"4. Output ONLY the exact category name, nothing else\n"
        f"5. The output must be one of: {categories_list}\n\n"
        f"Answer:")
    
    for i in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50, temperature=0.1
            )
            result = resp.choices[0].message.content.strip().strip('[]')
            
            # 結果の検証
            if result not in categories:
                # もし無効な結果が返された場合、警告を出してリトライ
                if i < retries - 1:
                    time.sleep(1)
                    continue
                else:
                    # 最後のリトライでも失敗した場合、最初のカテゴリを返す
                    return categories[0] if categories else "分類エラー: カテゴリが定義されていません"
            
            time.sleep(delay)
            return result
        except Exception as e:
            if i < retries - 1:
                time.sleep(2 ** i)
            else:
                return f"分類エラー: {e}"

# 結果表示とチャート出力
def display_final_results(df, invalid):
    st.header("📊 最終分類結果")
    
    errors = sorted(set(invalid['problem'] + invalid['solution']))
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("総件数", len(df))
    with col2:
        st.metric("要確認", len(errors))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 課題分類の分布")
        prob_counts = df['課題分類'].value_counts()
        if not prob_counts.empty:
            st.bar_chart(prob_counts)
        else:
            st.info("課題分類データがありません")
    with col2:
        st.subheader("🔧 解決手段分類の分布")
        sol_counts = df['解決手段分類'].value_counts()
        if not sol_counts.empty:
            st.bar_chart(sol_counts)
        else:
            st.info("解決手段分類データがありません")

    # Excel出力用データ準備
    df_excel = df.copy()
    df_excel['分類検証結果'] = 'OK'
    df_excel['問題詳細'] = ''
    df_excel['使用モデル'] = 'gpt-4o-mini'  # デフォルト
    
    # エラー行の記録
    for idx in invalid['problem']:
        df_excel.at[idx, '分類検証結果'] = '課題分類エラー'
        df_excel.at[idx, '問題詳細'] = f"課題分類: {df.at[idx, '課題分類']}"
    for idx in invalid['solution']:
        prev = df_excel.at[idx, '分類検証結果']
        if prev == 'OK':
            df_excel.at[idx, '分類検証結果'] = '解決手段分類エラー'
            df_excel.at[idx, '問題詳細'] = f"解決手段分類: {df.at[idx, '解決手段分類']}"
        else:
            df_excel.at[idx, '分類検証結果'] = prev + ', 解決手段分類エラー'
            df_excel.at[idx, '問題詳細'] += f"; 解決手段分類: {df.at[idx, '解決手段分類']}"
    
    # 再処理された行にモデル情報を記録
    if 'reprocessed_indices' in st.session_state and st.session_state.reprocessed_indices:
        for idx in st.session_state.reprocessed_indices:
            df_excel.at[idx, '使用モデル'] = PRECISION_REPROCESS_MODEL

    # Excelファイル生成とダウンロードボタン
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df_excel.to_excel(writer, index=False, sheet_name='分類結果')
        
        # 概要シートの追加
        summary_data = {
            '項目': ['総件数', '正常分類数', 'エラー数', '課題分類エラー', '解決手段分類エラー'],
            '件数': [
                len(df),
                len(df) - len(errors),
                len(errors),
                len(invalid['problem']),
                len(invalid['solution'])
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, index=False, sheet_name='概要')
    
    buf.seek(0)
    
    st.download_button(
        label="📥 分類結果をダウンロード (Excel)",
        data=buf.getvalue(),
        file_name=f"patent_classification_result_{datetime.now():%Y%m%d_%H%M%S}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# 再処理関数
def reprocess_invalid_classifications(df, invalid, prob_def, sol_def, client):
    st.subheader(f"🔄 {AVAILABLE_MODELS[PRECISION_REPROCESS_MODEL]['name']}による精密再処理中...")
    st.info("「該当なし」等の無効な分類を、最も適切なカテゴリに再分類します。")
    
    indices = sorted(set(invalid['problem'] + invalid['solution']))
    st.session_state.reprocessed_indices = indices  # 再処理した行を記録
    
    if not indices:
        st.info("再処理対象なし")
        return df
    progress = st.progress(0)
    total = len(indices)
    for i, idx in enumerate(indices):
        row = df.iloc[idx]
        current_prob = df.at[idx, '課題分類']
        current_sol = df.at[idx, '解決手段分類']
        
        # 無効な分類の場合のみ再処理
        if idx in invalid['problem']:
            with st.expander(f"行 {idx+1} - 課題分類を再処理中", expanded=False):
                st.write(f"現在の分類: {current_prob}")
                st.write(f"要約: {row['要約'][:100]}...")
            df.at[idx, '課題分類'] = generate_classification_with_retry(
                row['要約'], prob_def, 'problem', client, PRECISION_REPROCESS_MODEL)
        if idx in invalid['solution']:
            with st.expander(f"行 {idx+1} - 解決手段分類を再処理中", expanded=False):
                st.write(f"現在の分類: {current_sol}")
                st.write(f"要約: {row['要約'][:100]}...")
            df.at[idx, '解決手段分類'] = generate_classification_with_retry(
                row['要約'], sol_def, 'solution', client, PRECISION_REPROCESS_MODEL)
        progress.progress((i+1)/total)
    st.success("✅ 精密再処理完了")
    return df

# サイドバー
with st.sidebar:
    st.header("⚙️ 設定")
    st.subheader("🔑 APIキー")
    api_key = st.text_input("OpenAI API Key", type="password")
    client = OpenAI(api_key=api_key) if api_key else None
    st.subheader("🤖 モデル設定")
    st.info(f"初回: {AVAILABLE_MODELS['gpt-4o-mini']['name']} | 再処理: {AVAILABLE_MODELS[PRECISION_REPROCESS_MODEL]['name']}")
    
    # 分類ルールの説明
    with st.expander("📋 分類ルール", expanded=False):
        st.markdown("""
        **重要な分類ルール:**
        - ✅ 必ず定義されたカテゴリから選択
        - ❌ 「該当なし」は絶対に出力しない
        - 🎯 不確実な場合は最も近いカテゴリを選択
        - 🔄 無効な分類は自動的に再処理
        """)
    
    # リセットボタン
    if st.button("🔄 新規処理を開始"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# 分類定義入力
col1, col2 = st.columns(2)
with col1:
    st.subheader("📝 課題分類定義入力")
    problem_def = st.text_area(
        "課題分類",
        value="""[モータ効率・性能向上] 説明文: 電気モータの効率改善、小型化、高出力化に関する課題
[バッテリー技術] 説明文: バッテリーの容量、寿命、充電速度、安全性に関する課題
[制御システム] 説明文: モータ制御、インバータ制御、システム統合に関する課題""",
        height=200)
with col2:
    st.subheader("🔧 解決手段分類定義入力")
    solution_def = st.text_area(
        "解決手段分類",
        value="""[モータ構造の最適化] 説明文: ブラシレスモータのロータやステータ構造の改良
[材料技術の改善] 説明文: 新素材の採用、磁石材料の改良、絶縁材料の向上
[制御アルゴリズム] 説明文: 高効率制御手法、ベクトル制御、センサレス制御""",
        height=200)

# 処理完了後の結果表示
if st.session_state.processing_complete and st.session_state.processed_df is not None:
    df = st.session_state.processed_df
    
    # 手動修正が必要で、まだ完了していない場合
    if st.session_state.need_manual_correction and not st.session_state.manual_correction_complete:
        st.warning("🔴 再処理後も未分類項目があります。手動で修正してください。")
        
        # カテゴリリストを取得
        probs = extract_valid_categories(st.session_state.problem_def_used)
        sols = extract_valid_categories(st.session_state.solution_def_used)
        
        invalid = st.session_state.final_invalid
        indices = sorted(set(invalid['problem'] + invalid['solution']))
        
        # 手動修正用の入力フィールド
        st.markdown("### 📝 手動修正")
        for idx in indices:
            row = df.iloc[idx]
            st.markdown(f"**行 {idx+1}** - 要約: {row['要約'][:100]}...")
            col1, col2 = st.columns(2)
            with col1:
                manual_prob = st.text_input(
                    f"課題分類 (現在: {row['課題分類']})", 
                    value=row['課題分類'], 
                    key=f"manual_prob_{idx}"
                )
            with col2:
                manual_sol = st.text_input(
                    f"解決手段分類 (現在: {row['解決手段分類']})", 
                    value=row['解決手段分類'], 
                    key=f"manual_sol_{idx}"
                )
            st.markdown("---")
        
        # 手動修正の確定ボタン
        if st.button("✅ 手動修正を確定してExcelをダウンロード", type="primary"):
            # 手動修正の適用
            for idx in indices:
                df.at[idx, '課題分類'] = st.session_state[f"manual_prob_{idx}"]
                df.at[idx, '解決手段分類'] = st.session_state[f"manual_sol_{idx}"]
            
            # 最終検証
            final_invalid = validate_classification_results(df, probs, sols)
            
            # セッション状態を更新
            st.session_state.processed_df = df
            st.session_state.final_invalid = final_invalid
            st.session_state.manual_correction_complete = True
            st.rerun()
    
    # 処理完了後の結果表示
    else:
        if st.session_state.final_invalid is not None:
            display_final_results(df, st.session_state.final_invalid)
        else:
            display_final_results(df, st.session_state.invalid_rows_data)

# ファイルアップロード＆処理（処理が完了していない場合のみ表示）
elif not st.session_state.processing_complete:
    st.subheader("📁 ファイルアップロード")
    uploaded = st.file_uploader("Excel (.xlsx)", type=['xlsx'])
    if uploaded:
        df = pd.read_excel(uploaded)
        if '要約' not in df.columns:
            st.error("「要約」列がありません")
        else:
            total = len(df)
            st.success(f"{total}件読み込み完了")
            with st.expander("データプレビュー", False):
                st.dataframe(df.head())
            if st.button("分類開始", disabled=not client):
                progress = st.progress(0)
                status = st.empty()
                
                # 定義を保存
                st.session_state.problem_def_used = problem_def
                st.session_state.solution_def_used = solution_def
                
                for idx, row in df.iterrows():
                    df.at[idx, '課題分類'] = generate_classification_with_retry(
                        row['要約'], problem_def, 'problem', client)
                    df.at[idx, '解決手段分類'] = generate_classification_with_retry(
                        row['要約'], solution_def, 'solution', client)
                    progress.progress((idx+1)/total)
                    status.text(f"処理中: {idx+1}/{total}")

                # 初回検証
                probs = extract_valid_categories(problem_def)
                sols = extract_valid_categories(solution_def)
                invalid = validate_classification_results(df, probs, sols)
                st.session_state.invalid_rows_data = invalid
                st.session_state.processed_df = df.copy()

                if invalid['problem'] or invalid['solution']:
                    st.info("⚙️ 問題が検出されました。自動でGPT-4.1再処理を開始します...")
                    df = reprocess_invalid_classifications(df, invalid, problem_def, solution_def, client)

                    # 再処理後の再検証
                    new_invalid = validate_classification_results(df, probs, sols)
                    st.session_state.final_invalid = new_invalid
                    st.session_state.processed_df = df.copy()

                    if new_invalid['problem'] or new_invalid['solution']:
                        st.session_state.need_manual_correction = True
                        st.session_state.processing_complete = True
                        st.rerun()
                    else:
                        st.success("✅ 再処理で全て正常に分類されました")
                        st.session_state.processing_complete = True
                        st.session_state.need_manual_correction = False
                        st.rerun()
                else:
                    st.success("✅ 全て正常に分類されました")
                    st.session_state.processing_complete = True
                    st.session_state.need_manual_correction = False
                    st.rerun()

# フッター
st.markdown("---")
st.markdown("**Powered by OpenAI GPT Models**")
st.markdown("**Ⓒ2025**")
