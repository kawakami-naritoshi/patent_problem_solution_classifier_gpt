import streamlit as st
import pandas as pd
import time
from datetime import datetime
import io
from openai import OpenAI
import pickle
import os
import re

# ページ設定
st.set_page_config(
    page_title="PatentScope AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# レート制限設定（GPT-4o-mini用）
RATE_LIMIT_DELAY = 1.0  # GPT-4o-miniは高いレート制限のため短縮
BATCH_SIZE = 50  # バッチ処理サイズ
CHECKPOINT_FILE = "checkpoint.pkl"  # チェックポイントファイル

# セッション状態の初期化
if 'processing_state' not in st.session_state:
    st.session_state.processing_state = None
if 'current_batch' not in st.session_state:
    st.session_state.current_batch = 0
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# メインヘッダー
st.title("🔬 PatentScope AI")
st.subheader("次世代特許分析プラットフォーム - AI駆動型知財インテリジェンス")

# 改善点の説明
with st.expander("🔧 安定性向上機能", expanded=False):
    st.markdown("### ✨ 新機能")
    st.markdown("""
    - **GPT-4o-mini採用**: 高速・高精度な分類処理
    - **バッチ処理**: 大量データを小分けして処理
    - **中断・再開機能**: 処理中断時も続きから再開可能
    - **エラーハンドリング**: API エラー時の自動リトライ
    - **プログレス保存**: 処理状況の自動保存
    - **レート制限最適化**: GPT-4o-mini用に高速化（1秒間隔）
    - **分類結果検証**: 定義された分類カテゴリとの一致をチェック
    """)

# 分類定義から有効なカテゴリを抽出する関数
def extract_valid_categories(classification_def):
    """分類定義から有効なカテゴリ名を抽出"""
    categories = []
    lines = classification_def.strip().split('\n')
    for line in lines:
        # [カテゴリ名] 形式を抽出
        match = re.match(r'\[([^\]]+)\]', line.strip())
        if match:
            categories.append(match.group(1))
    return categories

# 分類結果を検証する関数
def validate_classification_results(df, problem_categories, solution_categories):
    """分類結果を検証し、問題のある行を特定"""
    invalid_rows = {
        'problem': [],
        'solution': [],
        'both': []
    }
    
    for idx, row in df.iterrows():
        problem_valid = True
        solution_valid = True
        
        # 課題分類のチェック
        if pd.notna(row.get('課題分類', '')):
            p_class = str(row['課題分類']).strip()
            if (p_class == "" or 
                p_class.startswith("エラー:") or 
                p_class == "該当するカテゴリはありません" or
                p_class not in problem_categories):
                problem_valid = False
                invalid_rows['problem'].append({
                    'index': idx + 1,  # 1-based index for display
                    'value': p_class,
                    'summary': row.get('要約', '')[:50] + '...' if len(str(row.get('要約', ''))) > 50 else row.get('要約', '')
                })
        
        # 解決手段分類のチェック
        if pd.notna(row.get('解決手段分類', '')):
            s_class = str(row['解決手段分類']).strip()
            if (s_class == "" or 
                s_class.startswith("エラー:") or 
                s_class == "該当するカテゴリはありません" or
                s_class not in solution_categories):
                solution_valid = False
                invalid_rows['solution'].append({
                    'index': idx + 1,
                    'value': s_class,
                    'summary': row.get('要約', '')[:50] + '...' if len(str(row.get('要約', ''))) > 50 else row.get('要約', '')
                })
        
        # 両方無効な場合
        if not problem_valid and not solution_valid:
            invalid_rows['both'].append(idx + 1)
    
    return invalid_rows

# チェックポイント復旧機能
def save_checkpoint(data, batch_num, stage):
    """処理状況をチェックポイントとして保存"""
    checkpoint = {
        'data': data,
        'batch_num': batch_num,
        'stage': stage,
        'timestamp': datetime.now()
    }
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(checkpoint, f)

def load_checkpoint():
    """チェックポイントから復旧"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'rb') as f:
                return pickle.load(f)
        except:
            return None
    return None

def cleanup_checkpoint():
    """チェックポイントファイルを削除"""
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

# サイドバー設定
with st.sidebar:
    st.header("⚙️ 設定")
    
    # APIキー入力
    st.subheader("🔑 OpenAI API設定")
    api_key = st.text_input(
        "APIキー",
        type="password",
        help="OpenAI Platform (https://platform.openai.com/api-keys) で取得できます"
    )
    
    if api_key:
        st.success("APIキーが設定されました ✅")
        # OpenAI クライアント初期化
        client = OpenAI(api_key=api_key)
    else:
        st.warning("APIキーを入力してください")
        client = None
    
    # バッチサイズ設定
    st.subheader("⚙️ 処理設定")
    batch_size = st.slider("バッチサイズ", 10, 100, BATCH_SIZE, 10)
    st.info(f"データを{batch_size}件ずつ処理します")
    
    # チェックポイント管理
    st.subheader("💾 復旧機能")
    checkpoint = load_checkpoint()
    if checkpoint:
        st.warning(f"⚠️ 未完了の処理があります")
        st.info(f"時刻: {checkpoint['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        st.info(f"バッチ: {checkpoint['batch_num']}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 処理を再開", type="primary"):
                st.session_state.processing_state = checkpoint
                st.rerun()
        with col2:
            if st.button("🗑️ リセット"):
                cleanup_checkpoint()
                st.rerun()

# 分類定義入力（元のコードと同じ）
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 課題分類定義の入力")
    problem_classification = st.text_area(
        "課題分類カテゴリ",
        value="""[モータ効率・性能向上] 説明文: 電気モータの効率改善、小型化...""",
        height=200,
        key="problem_def"
    )

with col2:
    st.subheader("🔧 解決手段分類定義の入力")
    solution_classification = st.text_area(
        "解決手段分類カテゴリ",
        value="""[モータ構造の最適化] 説明文: ブラシレスモータのロータ...""",
        height=200,
        key="solution_def"
    )

# ファイルアップロード
st.subheader("📁 データファイルアップロード")
uploaded_file = st.file_uploader(
    "Excelファイルを選択してください",
    type=['xlsx'],
    help="「要約」列を含むExcelファイル (.xlsx) をアップロードしてください"
)

# 改良された分類処理関数
def generate_classification_with_retry(text, classification_def, classification_type, client, max_retries=3):
    """リトライ機能付き分類処理（GPT-4o-mini用）"""
    for attempt in range(max_retries):
        try:
            if classification_type == "problem":
                prompt = f"""##Task: Classify the input problem description into one of the problem categories below. You MUST select the most appropriate category from the list. Do not answer "該当するカテゴリはありません" or similar. Output only the category name in Japanese WITHOUT square brackets [].

##Problem Categories: {classification_def}

##Instructions:
1. Read the input description carefully
2. Compare it with ALL categories
3. Select the MOST appropriate category (even if not perfect match)
4. Output ONLY the category name without brackets []
5. Answer in Japanese only

##Input: {text}

##Answer (category name only, no brackets):"""
            else:
                prompt = f"""##Task: Classify the input solution description into one of the solution categories below. You MUST select the most appropriate category from the list. Do not answer "該当するカテゴリはありません" or similar. Output only the category name in Japanese WITHOUT square brackets [].

##Solution Categories: {classification_def}

##Instructions:
1. Read the input description carefully
2. Compare it with ALL categories
3. Select the MOST appropriate category (even if not perfect match)
4. Output ONLY the category name without brackets []
5. Answer in Japanese only

##Input: {text}

##Answer (category name only, no brackets):"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            
            # []括弧がある場合は除去
            if result.startswith('[') and result.endswith(']'):
                result = result[1:-1]
            
            return result
            
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"API エラー (試行 {attempt + 1}): {str(e)} - リトライします...")
                time.sleep(2 ** attempt)  # 指数バックオフ
            else:
                return f"分類エラー: {str(e)}"

def process_batch(df_batch, start_idx, problem_def, solution_def, stage, client):
    """バッチ処理"""
    batch_results = []
    
    for i, (idx, row) in enumerate(df_batch.iterrows()):
        current_idx = start_idx + i
        
        try:
            if stage in ['problem', 'both']:
                # 課題分類
                p_class = generate_classification_with_retry(
                    row['要約'], problem_def, "problem", client
                )
                batch_results.append({
                    'index': idx,
                    'problem_class': p_class,
                    'solution_class': None
                })
            
            if stage in ['solution', 'both']:
                # 解決手段分類
                s_class = generate_classification_with_retry(
                    row['要約'], solution_def, "solution", client
                )
                
                if stage == 'solution' and batch_results:
                    batch_results[i]['solution_class'] = s_class
                elif stage == 'both':
                    batch_results[i]['solution_class'] = s_class
                else:
                    batch_results.append({
                        'index': idx,
                        'problem_class': None,
                        'solution_class': s_class
                    })
            
            # レート制限
            time.sleep(RATE_LIMIT_DELAY)
            
        except Exception as e:
            st.error(f"行 {current_idx} でエラー: {str(e)}")
            batch_results.append({
                'index': idx,
                'problem_class': f"エラー: {str(e)}",
                'solution_class': f"エラー: {str(e)}"
            })
    
    return batch_results

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        
        if '要約' not in df.columns:
            st.error("❌ エラー: 「要約」列が見つかりません")
        else:
            st.success("✅ ファイル読み込み完了")
            st.info(f"{len(df)}行のデータが読み込まれました")
            
            # データプレビュー
            with st.expander("📊 データプレビュー", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)
            
            # 処理時間の推定
            total_batches = (len(df) + batch_size - 1) // batch_size
            estimated_time = len(df) * RATE_LIMIT_DELAY * 2 / 60  # GPT-4o-mini用の短縮時間
            
            st.subheader("🚀 分類処理")
            st.info(f"📊 総データ数: {len(df)}件")
            st.info(f"📦 バッチ数: {total_batches}バッチ")
            st.info(f"⏱️ 推定処理時間: 約{estimated_time:.1f}分")
            
            # 処理開始ボタン
            if st.button("🚀 分類処理開始", type="primary", disabled=not client):
                if not client:
                    st.error("❌ APIキーが必要です")
                else:
                    # 処理状況の表示
                    progress_container = st.container()
                    log_container = st.empty()
                    
                    with progress_container:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        batch_info = st.empty()
                    
                    try:
                        # 結果格納用の列を初期化
                        if '課題分類' not in df.columns:
                            df['課題分類'] = ""
                        if '解決手段分類' not in df.columns:
                            df['解決手段分類'] = ""
                        
                        start_time = datetime.now()
                        
                        # バッチ処理ループ
                        for batch_num in range(total_batches):
                            start_idx = batch_num * batch_size
                            end_idx = min(start_idx + batch_size, len(df))
                            df_batch = df.iloc[start_idx:end_idx]
                            
                            # プログレス更新
                            progress = (batch_num / total_batches)
                            progress_bar.progress(progress)
                            status_text.text(f"バッチ {batch_num + 1}/{total_batches} 処理中...")
                            batch_info.info(f"📦 現在のバッチ: {start_idx + 1}～{end_idx}行目")
                            
                            # バッチ処理実行
                            batch_results = process_batch(
                                df_batch, start_idx, 
                                problem_classification, 
                                solution_classification, 
                                'both',
                                client
                            )
                            
                            # 結果をデータフレームに反映
                            for result in batch_results:
                                idx = result['index']
                                if result['problem_class'] is not None:
                                    df.at[idx, '課題分類'] = result['problem_class']
                                if result['solution_class'] is not None:
                                    df.at[idx, '解決手段分類'] = result['solution_class']
                            
                            # チェックポイント保存
                            save_checkpoint(df, batch_num + 1, 'both')
                            
                            # ログ更新
                            elapsed_time = (datetime.now() - start_time).total_seconds() / 60
                            log_container.info(f"バッチ {batch_num + 1} 完了 (経過時間: {elapsed_time:.1f}分)")
                        
                        # 処理完了
                        progress_bar.progress(1.0)
                        total_time = (datetime.now() - start_time).total_seconds() / 60
                        status_text.success(f"✅ 全処理完了！ (処理時間: {total_time:.1f}分)")
                        
                        # チェックポイントファイル削除
                        cleanup_checkpoint()
                        
                        # 分類結果の検証
                        st.header("🔍 分類結果の検証")
                        
                        # 有効なカテゴリを抽出
                        problem_categories = extract_valid_categories(problem_classification)
                        solution_categories = extract_valid_categories(solution_classification)
                        
                        # 検証実行
                        invalid_rows = validate_classification_results(df, problem_categories, solution_categories)
                        
                        # 検証結果の表示
                        total_invalid = len(invalid_rows['problem']) + len(invalid_rows['solution'])
                        
                        if total_invalid == 0:
                            st.success("✅ すべての分類が正しく付与されています！")
                        else:
                            st.error(f"⚠️ {total_invalid}件の分類に問題が見つかりました")
                            
                            # 問題の詳細表示
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if invalid_rows['problem']:
                                    st.error(f"❌ 課題分類の問題: {len(invalid_rows['problem'])}件")
                                    with st.expander("詳細を表示", expanded=True):
                                        for item in invalid_rows['problem'][:10]:  # 最初の10件のみ表示
                                            st.write(f"**行 {item['index']}**: '{item['value']}'")
                                            st.write(f"要約: {item['summary']}")
                                            st.divider()
                                        if len(invalid_rows['problem']) > 10:
                                            st.info(f"他 {len(invalid_rows['problem']) - 10}件...")
                            
                            with col2:
                                if invalid_rows['solution']:
                                    st.error(f"❌ 解決手段分類の問題: {len(invalid_rows['solution'])}件")
                                    with st.expander("詳細を表示", expanded=True):
                                        for item in invalid_rows['solution'][:10]:
                                            st.write(f"**行 {item['index']}**: '{item['value']}'")
                                            st.write(f"要約: {item['summary']}")
                                            st.divider()
                                        if len(invalid_rows['solution']) > 10:
                                            st.info(f"他 {len(invalid_rows['solution']) - 10}件...")
                            
                            # 有効なカテゴリ一覧の表示
                            with st.expander("📋 定義されている有効なカテゴリ", expanded=False):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**課題分類カテゴリ:**")
                                    for cat in problem_categories:
                                        st.write(f"• {cat}")
                                with col2:
                                    st.write("**解決手段分類カテゴリ:**")
                                    for cat in solution_categories:
                                        st.write(f"• {cat}")
                        
                        # 結果表示と統計
                        st.header("📊 分類結果")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("🎯 課題分類の分布")
                            p_counts = df['課題分類'].value_counts()
                            st.bar_chart(p_counts)
                        
                        with col2:
                            st.subheader("🔧 解決手段分類の分布")
                            s_counts = df['解決手段分類'].value_counts()
                            st.bar_chart(s_counts)
                        
                        # 結果ダウンロード
                        st.subheader("💾 結果ダウンロード")
                        
                        # 問題のある行にフラグを付けてダウンロード
                        df_download = df.copy()
                        df_download['分類検証結果'] = ''
                        
                        for item in invalid_rows['problem']:
                            idx = item['index'] - 1
                            df_download.at[idx, '分類検証結果'] = '課題分類エラー'
                        
                        for item in invalid_rows['solution']:
                            idx = item['index'] - 1
                            if df_download.at[idx, '分類検証結果']:
                                df_download.at[idx, '分類検証結果'] += ', 解決手段分類エラー'
                            else:
                                df_download.at[idx, '分類検証結果'] = '解決手段分類エラー'
                        
                        output_buffer = io.BytesIO()
                        with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
                            df_download.to_excel(writer, index=False, sheet_name='分類結果')
                        
                        st.download_button(
                            label="📥 Excelファイルでダウンロード（検証結果付き）",
                            data=output_buffer.getvalue(),
                            file_name=f"classification_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ 処理エラー: {str(e)}")
                        st.error("処理を中断しました。サイドバーから再開できます。")
                        
    except Exception as e:
        st.error(f"❌ ファイル読み込みエラー: {str(e)}")

# フッター
st.markdown("---")
st.markdown("### 🔬 PatentScope AI - GPT-4o-mini Edition")
st.markdown("**Powered by OpenAI GPT-4o-mini | 安定性向上版**")