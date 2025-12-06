#
# run with: streamlit run streamlit_csv_query_app_v23.py      cd c:\users\oakhtar\documents\pyprojs_local
#
# run with: streamlit run streamlit_csv_query_app_v27.py
#
#
# run with: streamlit run streamlit_csv_query_app_v28.py
#
#
# run with: streamlit run streamlit_csv_query_app_v29.py
#
import streamlit as st
import pandas as pd
import time
import os
import re
import shutil
import psutil
import logging
import json
import requests
import traceback
import gc
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

try:
    from fuzzywuzzy import fuzz
except ImportError:
    st.error("Failed to import fuzzywuzzy. Please install python-Levenshtein and fuzzywuzzy.")
    st.stop()

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================
MAX_FILE_SIZE_MB = 200
DEFAULT_CHUNK_SIZE = 100000
MAX_TEMP_STORAGE_MB = 1000
APP_DIR = Path(__file__).parent if "__file__" in dir() else Path(".")
DATA_DIR = APP_DIR / "data"
DOWNLOADS_DIR = APP_DIR / "downloads"
TEMP_DIR = APP_DIR / "temp_csv_query"

# Ensure directories exist
for d in [DATA_DIR, DOWNLOADS_DIR, TEMP_DIR]:
    d.mkdir(exist_ok=True)

# Regex Patterns
DATE_PATTERNS = [
    re.compile(r'^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}.*)?$', re.IGNORECASE),
    re.compile(r'^\d{2}/\d{2}/\d{4}$'),
    re.compile(r'^\d{4}/\d{2}/\d{2}$')
]
NUMERIC_PATTERN = re.compile(r'^-?\d*\.?\d*$', re.IGNORECASE)

logging.basicConfig(filename="bad_rows.log", level=logging.WARNING, format='%(asctime)s - %(message)s')

# ============================================================================
# SESSION STATE & UTILS
# ============================================================================

def init_session_state():
    defaults = {
        'dataset_configs': [],
        'columns_dict': {},
        'preprocess_columns': {},
        'interactive_state': {},
        # AI / Auth States
        'authenticated': False,
        'auth_error': False,
        'total_tokens': 0,
        'total_cost': 0.0,
        'ai_provider_config': {}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def perform_login():
    """Callback for password validation"""
    password = st.session_state.password_input
    # Check against secrets.toml or default 'admin'
    correct_password = st.secrets.get("auth_password", "admin")
    
    if password == correct_password:
        st.session_state['authenticated'] = True
        st.session_state['auth_error'] = False
        st.session_state['password_input'] = "" # Clear input
    else:
        st.session_state['auth_error'] = True

def logout():
    st.session_state['authenticated'] = False

def estimate_cost(input_str, output_str, price_in_per_m, price_out_per_m):
    """Rough estimation of cost based on char count (4 chars ~= 1 token)"""
    in_tokens = len(input_str) / 4
    out_tokens = len(output_str) / 4
    cost = (in_tokens / 1_000_000 * price_in_per_m) + (out_tokens / 1_000_000 * price_out_per_m)
    
    st.session_state['total_tokens'] += int(in_tokens + out_tokens)
    st.session_state['total_cost'] += cost

def get_safe_chunk_size():
    try:
        available_mem = psutil.virtual_memory().available
        safe_chunk = int((available_mem * 0.1) / 500)
        return max(10000, min(safe_chunk, 1000000))
    except:
        return DEFAULT_CHUNK_SIZE

def check_disk_space(path, required_mb):
    try:
        return psutil.disk_usage(str(path)).free / (1024 ** 2) > required_mb
    except:
        return True

# ============================================================================
# DATA PROCESSING
# ============================================================================

@st.cache_data(ttl=3600)
def get_columns_and_samples(dataset_configs_tuple):
    dataset_configs = list(dataset_configs_tuple)
    columns_dict = {}
    auto_preprocess = {}

    for file_path, dataset_name in dataset_configs:
        try:
            df = pd.read_csv(file_path, nrows=100, engine='c', dtype=str, low_memory=False)
            columns = df.columns.tolist()
            sample_values = {}
            for col in columns:
                non_null = df[col][~df[col].isna() & (df[col] != '')]
                val = non_null.iloc[0] if not non_null.empty else 'NaN'
                sample_values[col] = val
                
                key = f"{dataset_name}.{col}"
                val_str = str(val)
                if any(x in col.lower() for x in ['date', 'time']) or any(p.match(val_str) for p in DATE_PATTERNS):
                    auto_preprocess[key] = 'datetime'
                elif NUMERIC_PATTERN.match(val_str) and col.lower() not in ['id', 'zip']:
                    auto_preprocess[key] = 'numeric'
            columns_dict[dataset_name] = (columns, sample_values)
        except Exception as e:
            st.error(f"Error reading {dataset_name}: {e}")
            
    return columns_dict, auto_preprocess

def find_column(user_input, columns, threshold=70):
    if user_input in columns: return user_input
    for col in columns:
        if col.lower() == user_input.lower(): return col
    
    best_match, best_score = None, 0
    user_norm = user_input.lower().replace("_", "")
    for col in columns:
        col_norm = col.lower().replace("_", "")
        score = fuzz.token_sort_ratio(user_norm, col_norm)
        if score >= threshold and score > best_score:
            best_match, best_score = col, score
    if best_match: return best_match
    raise ValueError(f"Column '{user_input}' not found.")

def detect_join_conditions(datasets, columns_dict, interactive=False):
    join_conditions = []
    if len(datasets) < 2: return join_conditions
    
    for i in range(len(datasets) - 1):
        ds1, ds2 = datasets[i], datasets[i + 1]
        cols1 = {re.sub(r'\W+', '', c.lower()): c for c in columns_dict[ds1][0]}
        cols2 = {re.sub(r'\W+', '', c.lower()): c for c in columns_dict[ds2][0]}
        common = set(cols1.keys()) & set(cols2.keys())
        
        if common:
            norm_col = list(common)[0]
            join_conditions.append((f"{ds1}.{cols1[norm_col]}", f"{ds2}.{cols2[norm_col]}"))
    return join_conditions

# ============================================================================
# QUERY ENGINE & LLM
# ============================================================================

def run_llm_parse(query_text, columns_dict, config):
    """Execute LLM call to parse natural language to JSON structure"""
    try:
        llm = ChatOpenAI(
            model=config['model_name'],
            api_key=config['api_key'],
            base_url=config['base_url'],
            temperature=0
        )

        columns_info = {ds: cols[0] for ds, cols in columns_dict.items()}
        
        prompt = PromptTemplate(
            input_variables=["query", "columns_info"],
            template="""You are a SQL expert converting natural language to structured JSON for a CSV query tool.
Datasets and Columns available: {columns_info}

Rules:
1. Identify datasets (e.g., "A", "B").
2. Identify join conditions if multiple datasets (e.g. "A.id = B.id").
3. Identify filters. Operators: ==, !=, >, <, contains, like.
4. Return ONLY JSON. No markdown formatting.

Format:
{{
    "datasets": ["A", "B"],
    "join_conditions": [["A.Id", "B.Id"]],
    "filter_conditions": [
        {{"dataset": "A", "column": "Score", "operator": ">", "value": 50}},
        {{"dataset": "B", "column": "Name", "operator": "contains", "value": "John"}}
    ]
}}

Query: "{query}"
JSON:"""
        )
        
        chain = prompt | llm
        response = chain.invoke({"query": query_text, "columns_info": str(columns_info)})
        content = response.content.strip()
        
        # Estimate Cost
        estimate_cost(query_text + str(columns_info), content, config['price_in'], config['price_out'])
        
        # Parse JSON (Handle markdown code blocks)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
            
        parsed = json.loads(content)
        return parsed.get('datasets', []), parsed.get('join_conditions', []), parsed.get('filter_conditions', [])

    except Exception as e:
        st.error(f"AI Error: {e}")
        return [], [], []

def parse_query_string(query_str, columns_dict):
    datasets, join_conditions, filter_conditions, output_columns = [], [], [], []
    
    select_match = re.search(r'SELECT\s+(.+?)\s+FROM', query_str, re.IGNORECASE | re.DOTALL)
    if select_match and select_match.group(1).strip() != '*':
        output_columns = [c.strip() for c in select_match.group(1).split(',')]

    from_match = re.search(r'FROM\s+(\w+)', query_str, re.IGNORECASE)
    if from_match: datasets.append(from_match.group(1).upper())
    
    for m in re.finditer(r'JOIN\s+(\w+)\s+ON\s+([\w\.]+)\s*=\s*([\w\.]+)', query_str, re.IGNORECASE):
        datasets.append(m.group(1).upper())
        join_conditions.append((m.group(2), m.group(3)))

    where_match = re.search(r'WHERE\s+(.+)', query_str, re.IGNORECASE)
    if where_match:
        parts = re.split(r'\band\b', where_match.group(1), flags=re.IGNORECASE)
        for part in parts:
            m = re.match(r'(\w+)\.(\w+)\s*([=!<>]+|contains|num_contains|like)\s*(.+)', part.strip(), re.IGNORECASE)
            if m:
                ds, col, op, val = m.groups()
                ds = ds.upper()
                if ds in columns_dict:
                    real_col = find_column(col, columns_dict[ds][0])
                    val = val.strip().strip("'\"")
                    if op not in ['contains', 'like'] and val.replace('.','',1).isdigit():
                        val = float(val) if '.' in val else int(val)
                    filter_conditions.append({'dataset': ds, 'column': real_col, 'operator': op.lower(), 'value': val})

    if not datasets: raise ValueError("Missing FROM clause")
    if not join_conditions and len(datasets) > 1:
        join_conditions = detect_join_conditions(datasets, columns_dict)
        
    return datasets, join_conditions, filter_conditions, output_columns

def query_csvs(dataset_configs, datasets, join_conditions, filter_conditions, preprocess_cols, 
              chunk_size, max_rows, output_cols, max_temp_storage):
    
    path_map = {name: path for path, name in dataset_configs}
    TEMP_DIR.mkdir(exist_ok=True)
    filtered_chunks = {ds: [] for ds in datasets}
    total_rows = 0
    
    # Status Container for UX
    with st.status(f"🚀 Processing {len(datasets)} datasets...", expanded=True) as status:
        
        # 1. Prepare Columns
        status.write("Preparing Schema...")
        required_cols = {ds: set() for ds in datasets}
        for c in filter_conditions: 
            if c['dataset'] in datasets: required_cols[c['dataset']].add(c['column'])
        for l, r in join_conditions:
            required_cols[l.split('.')[0]].add(l.split('.')[1])
            required_cols[r.split('.')[0]].add(r.split('.')[1])
        cols_to_return = []
        for col_spec in (output_cols or []):
            ds, col = col_spec.split('.')
            if ds in datasets:
                required_cols[ds].add(col)
                cols_to_return.append(col_spec)

        # 2. Process Datasets
        for i, ds_name in enumerate(datasets):
            status.write(f"Scanning dataset {ds_name} (chunk size: {chunk_size:,})...")
            fpath = path_map.get(ds_name)
            
            # If output_cols is None (User wants everything), read all columns
            if output_cols is None:
                use_cols = None
            else:
                use_cols = list(required_cols[ds_name]) or None
            
            for chunk in pd.read_csv(fpath, chunksize=chunk_size, usecols=use_cols, dtype=str, on_bad_lines='warn'):
                # Preprocess
                for col in chunk.columns:
                    key = f"{ds_name}.{col}"
                    if preprocess_cols.get(key) == 'numeric':
                        chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
                    elif preprocess_cols.get(key) == 'datetime':
                        chunk[col] = pd.to_datetime(chunk[col], errors='coerce')

                # Filter
                mask = pd.Series(True, index=chunk.index)
                for cond in [c for c in filter_conditions if c['dataset'] == ds_name]:
                    col, op, val = cond['column'], cond['operator'], cond['value']
                    if col not in chunk.columns: continue
                    series = chunk[col]
                    if op == '==': mask &= (series == val)
                    elif op == '!=': mask &= (series != val)
                    elif op == '>': mask &= (series > val)
                    elif op == '<': mask &= (series < val)
                    elif op == 'contains': mask &= series.astype(str).str.contains(str(val), case=False, na=False)
                    
                chunk = chunk[mask]
                if not chunk.empty:
                    if join_conditions:
                        tpath = TEMP_DIR / f"{ds_name}_{len(filtered_chunks[ds_name])}.csv"
                        chunk.to_csv(tpath, index=False)
                        filtered_chunks[ds_name].append(tpath)
                    else:
                        chunk.columns = [f"{ds_name}.{c}" for c in chunk.columns]
                        filtered_chunks[ds_name].append(chunk)
                        total_rows += len(chunk)
                        if max_rows and total_rows >= max_rows: break

        # 3. Merge or Stack
        result = pd.DataFrame()
        
        # --- JOIN LOGIC ---
        if join_conditions:
            status.write("Merging/Joining filtered data...")
            if all(filtered_chunks.values()):
                base_chunks = [pd.read_csv(f) for f in filtered_chunks[datasets[0]]]
                if base_chunks:
                    full_df = pd.concat(base_chunks)
                    full_df.columns = [f"{datasets[0]}.{c}" for c in full_df.columns]
                    
                    for i in range(1, len(datasets)):
                        right_ds = datasets[i]
                        right_chunks = [pd.read_csv(f) for f in filtered_chunks[right_ds]]
                        if not right_chunks: 
                            full_df = pd.DataFrame(); break
                        right_df = pd.concat(right_chunks)
                        right_df.columns = [f"{right_ds}.{c}" for c in right_df.columns]
                        
                        left_on, right_on = join_conditions[i-1] 
                        full_df = full_df.merge(right_df, left_on=left_on, right_on=right_on)
                    result = full_df
        
        # --- STACK/APPEND LOGIC ---
        else:
            status.write("Stacking/Appending results...")
            dfs = []
            for ds in datasets:
                for item in filtered_chunks[ds]:
                    if isinstance(item, pd.DataFrame):
                        dfs.append(item)
                    elif isinstance(item, str) or isinstance(item, Path):
                         # Should not happen in stack mode (we store DFs in memory for simple queries)
                         # but handles fallback if logic changes
                         dfs.append(pd.read_csv(item))
                         
            if dfs: result = pd.concat(dfs, ignore_index=True)

        # 4. Cleanup
        if not result.empty:
            if cols_to_return:
                available = [c for c in cols_to_return if c in result.columns]
                result = result[available]
            if max_rows:
                result = result.head(max_rows)

        status.update(label="✅ Processing Complete!", state="complete", expanded=False)
        
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        gc.collect()
        return result

def display_query_results(result_df, filename_prefix="results"):
    if not result_df.empty:
        st.success(f"✓ Found {len(result_df):,} rows")
        st.dataframe(result_df, use_container_width=True)
        st.download_button(
            "📥 Download Results",
            result_df.to_csv(index=False),
            f"{filename_prefix}_{int(time.time())}.csv",
            "text/csv"
        )
    else:
        st.info("No matching records found.")

# ============================================================================
# MAIN APP
# ============================================================================

st.set_page_config(page_title="CSV Query Tool v29", page_icon="🔎", layout="wide")
init_session_state()

st.title("🔎 Large CSV Query Tool")

with st.expander("📚 User Guide & Data Privacy Warning (Read First)", expanded=False):
    st.warning("🛡️ DATA PRIVACY: Do not upload unmasked PII to Streamlit Cloud.")
    st.markdown("""
    **Usage:**
    1. **Upload** CSVs in sidebar.
    2. **Load Columns** to analyze schema.
    3. **Query** via AI, SQL, or Builder.
    **Tip:** Use the 'Max Rows' limit to prevent crashes on large datasets. Set to 0 for unlimited.
    """)

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Setup")
    
    # --- AUTHENTICATION SECTION ---
    st.divider()
    if st.session_state['authenticated']:
        st.success("🔓 AI Features Unlocked")
        
        with st.expander("🤖 AI Settings", expanded=True):
            ai_provider = st.radio("Provider", ["OpenAI (GPT-4o)", "xAI (Grok)"])
            
            if "OpenAI" in ai_provider:
                api_key_name = "openai_api_key"
                base_url = None 
                model_name = "gpt-4o"
                price_in, price_out = 2.50, 10.00
            else:
                api_key_name = "xai_api_key"
                base_url = "https://api.x.ai/v1"
                model_name = "grok-2-1212" 
                price_in, price_out = 2.00, 10.00

            model_name = st.text_input("Model", value=model_name)
            api_key = st.secrets.get(api_key_name)
            
            # Save Config
            st.session_state['ai_provider_config'] = {
                'model_name': model_name,
                'api_key': api_key,
                'base_url': base_url,
                'price_in': price_in,
                'price_out': price_out
            }

        with st.expander("💰 Cost Estimator", expanded=False):
            c1, c2 = st.columns(2)
            c1.markdown(f"**Tokens:**\n{st.session_state['total_tokens']:,}")
            c2.markdown(f"**Cost:**\n`${st.session_state['total_cost']:.5f}`")
            if st.button("Reset Cost"):
                st.session_state['total_cost'] = 0.0
                st.session_state['total_tokens'] = 0
                st.rerun()
        
        if st.button("Logout"):
            logout()
            st.rerun()
    else:
        with st.expander("🔐 AI Login (Locked)", expanded=True):
            st.text_input("Password", type="password", key="password_input", on_change=perform_login)
            if st.session_state['auth_error']: st.error("Incorrect password.")

    # --- FILE UPLOAD SECTION ---
    st.divider()
    uploaded_files = st.file_uploader("Upload CSVs", accept_multiple_files=True, type="csv")
    
    if uploaded_files:
        existing_filenames = [Path(p).name.split('_', 1)[1] for p, _ in st.session_state['dataset_configs']]
        files_to_process = [f for f in uploaded_files if f.name not in existing_filenames]
        
        if files_to_process:
            new_data = []
            start_char = len(st.session_state['dataset_configs'])
            
            with st.status("📂 Processing Uploads...", expanded=True) as status:
                for i, f in enumerate(files_to_process):
                    status.write(f"💾 Saving {f.name} ({f.size / (1024*1024):.1f} MB)...")
                    if f.size / (1024*1024) > MAX_FILE_SIZE_MB:
                        st.error(f"❌ {f.name} too large (> {MAX_FILE_SIZE_MB}MB).")
                        continue
                    ds_name = chr(65 + start_char + i)
                    path = DOWNLOADS_DIR / f"{ds_name}_{f.name}"
                    path.write_bytes(f.getvalue())
                    new_data.append((str(path), ds_name))
                    time.sleep(0.5) 
                status.update(label="✅ Uploads Ready!", state="complete", expanded=False)
                time.sleep(1) 
            
            if new_data:
                st.session_state['dataset_configs'].extend(new_data)
                st.rerun()

    # --- DATASET LIST ---
    configs = st.session_state['dataset_configs']
    if configs:
        st.subheader("Loaded Datasets")
        for p, n in configs:
            st.text(f"{n}: {Path(p).name}")
        
        if st.button("Clear All"):
            st.session_state['dataset_configs'] = []
            st.session_state['columns_dict'] = {}
            st.rerun()
            
        if st.button("Load Columns", type="primary"):
            with st.spinner("Analyzing schema..."):
                cols, pre = get_columns_and_samples(tuple(tuple(x) for x in configs))
                st.session_state['columns_dict'] = cols
                st.session_state['preprocess_columns'] = pre
                st.rerun()

# ============================================================================
# MAIN CONTENT (UPDATED BUILDER)
# ============================================================================
cols_dict = st.session_state['columns_dict']

if not cols_dict:
    st.info("👈 Please upload files and click 'Load Columns' to start.")
else:
    tab1, tab2, tab3 = st.tabs(["🗣️ AI Query", "📝 SQL Query", "🔨 Builder"])
    
    # TAB 1: AI QUERY
    with tab1:
        st.write("Ask questions in plain English.")
        c1, c2 = st.columns([3, 1])
        nl_query = c1.text_area("Query:", placeholder="Show me datasets from A where Score > 50", height=100, key="ai_input")
        limit_ai = c2.number_input("Max Rows", min_value=0, value=5000, step=1000, key="limit_ai", help="0 = No Limit")
        
        if st.button("Run AI Query", type="primary", key="btn_ai"):
            if not st.session_state.get('authenticated'):
                st.error("🔒 Please log in via Sidebar to use AI features.")
            else:
                config = st.session_state['ai_provider_config']
                if not config.get('api_key'):
                    st.error("Missing API Key in secrets.")
                else:
                    with st.status("🤖 AI is thinking...", expanded=True):
                        datasets, join_cond, filter_cond = run_llm_parse(nl_query, cols_dict, config)
                    
                    if datasets:
                        final_limit = limit_ai if limit_ai > 0 else None
                        res = query_csvs(configs, datasets, join_cond, filter_cond, 
                                       st.session_state['preprocess_columns'], 
                                       get_safe_chunk_size(), final_limit, None, MAX_TEMP_STORAGE_MB)
                        display_query_results(res, "ai_result")
                    else:
                        st.error("AI could not interpret the query.")

    # TAB 2: SQL QUERY
    with tab2:
        c1, c2 = st.columns([3, 1])
        q_str = c1.text_area("SQL Query:", height=100, placeholder="SELECT * FROM A WHERE A.Val > 10", key="sql_input")
        limit_sql = c2.number_input("Max Rows", min_value=0, value=5000, step=1000, key="limit_sql", help="0 = No Limit")

        if st.button("Run SQL", key="btn_sql"):
            try:
                ds, join, filt, out = parse_query_string(q_str, cols_dict)
                final_limit = limit_sql if limit_sql > 0 else None
                res = query_csvs(configs, ds, join, filt, st.session_state['preprocess_columns'], 
                               get_safe_chunk_size(), final_limit, out, MAX_TEMP_STORAGE_MB)
                display_query_results(res, "sql_result")
            except Exception as e:
                st.error(f"Error: {e}")

    # TAB 3: BUILDER (UPDATED WITH JOIN VS STACK)
    with tab3:
        c1, c2 = st.columns(2)
        sel_ds = c1.multiselect("Datasets", list(cols_dict.keys()), key="builder_ds")
        limit = c2.number_input("Limit rows (0 = All)", min_value=0, max_value=None, value=10000, step=1000, key="builder_limit")
        
        if sel_ds:
            final_join_conds = []
            
            if len(sel_ds) > 1:
                st.divider()
                # Radio button to choose mode
                combine_mode = st.radio(
                    "How do you want to combine these datasets?",
                    ["🔗 Join (Merge columns on Key)", "📚 Stack (Append rows vertically)"],
                    key="combine_mode"
                )
                
                if "Join" in combine_mode:
                    st.markdown("##### 🔗 Join Settings")
                    # Compare adjacent datasets (A->B, B->C)
                    for i in range(len(sel_ds) - 1):
                        ds1, ds2 = sel_ds[i], sel_ds[i+1]
                        
                        # Find common columns
                        cols1 = {re.sub(r'\W+', '', c.lower()): c for c in cols_dict[ds1][0]}
                        cols2 = {re.sub(r'\W+', '', c.lower()): c for c in cols_dict[ds2][0]}
                        common_keys = list(set(cols1.keys()) & set(cols2.keys()))
                        
                        if common_keys:
                            common_names = [cols1[k] for k in common_keys]
                            selected_join_col = st.selectbox(
                                f"Join {ds1} and {ds2} on:", 
                                common_names, 
                                key=f"join_{ds1}_{ds2}"
                            )
                            key_norm = re.sub(r'\W+', '', selected_join_col.lower())
                            final_join_conds.append((f"{ds1}.{cols1[key_norm]}", f"{ds2}.{cols2[key_norm]}"))
                        else:
                            st.warning(f"⚠️ No common columns found between {ds1} and {ds2}.")
                            
                    if final_join_conds:
                        st.info(f"Using join: {'; '.join([f'{l}={r}' for l,r in final_join_conds])}")
                else:
                    # Stack Mode
                    st.info("Datasets will be stacked one after another.")
                    final_join_conds = [] # Ensure no join conditions are passed

            st.divider()
            filter_txt = st.text_area("Filters (one per line, e.g. 'A.Score > 50')", key="builder_filter")
            
            if st.button("Run Builder", key="btn_builder"):
                filt_conds = []
                for line in filter_txt.split('\n'):
                    if line.strip():
                        m = re.match(r'(\w+)\.(\w+)\s*([=!<>]+)\s*(.+)', line)
                        if m:
                            filt_conds.append({'dataset':m.group(1), 'column':m.group(2), 
                                             'operator':m.group(3), 'value':m.group(4)})
                
                final_limit = limit if limit > 0 else None
                
                res = query_csvs(configs, sel_ds, final_join_conds, 
                               filt_conds, st.session_state['preprocess_columns'],
                               get_safe_chunk_size(), final_limit, None, MAX_TEMP_STORAGE_MB)
                display_query_results(res, "builder_result")
