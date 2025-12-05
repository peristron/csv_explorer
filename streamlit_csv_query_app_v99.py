#
# run with: streamlit run streamlit_csv_query_app_v23.py      cd c:\users\oakhtar\documents\pyprojs_local
#
# CLOUD-READY VERSION (v23)
# - Removed hardcoded Windows paths
# - Added file size limits for uploads
# - Added memory-aware chunk sizing
# - Added Streamlit secrets support
# - Added result caching
# - Added timeout protection (Unix)
# - Optimized for Streamlit Community Cloud (works with files up to ~200MB)
# - Still works locally for larger files
#
# For large files (30GB+), run locally or deploy to a cloud VM
#
# setup instructions:
# 1. install libraries:
#    pip install pandas fuzzywuzzy python-Levenshtein numpy langchain langchain-core langchain-openai psutil streamlit requests
# 2. create .streamlit/secrets.toml with TOGETHER_API_KEY (optional)
# 3. run: streamlit run streamlit_csv_query_app_v23.py

import streamlit as st
import pandas as pd
import time
import os
import re
import numpy as np
from fuzzywuzzy import fuzz
import logging
import shutil
import psutil
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate  # FIXED: Updated import
import json
import requests
import traceback
import hashlib
from pathlib import Path

# ============================================================================
# CONFIGURATION - Adjust these for your deployment
# ============================================================================

# Maximum file size for uploads (in MB) - keep low for Streamlit Cloud
MAX_FILE_SIZE_MB = 200

# Default chunk size limits
MIN_CHUNK_SIZE = 10000
MAX_CHUNK_SIZE = 1000000
DEFAULT_CHUNK_SIZE = 100000

# Maximum temp storage (in MB)
DEFAULT_MAX_TEMP_STORAGE_MB = 1000  # Lower for cloud

# App directories (relative paths for cloud compatibility)
APP_DIR = Path(__file__).parent if "__file__" in dir() else Path(".")
DATA_DIR = APP_DIR / "data"
DOWNLOADS_DIR = APP_DIR / "downloads"
TEMP_DIR = APP_DIR / "temp_csv_query"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
DOWNLOADS_DIR.mkdir(exist_ok=True)

# ============================================================================
# LOGGING SETUP
# ============================================================================

# Basic logging for bad rows
logging.basicConfig(
    filename="bad_rows.log",
    level=logging.WARNING,
    format='%(asctime)s - %(message)s'
)

# Separate logging for streamlit debug
debug_logger = logging.getLogger('streamlit_debug')
debug_handler = logging.FileHandler('streamlit_debug.log')
debug_handler.setLevel(logging.DEBUG)
debug_formatter = logging.Formatter('%(asctime)s - %(message)s')
debug_handler.setFormatter(debug_formatter)
debug_logger.addHandler(debug_handler)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

column_cache = {}


def get_api_key():
    """Get API key from various sources with priority."""
    # Priority: 1. Session state, 2. Environment variable, 3. Streamlit secrets
    if 'api_key' in st.session_state and st.session_state['api_key']:
        return st.session_state['api_key']
    if os.getenv("TOGETHER_API_KEY"):
        return os.getenv("TOGETHER_API_KEY")
    try:
        if "TOGETHER_API_KEY" in st.secrets:
            return st.secrets["TOGETHER_API_KEY"]
    except Exception:
        pass
    return None


def validate_file_size(uploaded_file):
    """Check if uploaded file is within size limits."""
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"❌ File too large: {file_size_mb:.1f}MB. Maximum allowed: {MAX_FILE_SIZE_MB}MB")
        st.info(f"💡 For larger files, run this app locally or deploy to a cloud VM.")
        return False
    return True


def get_safe_chunk_size():
    """Dynamically set chunk size based on available memory."""
    try:
        available_memory = psutil.virtual_memory().available
        # Use ~10% of available memory per chunk
        bytes_per_row_estimate = 500  # Adjust based on your data
        safe_chunk = int((available_memory * 0.1) / bytes_per_row_estimate)
        return max(MIN_CHUNK_SIZE, min(safe_chunk, MAX_CHUNK_SIZE))
    except Exception:
        return DEFAULT_CHUNK_SIZE


def get_memory_usage():
    """Get current memory usage as a percentage."""
    try:
        return psutil.virtual_memory().percent
    except Exception:
        return 0


def check_memory_warning():
    """Display warning if memory usage is high."""
    usage = get_memory_usage()
    if usage > 85:
        st.warning(f"⚠️ High memory usage: {usage:.1f}%. Consider reducing chunk size or limiting results.")
        return True
    elif usage > 70:
        st.info(f"ℹ️ Memory usage: {usage:.1f}%")
    return False


def get_query_hash(*args):
    """Generate a hash for query parameters for caching."""
    return hashlib.md5(str(args).encode()).hexdigest()


def check_disk_space(path, required_mb):
    """Check if there's enough disk space."""
    try:
        disk = psutil.disk_usage(str(path))
        return disk.free / (1024 ** 2) > required_mb
    except Exception as e:
        st.warning(f"Could not check disk space: {e}")
        return True  # Assume OK if we can't check


def count_csv_rows(file_path):
    """Count rows in a CSV file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for _ in f) - 1
    except Exception as e:
        st.error(f"Error counting rows: {e}")
        return None


def get_file_size_mb(file_path):
    """Get file size in MB."""
    try:
        return os.path.getsize(file_path) / (1024 * 1024)
    except Exception:
        return 0


# ============================================================================
# COLUMN AND DATA FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)
def get_columns_and_samples(dataset_configs_tuple, preprocess_columns=None):
    """
    Get columns and sample values from datasets.
    Cached for 1 hour to improve performance.
    """
    dataset_configs = list(dataset_configs_tuple)  # Convert back from tuple for caching
    columns_dict = {}
    auto_preprocess = preprocess_columns if preprocess_columns else {}
    date_patterns = [
        r'^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}.*)?$',
        r'^\d{2}/\d{2}/\d{4}$',
        r'^\d{4}/\d{2}/\d{2}$',
    ]
    numeric_pattern = r'^-?\d*\.?\d*$'

    for file_path, dataset_name in dataset_configs:
        if file_path in column_cache:
            columns, sample_values = column_cache[file_path]
        else:
            try:
                df = pd.read_csv(file_path, nrows=100, engine='c', dtype=str, low_memory=False)
                columns = df.columns.tolist()
                sample_values = {}
                for col in columns:
                    non_null = df[col][~df[col].isna() & (df[col] != '')]
                    sample_values[col] = non_null.iloc[0] if not non_null.empty else 'NaN'
                    if f"{dataset_name}.{col}" not in auto_preprocess:
                        if any(col.lower().endswith(suffix) for suffix in ['date', 'time']) or \
                           any(re.match(pat, str(sample_values.get(col, '')), re.IGNORECASE) for pat in date_patterns):
                            auto_preprocess[f"{dataset_name}.{col}"] = 'datetime'
                        elif col.lower() in ['score', 'levelachievedid', 'orgunitid', 'userid', 'assessmentid'] or \
                             re.match(numeric_pattern, str(sample_values.get(col, '')), re.IGNORECASE):
                            auto_preprocess[f"{dataset_name}.{col}"] = 'numeric'
                column_cache[file_path] = (columns, sample_values)
            except Exception as e:
                st.error(f"Error reading columns from {file_path}: {e}")
                raise
        columns_dict[dataset_name] = (columns, sample_values)
    
    return columns_dict, auto_preprocess


def find_column(user_input, columns, threshold=70):
    """Find a column using fuzzy matching."""
    user_input = user_input.lower()
    if user_input in [col.lower() for col in columns]:
        for col in columns:
            if col.lower() == user_input:
                return col
    user_input_norm = user_input.replace(" ", "").replace("_", "")
    best_match = None
    best_score = 0
    for col in columns:
        col_norm = col.lower().replace(" ", "").replace("_", "")
        score = fuzz.token_sort_ratio(user_input_norm, col_norm)
        if score >= threshold and score > best_score:
            best_match = col
            best_score = score
    if best_match:
        st.info(f"Matched '{user_input}' to column '{best_match}' (score: {best_score})")
        return best_match
    raise ValueError(f"No column named '{user_input}' exists; choose from {columns}")


def find_operator(user_input, valid_operators, threshold=80):
    """Find an operator using fuzzy matching."""
    user_input = user_input.lower()
    if user_input in valid_operators:
        return user_input
    best_match = None
    best_score = 0
    for op in valid_operators:
        score = fuzz.token_sort_ratio(user_input, op)
        if score >= threshold and score > best_score:
            best_match = op
            best_score = score
    if best_match:
        st.info(f"Matched '{user_input}' to operator '{best_match}' (score: {best_score})")
        return best_match
    return None


# ============================================================================
# JOIN DETECTION
# ============================================================================

def detect_join_conditions(datasets, columns_dict, interactive=False):
    """Detect join conditions between datasets."""
    def normalize_name(name):
        return re.sub(r'\W+', '', str(name).lower())
    
    join_conditions = []
    if len(datasets) < 2:
        return join_conditions
    
    for i in range(len(datasets) - 1):
        ds1, ds2 = datasets[i], datasets[i + 1]
        cols1 = columns_dict[ds1][0]
        cols2 = columns_dict[ds2][0]
        norm_cols1 = {normalize_name(col): col for col in cols1}
        norm_cols2 = {normalize_name(col): col for col in cols2}
        common_norm_cols = set(norm_cols1.keys()).intersection(norm_cols2.keys())
        
        if not common_norm_cols:
            st.warning(f"No common columns between {ds1} and {ds2}.")
            if interactive:
                join_input = st.text_input(
                    f"Enter join condition (e.g., '{ds1}.Col1 = {ds2}.Col2' or 'none' to skip):",
                    key=f"join_input_{i}"
                )
                if join_input.lower() != 'none':
                    match = re.match(r'(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)', join_input, re.IGNORECASE)
                    if match:
                        ds1_input, col1, ds2_input, col2 = match.groups()
                        if ds1_input.upper() == ds1 and ds2_input.upper() == ds2:
                            try:
                                find_column(col1, cols1)
                                find_column(col2, cols2)
                                join_conditions.append((f"{ds1}.{col1}", f"{ds2}.{col2}"))
                            except ValueError as e:
                                st.error(f"Invalid join: {e}")
            continue
        
        if len(common_norm_cols) == 1:
            norm_col = common_norm_cols.pop()
            col1 = norm_cols1[norm_col]
            col2 = norm_cols2[norm_col]
            join_conditions.append((f"{ds1}.{col1}", f"{ds2}.{col2}"))
            st.info(f"Auto-detected join: {ds1}.{col1} = {ds2}.{col2}")
        elif len(common_norm_cols) > 1 and interactive:
            st.write(f"Multiple common columns between {ds1} and {ds2}: {[norm_cols1[n] for n in common_norm_cols]}")
            choice = st.selectbox(
                "Select a join column:",
                options=[norm_cols1[n] for n in common_norm_cols] + ["none", "custom"],
                key=f"join_choice_{i}"
            )
            if choice == "none":
                continue
            elif choice == "custom":
                join_input = st.text_input(
                    f"Enter join condition (e.g., '{ds1}.Col1 = {ds2}.Col2'):",
                    key=f"custom_join_{i}"
                )
                match = re.match(r'(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)', join_input, re.IGNORECASE)
                if match:
                    ds1_input, col1, ds2_input, col2 = match.groups()
                    if ds1_input.upper() == ds1 and ds2_input.upper() == ds2:
                        try:
                            find_column(col1, cols1)
                            find_column(col2, cols2)
                            join_conditions.append((f"{ds1}.{col1}", f"{ds2}.{col2}"))
                        except ValueError as e:
                            st.error(f"Error in join condition: {e}")
            else:
                col1 = choice
                col2 = norm_cols2[normalize_name(choice)]
                join_conditions.append((f"{ds1}.{col1}", f"{ds2}.{col2}"))
                st.info(f"Selected join: {ds1}.{col1} = {ds2}.{col2}")
        elif not interactive and common_norm_cols:
            norm_col = common_norm_cols.pop()
            col1 = norm_cols1[norm_col]
            col2 = norm_cols2[norm_col]
            join_conditions.append((f"{ds1}.{col1}", f"{ds2}.{col2}"))
            st.info(f"Auto-detected join (non-interactive): {ds1}.{col1} = {ds2}.{col2}")
    
    return join_conditions


# ============================================================================
# QUERY PARSING
# ============================================================================

def parse_query_string(query_str, columns_dict):
    """Parse a SQL-like query string."""
    output_columns = []
    datasets = []
    join_conditions = []
    filter_conditions = []
    valid_operators = ['==', '!=', '<', '>', 'contains', 'num_contains', 'like']

    select_pattern = re.compile(r'SELECT\s+(.+?)\s+FROM', re.IGNORECASE | re.DOTALL)
    from_pattern = re.compile(r'FROM\s+(\w+)', re.IGNORECASE)
    join_pattern = re.compile(r'JOIN\s+(\w+)\s+ON\s+([\w\.]+)\s*=\s*([\w\.]+)', re.IGNORECASE)
    where_pattern = re.compile(r'WHERE\s+(.+)', re.IGNORECASE)

    select_match = select_pattern.search(query_str)
    
    if select_match:
        column_str = select_match.group(1).replace('\n', '').strip()
        output_columns = [col.strip() for col in column_str.split(',')]
        if output_columns == ['*']:
            output_columns = []

    from_match = from_pattern.search(query_str)
    if from_match:
        datasets.append(from_match.group(1).upper())
    
    for join_match in join_pattern.finditer(query_str):
        datasets.append(join_match.group(1).upper())
        left_on = join_match.group(2)
        right_on = join_match.group(3)
        join_conditions.append((left_on, right_on))

    if not datasets:
        raise ValueError("Invalid query: missing FROM clause")
    
    if not join_conditions and len(datasets) > 1:
        join_conditions = detect_join_conditions(datasets, columns_dict)

    where_match = where_pattern.search(query_str)
    if where_match:
        where_clause = where_match.group(1)
        query_parts = [part.strip() for part in re.split(r'\band\b', where_clause, flags=re.IGNORECASE)]
        for part in query_parts:
            match = re.match(r'(\w+)\.(\w+)\s*([=!<>]+|contains|num_contains|like)\s*(.+)', part, re.IGNORECASE)
            if not match:
                raise ValueError(f"Invalid query syntax: '{part}'. Expected format: 'dataset.column operator value'")
            
            ds, col, op, val = match.groups()
            ds_upper = ds.upper()
            if ds_upper not in columns_dict:
                raise ValueError(f"Unknown dataset: {ds_upper}")
            
            mapped_col = find_column(col, columns_dict[ds_upper][0])
            val = val.strip().strip("'\"")

            op_lower = op.lower()
            if val.lower() == 'nan':
                val = 'NaN'
            elif val.lower() in ['true', 'false']:
                val = val.lower() == 'true'
            elif op_lower not in ['like', 'contains', 'num_contains'] and re.match(r'^-?\d+\.?\d*$', val):
                try:
                    val = float(val) if '.' in val else int(val)
                except ValueError:
                    pass

            filter_conditions.append({
                'dataset': ds_upper,
                'column': mapped_col,
                'operator': op_lower,
                'value': val
            })
    
    return datasets, join_conditions, filter_conditions, output_columns


def parse_natural_language_query(nl_query, columns_dict):
    """Parse a natural language query using LLM."""
    valid_operators = ['==', '!=', '<', '>', 'contains', 'num_contains', 'like']
    
    api_key = get_api_key()
    if not api_key:
        st.error("❌ No API key found. Set TOGETHER_API_KEY in secrets or enter it in the sidebar.")
        return [], [], []
    
    try:
        llm = ChatOpenAI(
            model="meta-llama/Llama-3-8b-chat-hf",
            api_key=api_key,
            base_url="https://api.together.xyz/v1",
            temperature=0
        )
        
        columns_info = {ds: cols[0] for ds, cols in columns_dict.items()}
        prompt_template = PromptTemplate(
            input_variables=["query", "columns_info", "operators"],
            template="""Parse the natural language query: "{query}" into a structured JSON object with:
- datasets: List of dataset names (e.g., ["A", "B"]).
- join_conditions: List of join pairs (e.g., [["A.UserId", "B.UserId"]]).
- filter_conditions: List of filters (e.g., [{{"dataset": "A", "column": "Score", "operator": ">", "value": 4}}]).
Available datasets and columns: {columns_info}.
Valid operators: {operators}.
For text columns (e.g., Feedback), use 'contains' or 'like'.
For numeric columns (e.g., UserId), use 'num_contains', 'like', or numeric operators (==, >, <, !=).
Example input: "Join A and B on UserId where A.Score is above 4 and B.Feedback contains excellent"
Example output: {{{{"datasets": ["A", "B"], "join_conditions": [["A.UserId", "B.UserId"]], "filter_conditions": [{{"dataset": "A", "column": "Score", "operator": ">", "value": 4}}, {{"dataset": "B", "column": "Feedback", "operator": "contains", "value": "excellent"}}]}}}}
If the query cannot be parsed or columns/operators are invalid, return {{{{"datasets": [], "join_conditions": [], "filter_conditions": []}}}}.
Ensure column names match exactly (case-sensitive)."""
        )
        chain = prompt_template | llm
        response = chain.invoke({
            "query": nl_query,
            "columns_info": columns_info,
            "operators": valid_operators
        })
        parsed = json.loads(response.content.strip() if hasattr(response, "content") else response.strip())
        
        datasets = parsed.get('datasets', [])
        join_conditions = parsed.get('join_conditions', [])
        filter_conditions = parsed.get('filter_conditions', [])
        
        if not join_conditions and len(datasets) > 1:
            join_conditions = detect_join_conditions(datasets, columns_dict)
        
        validated_filter_conditions = []
        for cond in filter_conditions:
            try:
                ds = cond['dataset']
                if ds not in columns_dict:
                    raise ValueError(f"Unknown dataset: {ds}")
                col = find_column(cond['column'], columns_dict[ds][0])
                op = cond['operator']
                val = cond['value']
                if op not in valid_operators:
                    suggested_op = find_operator(op, valid_operators)
                    if suggested_op:
                        st.warning(f"Operator '{op}' invalid; using '{suggested_op}'")
                        op = suggested_op
                    else:
                        raise ValueError(f"Unsupported operator: {op}")
                if isinstance(val, str):
                    if val.lower() == 'nan':
                        val = 'NaN'
                    elif val.lower() in ['true', 'false']:
                        val = val.lower() == 'true'
                    elif val.isdigit():
                        val = int(val)
                    elif re.match(r'^-?\d+\.?\d*$', val):
                        try:
                            val = float(val)
                        except ValueError:
                            pass
                if op == 'like' and isinstance(val, str) and '%' not in val:
                    val = f'%{val}%'
                    st.info(f"Normalized 'like' to '{val}'")
                validated_filter_conditions.append({
                    'dataset': ds,
                    'column': col,
                    'operator': op,
                    'value': val
                })
            except ValueError as e:
                st.error(f"Error in parsed condition {cond}: {e}")
                continue
        
        validated_join_conditions = []
        for left_col, right_col in join_conditions:
            try:
                left_ds, left_col_name = left_col.split('.')
                right_ds, right_col_name = right_col.split('.')
                if left_ds not in columns_dict or right_ds not in columns_dict:
                    raise ValueError(f"Invalid datasets in join: {left_ds}, {right_ds}")
                find_column(left_col_name, columns_dict[left_ds][0])
                find_column(right_col_name, columns_dict[right_ds][0])
                validated_join_conditions.append((left_col, right_col))
            except ValueError as e:
                st.error(f"Error in join {left_col} = {right_col}: {e}")
                continue
        
        return datasets, validated_join_conditions, validated_filter_conditions
    
    except Exception as e:
        st.error(f"Error parsing natural language query: {e}")
        debug_logger.error(f"NL parse error: {e}\n{traceback.format_exc()}")
        return [], [], []


# ============================================================================
# MAIN QUERY ENGINE
# ============================================================================

def query_multiple_large_csvs(
    dataset_configs,
    datasets,
    join_conditions,
    filter_conditions,
    preprocess_columns=None,
    chunk_size=100000,
    max_rows_to_return=None,
    output_columns=None,
    max_temp_storage_mb=1000
):
    """Execute query on large CSV files with chunking."""
    preprocess_columns = preprocess_columns or {}
    path_map = {name: path for path, name in dataset_configs}
    columns_dict = st.session_state.get('columns_dict', {})
    
    # Use relative temp directory
    temp_dir = TEMP_DIR
    temp_dir.mkdir(exist_ok=True)
    
    filtered_chunks = {ds: [] for ds in datasets}
    total_rows = 0
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Check disk space
        if not check_disk_space(temp_dir.parent, max_temp_storage_mb):
            raise ValueError(f"Insufficient disk space; need {max_temp_storage_mb}MB")

        # Check memory at start
        check_memory_warning()

        if not output_columns:
            output_columns = []
            for cond in filter_conditions:
                output_columns.append(f"{cond['dataset']}.{cond['column']}")
            if not output_columns and datasets:
                first_ds = datasets[0]
                output_columns = [f"{first_ds}.{col}" for col in columns_dict.get(first_ds, ([], ))[0]]
        
        required_cols = {ds: set() for ds in datasets}
        for cond in filter_conditions:
            if cond['dataset'] in datasets:
                required_cols[cond['dataset']].add(cond['column'])
        for left_col, right_col in join_conditions:
            left_ds, left_col_name = left_col.split('.')
            right_ds, right_col_name = right_col.split('.')
            required_cols[left_ds].add(left_col_name)
            required_cols[right_ds].add(right_col_name)
        for col in output_columns:
            ds, col_name = col.split('.')
            if ds in required_cols:
                required_cols[ds].add(col_name)
        
        dtypes = {ds: {col: 'object' for col in cols} for ds, cols in required_cols.items()}
        
        def preprocess_chunk(chunk, dataset_name):
            for col_spec, preprocess_type in preprocess_columns.items():
                ds, col_name = col_spec.split('.')
                if ds == dataset_name and col_name in chunk.columns:
                    if preprocess_type == 'datetime':
                        chunk[col_name] = pd.to_datetime(chunk[col_name], errors='coerce')
                    elif preprocess_type == 'numeric':
                        chunk[col_name] = pd.to_numeric(chunk[col_name], errors='coerce')
            return chunk

        def apply_filters(chunk, conditions, dataset_name):
            final_mask = pd.Series(True, index=chunk.index)
            relevant_conditions = [c for c in conditions if c['dataset'] == dataset_name]
            if not relevant_conditions:
                return final_mask
            
            for cond in relevant_conditions:
                col, op, val = cond['column'], cond['operator'], cond['value']
                if col not in chunk.columns:
                    continue
                if pd.isna(val) or str(val).lower() == 'nan':
                    condition_mask = chunk[col].isna() if op == '==' else ~chunk[col].isna()
                else:
                    col_series = chunk[col]
                    col_preprocess_type = preprocess_columns.get(f"{dataset_name}.{col}")
                    if col_preprocess_type == 'numeric':
                        col_series = pd.to_numeric(col_series, errors='coerce')
                    elif col_preprocess_type == 'datetime':
                        col_series = pd.to_datetime(col_series, errors='coerce')
                    
                    if op == '==':
                        if pd.api.types.is_string_dtype(col_series) or isinstance(col_series.dtype, object):
                            condition_mask = col_series.str.lower() == str(val).lower()
                        else:
                            condition_mask = col_series == val
                    elif op == '!=':
                        condition_mask = col_series != val
                    elif op == '<':
                        condition_mask = col_series < val
                    elif op == '>':
                        condition_mask = col_series > val
                    elif op in ['contains', 'num_contains']:
                        condition_mask = col_series.astype(str).str.contains(str(val), case=False, na=False)
                    elif op == 'like':
                        condition_mask = col_series.astype(str).str.contains(
                            re.escape(str(val).strip('%')), case=False, regex=True, na=False
                        )
                    else:
                        raise ValueError(f"Unsupported operator: {op}")
                final_mask &= condition_mask
            return final_mask

        filtering_start = time.time()
        total_datasets_to_process = len(datasets)
        
        for i, dataset_name in enumerate(datasets):
            file_path = path_map.get(dataset_name)
            if not file_path:
                continue
            
            status_text.text(f"Filtering dataset {dataset_name} ({i + 1}/{total_datasets_to_process})...")
            use_cols = list(required_cols.get(dataset_name, [])) or None
            
            chunk_iterator = pd.read_csv(
                file_path,
                chunksize=chunk_size,
                usecols=use_cols,
                dtype=dtypes.get(dataset_name, 'object'),
                engine='c',
                low_memory=False,
                on_bad_lines='warn'
            )
            
            for chunk_idx, chunk in enumerate(chunk_iterator):
                # Check memory periodically
                if chunk_idx % 10 == 0:
                    check_memory_warning()
                
                chunk = preprocess_chunk(chunk, dataset_name)
                mask = apply_filters(chunk, filter_conditions, dataset_name)
                
                if mask.any():
                    filtered_chunk = chunk[mask]
                    if join_conditions:
                        temp_file = temp_dir / f'{dataset_name}_chunk_{chunk_idx}.csv'
                        filtered_chunk.to_csv(temp_file, index=False)
                        filtered_chunks[dataset_name].append(str(temp_file))
                    else:
                        raw_output_cols = [col.split('.')[1] for col in output_columns if col.split('.')[0] == dataset_name]
                        final_chunk = filtered_chunk[[c for c in raw_output_cols if c in filtered_chunk.columns]]
                        final_chunk.columns = [f"{dataset_name}.{col}" for col in final_chunk.columns]
                        filtered_chunks[dataset_name].append(final_chunk)
                        total_rows += len(final_chunk)
                    
                    if max_rows_to_return and total_rows >= max_rows_to_return:
                        break
            
            if max_rows_to_return and total_rows >= max_rows_to_return:
                break
        
        filtering_time = time.time() - filtering_start

        merging_start = time.time()
        result_chunks = []
        
        if join_conditions:
            status_text.text("Joining filtered data...")
            chunk_sets = [filtered_chunks[ds] for ds in datasets]
            if not all(chunk_sets):
                st.warning("Some datasets had no matching rows after filtering.")
                return pd.DataFrame()

            total_join_chunks = min(len(chunks) for chunks in chunk_sets) if chunk_sets else 0
            for chunk_idx, chunk_files in enumerate(zip(*chunk_sets)):
                status_text.text(f"Processing join chunk {chunk_idx + 1}/{total_join_chunks}")
                dfs = [pd.read_csv(f) for f in chunk_files]
                merged = dfs[0].copy()
                merged.columns = [f"{datasets[0]}.{col}" for col in merged.columns]
                
                for i, (left_col, right_col) in enumerate(join_conditions):
                    right_df = dfs[i + 1]
                    right_df.columns = [f"{datasets[i + 1]}.{col}" for col in right_df.columns]
                    merged = merged.merge(right_df, left_on=left_col, right_on=right_col, how='inner')
                
                if not merged.empty:
                    valid_output_columns = [col for col in output_columns if col in merged.columns]
                    result_chunks.append(merged[valid_output_columns])
                    total_rows += len(merged)
                    if max_rows_to_return and total_rows >= max_rows_to_return:
                        break
                
                progress_bar.progress((chunk_idx + 1) / total_join_chunks)
        else:
            status_text.text("Concatenating results...")
            for ds in datasets:
                result_chunks.extend(filtered_chunks[ds])
        
        if result_chunks:
            result = pd.concat(result_chunks, ignore_index=True)
            if max_rows_to_return:
                result = result.head(max_rows_to_return)
        else:
            result = pd.DataFrame()

        merging_time = time.time() - merging_start

        st.info(f"⏱️ Filtering time: {filtering_time:.2f}s | Merging time: {merging_time:.2f}s")

        if not result.empty:
            return result
        else:
            st.info("No matching rows found.")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error during query execution: {str(e)}")
        st.code(traceback.format_exc())
        return pd.DataFrame()
    
    finally:
        # Cleanup temp files
        shutil.rmtree(temp_dir, ignore_errors=True)
        progress_bar.empty()
        status_text.empty()


# ============================================================================
# INTERACTIVE QUERY BUILDER
# ============================================================================

def build_query_interactively(columns_dict, dataset_configs):
    """Build a query interactively using UI elements."""
    if 'interactive_state' not in st.session_state:
        st.session_state['interactive_state'] = {
            'datasets': [],
            'join_conditions': [],
            'filter_conditions': [],
            'output_columns': [],
            'max_rows_to_return': 10,
            'debug': False
        }
    
    state = st.session_state['interactive_state']
    
    st.write("**Building query interactively**")
    st.caption("Available operators: `==`, `!=`, `<`, `>`, `contains` (text), `num_contains` (numeric), `like` (pattern)")
    
    with st.form(key="interactive_query_form"):
        debug = st.checkbox("Enable debug logging", value=state['debug'], key="interactive_debug")
        state['debug'] = debug
        if debug:
            st.json(state)
        
        datasets = st.multiselect(
            "Select datasets:",
            options=list(columns_dict.keys()),
            default=state['datasets'],
            key="interactive_datasets"
        )
        state['datasets'] = datasets
        
        if not datasets:
            if st.form_submit_button("Run interactive query"):
                st.error("Select at least one dataset to continue")
            return [], [], [], None, []
        
        if len(datasets) > 1:
            with st.expander("Configure joins (if needed)"):
                state['join_conditions'] = detect_join_conditions(datasets, columns_dict, interactive=True)
        else:
            state['join_conditions'] = []
        
        with st.expander("Add filter conditions"):
            filter_input = st.text_area(
                "Enter conditions (e.g., 'A.Source == Brightspace'), one per line:",
                value='\n'.join([f"{c['dataset']}.{c['column']} {c['operator']} {c['value']}" for c in state['filter_conditions']]),
                key="interactive_filters"
            )
            valid_operators = ['==', '!=', '<', '>', 'contains', 'num_contains', 'like']
            new_filter_conditions = []
            
            for cond_input in filter_input.split('\n'):
                cond_input = cond_input.strip().replace("'", "").replace('"', "")
                if not cond_input:
                    continue
                
                match = re.match(r'(\w+)\.(\w+)\s*([=!<>]+|contains|num_contains|like)\s*(.+)', cond_input, re.IGNORECASE)
                ds, col, op, val = (None,) * 4
                
                if match:
                    ds, col, op, val = match.groups()
                    ds = ds.upper()
                elif len(datasets) == 1:
                    match = re.match(r'(\w+)\s*([=!<>]+|contains|num_contains|like)\s*(.+)', cond_input, re.IGNORECASE)
                    if match:
                        col, op, val = match.groups()
                        ds = datasets[0]
                
                if not match:
                    st.error(f"Invalid format: {cond_input}")
                    continue
                
                if ds not in datasets:
                    st.error(f"Invalid dataset: {ds}")
                    continue
                
                try:
                    mapped_col = find_column(col, columns_dict[ds][0])
                    op_lower = op.lower()
                    
                    if op_lower not in valid_operators:
                        suggested_op = find_operator(op_lower, valid_operators)
                        if suggested_op:
                            st.info(f"Using '{suggested_op}' instead of '{op_lower}'")
                            op_lower = suggested_op
                        else:
                            continue
                    
                    val = val.strip()
                    if val.lower() == 'nan':
                        val = 'NaN'
                    elif val.lower() in ['true', 'false']:
                        val = val.lower() == 'true'
                    elif op_lower not in ['like', 'contains', 'num_contains'] and re.match(r'^-?\d+\.?\d*$', val):
                        val = float(val) if '.' in val else int(val)
                    
                    if op_lower == 'like' and isinstance(val, str) and '%' not in val:
                        val = f'%{val}%'
                        st.info(f"Normalized 'like' to '{val}'")
                    
                    new_filter_conditions.append({
                        'dataset': ds,
                        'column': mapped_col,
                        'operator': op_lower,
                        'value': val
                    })
                except ValueError as e:
                    st.error(f"Error: {e}")
            
            state['filter_conditions'] = new_filter_conditions
        
        with st.expander("Select output columns"):
            temp_output_columns = []
            for ds in datasets:
                default_cols = [c.split('.')[1] for c in state.get('output_columns', []) if c.split('.')[0] == ds]
                cols = st.multiselect(
                    f"Output columns for {ds}:",
                    options=columns_dict[ds][0],
                    default=default_cols,
                    key=f"interactive_output_{ds}"
                )
                temp_output_columns.extend([f"{ds}.{col}" for col in cols])
            
            if not temp_output_columns and new_filter_conditions:
                temp_output_columns.extend([f"{c['dataset']}.{c['column']}" for c in new_filter_conditions])
            
            state['output_columns'] = temp_output_columns
        
        max_rows_val = st.number_input(
            "Max rows to return (0 for all):",
            min_value=0,
            value=state.get('max_rows_to_return', 10) or 10,
            key="interactive_max_rows"
        )
        state['max_rows_to_return'] = max_rows_val if max_rows_val > 0 else None
        
        submitted = st.form_submit_button("🔍 Run interactive query")
        
        if submitted:
            if len(state['datasets']) > 1 and not state['join_conditions']:
                st.error("Multiple datasets selected but no join configured. Please define a join condition.")
                return [], [], [], None, []
            return (
                state['datasets'],
                state['join_conditions'],
                state['filter_conditions'],
                state['max_rows_to_return'],
                state['output_columns']
            )
    
    if st.button("🔄 Reset query state", key="reset_interactive"):
        st.session_state.pop('interactive_state', None)
        st.rerun()
    
    return [], [], [], None, []


# ============================================================================
# STREAMLIT MAIN APP
# ============================================================================

st.set_page_config(
    page_title="CSV Query Tool (v23)",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Large CSV Query Tool (v23)")
st.caption("Query CSV files using SQL-like syntax, natural language, or interactive builder")

# Display system info
with st.expander("ℹ️ System Information", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Memory Usage", f"{get_memory_usage():.1f}%")
    with col2:
        st.metric("Safe Chunk Size", f"{get_safe_chunk_size():,} rows")
    with col3:
        st.metric("Max Upload Size", f"{MAX_FILE_SIZE_MB} MB")

# ============================================================================
# SIDEBAR - CONFIGURATION
# ============================================================================

st.sidebar.header("⚙️ Configuration")

# API Key
st.sidebar.subheader("🔑 API Settings")
api_key_input = st.sidebar.text_input(
    "Together AI API Key:",
    type="password",
    help="Required for natural language queries. Get one at https://api.together.xyz"
)
if api_key_input:
    st.session_state['api_key'] = api_key_input
    os.environ["TOGETHER_API_KEY"] = api_key_input

if get_api_key():
    st.sidebar.success("✓ API key configured")
else:
    st.sidebar.warning("⚠ No API key (NL queries disabled)")

# Dataset Configuration
st.sidebar.subheader("📁 Dataset Configuration")

# Initialize dataset_configs from session state
if 'dataset_configs' not in st.session_state:
    st.session_state['dataset_configs'] = []

dataset_configs = st.session_state['dataset_configs']

# Option 1: Upload files
with st.sidebar.expander("📤 Upload CSV Files", expanded=True):
    uploaded_files = st.file_uploader(
        "Upload CSV files:",
        type="csv",
        accept_multiple_files=True,
        help=f"Max size per file: {MAX_FILE_SIZE_MB}MB"
    )
    
    if st.button("Process Uploads") and uploaded_files:
        DOWNLOADS_DIR.mkdir(exist_ok=True)
        new_configs = []
        
        for uploaded_file in uploaded_files:
            if not validate_file_size(uploaded_file):
                continue
            
            # Assign dataset letter
            existing_letters = set(config[1] for config in dataset_configs)
            next_letter = chr(65 + len(existing_letters))
            while next_letter in existing_letters:
                next_letter = chr(ord(next_letter) + 1)
            
            file_path = DOWNLOADS_DIR / f"{next_letter}_{uploaded_file.name}"
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            new_configs.append((str(file_path), next_letter))
            st.success(f"✓ Uploaded {uploaded_file.name} as Dataset {next_letter}")
        
        dataset_configs.extend(new_configs)
        st.session_state['dataset_configs'] = dataset_configs
        st.rerun()

# Option 2: Specify local paths (for local development)
with st.sidebar.expander("📂 Local File Paths", expanded=False):
    st.caption("For local development only - specify paths to existing CSV files")
    
    num_datasets = st.number_input("Number of datasets:", min_value=0, max_value=10, value=0)
    
    local_configs = []
    for i in range(num_datasets):
        path = st.text_input(f"Path for Dataset {chr(65 + i)}:", key=f"local_path_{i}")
        if path and os.path.exists(path):
            local_configs.append((path, chr(65 + i)))
        elif path:
            st.warning(f"Path not found: {path}")
    
    if st.button("Use Local Paths") and local_configs:
        dataset_configs = local_configs
        st.session_state['dataset_configs'] = dataset_configs
        st.success(f"Loaded {len(local_configs)} local datasets")
        st.rerun()

# Option 3: Auto-download (advanced)
with st.sidebar.expander("🌐 Auto-Download (Advanced)", expanded=False):
    st.caption("Download CSVs directly from authenticated URLs")
    
    urls_input = st.text_area(
        "Enter URLs (one per line):",
        placeholder="https://example.com/data.csv"
    )
    urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
    
    cookies = st.text_input(
        "Authenticated cookies (from browser dev tools):",
        type="password"
    )
    
    if st.button("🔽 Auto-Download CSVs") and urls and cookies:
        DOWNLOADS_DIR.mkdir(exist_ok=True)
        
        for i, url in enumerate(urls):
            dataset_name = chr(65 + len(dataset_configs) + i)
            try:
                with st.spinner(f"Downloading {url}..."):
                    response = requests.get(
                        url,
                        headers={'Cookie': cookies, 'User-Agent': 'Mozilla/5.0'},
                        timeout=60
                    )
                
                if response.status_code == 200 and 'text/csv' in response.headers.get('Content-Type', ''):
                    file_path = DOWNLOADS_DIR / f"{dataset_name}.csv"
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    dataset_configs.append((str(file_path), dataset_name))
                    st.success(f"✓ Downloaded {dataset_name} ({len(response.content) / 1024 / 1024:.1f}MB)")
                else:
                    st.error(f"✗ Failed: {url} (status: {response.status_code})")
            except Exception as e:
                st.error(f"✗ Error downloading {url}: {e}")
        
        st.session_state['dataset_configs'] = dataset_configs
        st.rerun()

# Show current datasets
st.sidebar.subheader("📊 Current Datasets")
if dataset_configs:
    for path, name in dataset_configs:
        file_size = get_file_size_mb(path)
        st.sidebar.text(f"{name}: {os.path.basename(path)} ({file_size:.1f}MB)")
    
    if st.sidebar.button("🗑️ Clear All Datasets"):
        st.session_state['dataset_configs'] = []
        st.session_state.pop('columns_dict', None)
        st.session_state.pop('preprocess_columns', None)
        st.rerun()
else:
    st.sidebar.info("No datasets loaded. Upload or specify files above.")

# Processing settings
st.sidebar.subheader("⚡ Processing Settings")
chunk_size = st.sidebar.number_input(
    "Chunk size (rows):",
    min_value=MIN_CHUNK_SIZE,
    max_value=MAX_CHUNK_SIZE,
    value=get_safe_chunk_size(),
    help="Rows processed at a time. Lower = less memory, slower."
)

max_temp_storage_mb = st.sidebar.number_input(
    "Max temp storage (MB):",
    min_value=100,
    max_value=10000,
    value=DEFAULT_MAX_TEMP_STORAGE_MB
)

count_rows = st.sidebar.checkbox(
    "Count rows in datasets",
    help="Can take time for large files"
)

# Load Columns Button
if st.sidebar.button("📋 Load Columns", type="primary"):
    if not dataset_configs:
        st.sidebar.error("No datasets configured")
    else:
        with st.spinner("Loading columns and samples..."):
            try:
                # Convert to tuple for caching
                columns_dict, preprocess_columns = get_columns_and_samples(
                    tuple(tuple(x) for x in dataset_configs)
                )
                st.session_state['columns_dict'] = columns_dict
                st.session_state['preprocess_columns'] = preprocess_columns
                
                if count_rows:
                    st.sidebar.write("**Row Counts:**")
                    for file_path, ds in dataset_configs:
                        rows = count_csv_rows(file_path)
                        if rows is not None:
                            st.sidebar.text(f"{ds}: {rows:,} rows")
                        else:
                            st.sidebar.text(f"{ds}: Error counting")
                
                st.sidebar.success("✓ Columns loaded!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

columns_dict = st.session_state.get('columns_dict', {})
preprocess_columns = st.session_state.get('preprocess_columns', {})

if not columns_dict:
    st.warning("👈 Load columns first using the sidebar")
    
    # Show sample query examples
    st.subheader("📖 Quick Start Guide")
    
    st.markdown("""
    ### Steps to use this tool:
    
    1. **Upload CSV files** using the sidebar (or specify local paths)
    2. Click **Load Columns** to analyze your data
    3. Choose a query method below
    
    ### Query Examples:
    
    **Query String (SQL-like):**
    ```sql
    SELECT A.Name, A.Score FROM A WHERE A.Score > 90
    ```
    
    **Join Query:**
    ```sql
    SELECT A.Name, B.Department 
    FROM A JOIN B ON A.UserId = B.UserId 
    WHERE A.Score > 50
    ```
    
    **Natural Language:**
    ```
    Show me all records from dataset A where score is greater than 90
    ```
    """)
    
else:
    # Show loaded column info
    with st.expander("📋 View Loaded Column Samples", expanded=False):
        for dataset_name, (columns, sample_values) in columns_dict.items():
            st.write(f"**Dataset {dataset_name}** ({len(columns)} columns)")
            df_samples = pd.DataFrame.from_dict(sample_values, orient='index', columns=['Sample Value'])
            st.dataframe(df_samples, use_container_width=True)
    
    with st.expander("🔧 View Auto-Detected Column Types", expanded=False):
        st.json(preprocess_columns)
    
    # Query Tabs
    tab1, tab2, tab3 = st.tabs(["🗣️ Natural Language", "📝 Query String", "🔨 Interactive Builder"])

    # -------------------------------------------------------------------------
    # TAB 1: Natural Language Query
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("Natural Language Query")
        st.caption("Describe your query in plain English (requires API key)")
        
        if not get_api_key():
            st.warning("⚠️ Set your Together AI API key in the sidebar to use natural language queries.")
        
        nl_query = st.text_area(
            "Enter your query:",
            placeholder="e.g., Join A and B on UserId where A.Score is greater than 4",
            key="nl_query_input"
        )
        
        col1, col2 = st.columns([1, 2])
        with col1:
            max_rows_nl = st.number_input("Max rows:", value=10, min_value=1, key="nl_max_rows")
        
        if st.button("🔍 Run Natural Language Query", type="primary"):
            if not nl_query:
                st.warning("Please enter a query")
            elif not get_api_key():
                st.error("API key required for natural language queries")
            else:
                with st.spinner("Parsing query with AI..."):
                    datasets, join_conditions, filter_conditions = parse_natural_language_query(nl_query, columns_dict)
                
                if datasets:
                    all_possible_cols = [f"{ds}.{col}" for ds in datasets for col in columns_dict.get(ds, ([], ))[0]]
                    output_columns = st.multiselect(
                        "Select output columns:",
                        options=all_possible_cols,
                        default=all_possible_cols[:5],
                        key="nl_output_cols"
                    )
                    
                    with st.spinner("Executing query..."):
                        result = query_multiple_large_csvs(
                            dataset_configs, datasets, join_conditions, filter_conditions,
                            preprocess_columns, chunk_size, max_rows_nl, output_columns, max_temp_storage_mb
                        )
                    
                    if not result.empty:
                        st.success(f"✓ Found {len(result):,} rows")
                        st.dataframe(result, use_container_width=True)
                        st.download_button(
                            "📥 Download Results",
                            result.to_csv(index=False),
                            "nl_results.csv",
                            "text/csv",
                            key="download_nl"
                        )
                else:
                    st.error("Could not parse query. Try the Query String tab instead.")

    # -------------------------------------------------------------------------
    # TAB 2: Query String
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("Query String (SQL-like)")
        st.caption("Write queries using SQL-like syntax")
        
        # Query examples
        with st.expander("📖 Query Examples"):
            st.code("""
# Simple query
SELECT A.Name, A.Score FROM A WHERE A.Score > 90

# With contains
SELECT A.* FROM A WHERE A.Name contains 'John'

# Join query
SELECT A.Name, B.Department 
FROM A JOIN B ON A.UserId = B.UserId 
WHERE A.Score > 50

# Multiple conditions
SELECT A.* FROM A WHERE A.Score > 50 AND A.Status == Active
            """, language="sql")
        
        query_str = st.text_area(
            "Enter your query:",
            placeholder="SELECT A.Name, A.Score FROM A WHERE A.Score > 90",
            key="query_string_input",
            height=150
        )
        
        col1, col2 = st.columns([1, 2])
        with col1:
            max_rows_qs = st.number_input(
                "Max rows (0 for all):",
                min_value=0,
                value=10,
                key="query_string_max_rows"
            ) or None

        if st.button("🔍 Run Query String", type="primary"):
            if not query_str:
                st.warning("Please enter a query string")
            else:
                try:
                    datasets, join_conditions, filter_conditions, output_columns = parse_query_string(
                        query_str, columns_dict
                    )
                    
                    if datasets:
                        if not output_columns:
                            for ds in datasets:
                                cols_for_ds = columns_dict.get(ds, ([], ))[0]
                                output_columns.extend([f"{ds}.{col}" for col in cols_for_ds])
                            st.info(f"Selecting all columns for: {', '.join(datasets)}")
                        else:
                            st.info(f"Selecting specified columns for: {', '.join(datasets)}")

                        with st.spinner("Executing query..."):
                            result = query_multiple_large_csvs(
                                dataset_configs, datasets, join_conditions, filter_conditions,
                                preprocess_columns, chunk_size, max_rows_qs, output_columns, max_temp_storage_mb
                            )
                        
                        if not result.empty:
                            st.success(f"✓ Found {len(result):,} rows")
                            st.dataframe(result, use_container_width=True)
                            st.download_button(
                                "📥 Download Results",
                                result.to_csv(index=False),
                                "query_results.csv",
                                "text/csv",
                                key="download_qs"
                            )
                
                except ValueError as e:
                    st.error(f"❌ Query Error: {e}")
                except Exception as e:
                    st.error(f"❌ Unexpected Error: {e}")
                    st.code(traceback.format_exc())

    # -------------------------------------------------------------------------
    # TAB 3: Interactive Builder
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("Interactive Query Builder")
        st.caption("Build queries step-by-step using form controls")
        
        datasets, join_conditions, filter_conditions, max_rows, output_columns = build_query_interactively(
            columns_dict, dataset_configs
        )
        
        if datasets:
            st.info("Query submitted. Executing...")
            with st.spinner("Running query..."):
                result = query_multiple_large_csvs(
                    dataset_configs, datasets, join_conditions, filter_conditions,
                    preprocess_columns, chunk_size, max_rows, output_columns, max_temp_storage_mb
                )
            
            if not result.empty:
                st.success(f"✓ Found {len(result):,} rows")
                st.dataframe(result, use_container_width=True)
                st.download_button(
                    "📥 Download Results",
                    result.to_csv(index=False),
                    "interactive_results.csv",
                    "text/csv",
                    key="download_interactive"
                )
        else:
            st.info("👆 Build a query using the form above and click 'Run interactive query'")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.caption("""
**CSV Query Tool v23** | 
For large files (>200MB), run locally with `streamlit run streamlit_csv_query_app_v23.py` |
[Report Issues](https://github.com/yourusername/csv-query-tool/issues)
""")