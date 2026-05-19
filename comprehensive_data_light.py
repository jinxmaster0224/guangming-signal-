import pandas as pd
import os
from datetime import datetime
import math
import re
import gc
import numpy as np

# ================= 核心业务与拓扑配置区 =================
TARGET_DIRECTION = 'S' 

INTER_ID_NAME_MAP = {
    '6caaa6f15b735a': '光明大道与光辉大道',
    '6caa3ec15b566e': '光明大道与光明大街',
    '6caa06b15b50eb': '光明大道与光安路',
    '6ca93d215b3ee3': '光明大道与华夏路',
    '6ca887115b2e22': '光明大道与华裕路'
}

CORRIDOR_LINKS = [
    {"name": "光明大道与光辉大道", "seq": 1, "dist_to_next": 857},
    {"name": "光明大道与光明大街", "seq": 2, "dist_to_next": 150},
    {"name": "光明大道与光安路",   "seq": 3, "dist_to_next": 600},
    {"name": "光明大道与华夏路",   "seq": 4, "dist_to_next": 555}, 
    {"name": "光明大道与华裕路",   "seq": 5, "dist_to_next": 0}
]

INTER_OFFSET_MAP = {
    '光明大道与光辉大道': 0.0,
    '光明大道与光明大街': -32.57,
    '光明大道与光安路': -33.69,
    '光明大道与华夏路': -63.5,
    '光明大道与华裕路': -63.0,
}

INTER_CONFIG = {}
for i_id, i_name in INTER_ID_NAME_MAP.items():
    link_info = next((item for item in CORRIDOR_LINKS if item["name"] == i_name), None)
    INTER_CONFIG[i_id] = {
        "name": i_name,
        "offset": INTER_OFFSET_MAP.get(i_name, 0.0),  
        "seq": link_info["seq"] if link_info else 99,
        "dist_to_next": link_info["dist_to_next"] if link_info else 0
    }

CARDINAL_HANZI = {"E": "东向", "W": "西向", "S": "南向", "N": "北向"}
_WKT_COORD_PATTERN = re.compile(r'(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)')

def extract_direction_from_coords(lng_lat_seq: str, offset_degree: float = 0.0) -> str:
    if not isinstance(lng_lat_seq, str) or not str(lng_lat_seq).strip(): return None
    matches = _WKT_COORD_PATTERN.findall(str(lng_lat_seq))
    if len(matches) < 2: return None
    try: points = [(float(x), float(y)) for x, y in matches]
    except ValueError: return None
    dx = points[-1][0] - points[0][0]
    dy = points[-1][1] - points[0][1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6: return None
    angle_deg = math.degrees(math.atan2(dy, dx)) - offset_degree
    if angle_deg > 180: angle_deg -= 360
    elif angle_deg <= -180: angle_deg += 360
    if -45.0 <= angle_deg < 45.0: return "E"
    if 45.0 <= angle_deg < 135.0: return "N"
    if -135.0 <= angle_deg < -45.0: return "S"
    return "W"

def enrich_direction_features(df: pd.DataFrame, offset_degree: float = 0.0) -> pd.DataFrame:
    if 'lng_lat_seq' in df.columns:
        df['lng_lat_seq'] = df['lng_lat_seq'].bfill(limit=3)
        
    mask = df['lng_lat_seq'].notna() & (df['lng_lat_seq'] != '')
    df.loc[mask, '_raw_direction'] = df.loc[mask, 'lng_lat_seq'].apply(
        lambda x: extract_direction_from_coords(x, offset_degree)
    )
    
    if "frid" in df.columns:
        valid_dirs = df[['frid', '_raw_direction']].dropna()
        if not valid_dirs.empty:
            counts = valid_dirs.groupby(['frid', '_raw_direction']).size().reset_index(name='count')
            counts = counts.sort_values(by=['frid', 'count'], ascending=[True, False])
            best_dirs = counts.drop_duplicates(subset=['frid']).set_index('frid')['_raw_direction']
            df['main_direction'] = df['frid'].map(best_dirs)
            del valid_dirs, counts, best_dirs
        else: df['main_direction'] = df['_raw_direction']
    else: df['main_direction'] = df['_raw_direction']
    
    if '_raw_direction' in df.columns: df.drop(columns=['_raw_direction'], inplace=True)
    gc.collect()
    return df

def get_date_ranges():
    time_periods = [
        (True, '2026-03-25 00:00:00', '2026-03-25 23:59:59'),  
        (False, '2026-03-11 00:00:00', '2026-03-11 23:59:59'),  
        (False, '2026-03-12 00:00:00', '2026-03-12 23:59:59')
    ]
    date_ranges = []
    for enable, start_date, end_date in time_periods:
        if enable:
            try: date_ranges.append((pd.Timestamp(start_date), pd.Timestamp(end_date)))
            except: pass
    if not date_ranges: date_ranges.append(None)
    return date_ranges

# ================= 核心分析模块 (强制 5 分钟切片驱动版) =================
def analyze_multiple_files(file_paths, date_ranges, inter_config):
    if not file_paths:
        print("❌ 当前文件夹下未找到任何包含 'index_view' 的文件。")
        return

    global_granular_data = [] 

    for file_path in file_paths:
        print(f"\n📂 正在读取文件: {os.path.basename(file_path)}")
        try: df_dir = pd.read_csv(file_path, encoding='GBK')
        except: df_dir = pd.read_csv(file_path, encoding='utf-8')
        
        if df_dir.empty or 'inter_id' not in df_dir.columns:
            continue
            
        current_inter_id = str(df_dir['inter_id'].iloc[0]).strip().lower()
        if current_inter_id not in inter_config:
            continue
            
        config = inter_config[current_inter_id]
        inter_name = config.get('name', f"未知路口_{current_inter_id}")
        offset = config.get('offset', 0.0)
        
        df_dir['create_time'] = pd.to_datetime(df_dir['create_time'])

        if date_ranges and date_ranges[0] is not None:
            mask_dir = pd.Series(False, index=df_dir.index)
            for start_time, end_time in date_ranges:
                mask_dir |= (df_dir['create_time'] >= start_time) & (df_dir['create_time'] <= end_time)
            df_dir = df_dir[mask_dir].copy()

        df_dir = df_dir[df_dir['create_time'].dt.minute % 5 == 0].copy()
        df_dir = df_dir.sort_values('create_time')

        if df_dir.empty:
            print(f"⚠️ 路口 {inter_name} 在指定日期范围内无有效抽样数据，跳过。")
            continue

        print("  -> 正在清洗流向并提取基础指标...")
        df_dir = enrich_direction_features(df_dir, offset_degree=offset)
        
        if 'turn_dir_no' in df_dir.columns:
            df_dir = df_dir[df_dir['turn_dir_no'].isin([1, 2])].copy()

        df_dir['路口名称'] = inter_name
        df_dir['进口道方向'] = df_dir['main_direction'].map(CARDINAL_HANZI)
        df_dir['转向'] = df_dir['turn_dir_no'].map({1: '左转', 2: '直行'})
        df_dir['延误指数'] = df_dir.get('delay_index', 0.0)

        pass_flow_cols = [c for c in df_dir.columns if 'pass_flow' in c.lower() or '分均流量' in c]
        if pass_flow_cols:
            p_col = pass_flow_cols[0]
            df_dir['pass_flow'] = df_dir[p_col].astype(str).str.replace(',', '', regex=False)
            df_dir['pass_flow'] = df_dir['pass_flow'].str.extract(r'(\d+\.?\d*)')[0]
            df_dir['pass_flow'] = pd.to_numeric(df_dir['pass_flow'], errors='coerce').fillna(0)
        else:
            df_dir['pass_flow'] = 0

        print("  -> 正在基于 5 分钟代表值计算 [历史最大流量] 与 [瞬时流量占比]...")
        hist_max_df = df_dir.groupby(['进口道方向', '转向'])['pass_flow'].max().reset_index()
        hist_max_df.rename(columns={'pass_flow': '历史最大分均流量'}, inplace=True)
        df_dir = pd.merge(df_dir, hist_max_df, on=['进口道方向', '转向'], how='left')

        total_flow_5min = df_dir.groupby('create_time')['pass_flow'].transform('sum')
        df_dir['流量占比(%)'] = np.where(total_flow_5min > 0, (df_dir['pass_flow'] / total_flow_5min) * 100, 0.0)

        df_export = df_dir.dropna(subset=['进口道方向', '转向']).copy()
        export_cols = [
            '路口名称', '进口道方向', '转向', 'create_time', '延误指数', 
            'pass_flow', '历史最大分均流量', '流量占比(%)',
            'queue_len_avg', 'queue_len_max'
        ]
        
        for c in export_cols:
            if c not in df_export.columns: df_export[c] = None
        
        df_final_export = df_export[export_cols].rename(columns={
            'pass_flow': '5分钟车流量',
            '历史最大分均流量': '历史最大5分钟流量',
            'queue_len_avg': '平均排队长度',
            'queue_len_max': '最大排队长度'
        })
        global_granular_data.append(df_final_export)

    base_output_dir = os.path.dirname(file_paths[0]) if file_paths else ""

    if global_granular_data:
        df_all_granular = pd.concat(global_granular_data, ignore_index=True)
        granular_save_path = os.path.join(base_output_dir, "干线全流向_延误与流量占比明细表.csv")
        try:
            df_all_granular.to_csv(granular_save_path, index=False, encoding='GBK')
            print(f"\n✅ [供推演引擎使用] 带精确5分钟切片与极大值的明细表已保存至: {granular_save_path}")
        except PermissionError:
            print(f"\n⚠️ 警告: 无法保存明细表CSV！请先关闭正在 Excel 中打开的文件。")

    print("\n✅ 所有配置路口的分析任务均已执行完毕！")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    csv_files = [os.path.join(script_dir, f) for f in os.listdir(script_dir) if "index_view" in f and f.endswith('.csv')]
    
    print(f"=================================")
    print(f"共发现 {len(csv_files)} 个 index_view 数据文件")
    print(f"=================================\n")
    
    date_ranges = get_date_ranges()
    
    analyze_multiple_files(
        file_paths=csv_files, 
        date_ranges=date_ranges, 
        inter_config=INTER_CONFIG
    )