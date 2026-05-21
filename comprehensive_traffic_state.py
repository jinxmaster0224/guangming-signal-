import pandas as pd
import numpy as np
import os
import math
import re
import gc
from datetime import datetime, time

# ================= 全局配置区 =================
# 【新增双接口】分别指定工作日与非工作日的分析目标日期
TARGET_DATE_WD = '2026-03-24'
DAY_TYPE_WD = '工作日'

TARGET_DATE_WE = '2026-03-21'
DAY_TYPE_WE = '非工作日'

INTERSECTION = '光明大道与光明大街'
INTERSECTIONS = [INTERSECTION]

MIN_K = 5
MAX_K = 20
IMPROVEMENT_THRESHOLD = 0.10  

# ================= 现行方案输入接口 =================
# 1. 工作日现行方案
CURRENT_SIGNAL_PLAN_WEEKDAY = {
    '现行周期方案': [
        {'start': '00:00', 'end': '06:00', 'scheme': '方案10', 'cycle': 99, 'phase_times': {'北向全放': 25, '东向全放': 25, '西向全放': 25, '南向全放': 24}},
        {'start': '06:00', 'end': '07:30', 'scheme': '方案2', 'cycle': 114, 'phase_times': {'北向全放': 28, '东向全放': 28, '西向全放': 30, '南向全放': 28}},
        {'start': '07:30', 'end': '09:00', 'scheme': '方案24', 'cycle': 167, 'phase_times': {'北向全放': 46, '东向全放': 40, '西向全放': 30, '南向全放': 51}},
        {'start': '09:00', 'end': '12:00', 'scheme': '方案26', 'cycle': 152, 'phase_times': {'北向全放': 40, '东向全放': 35, '西向全放': 30, '南向全放': 47}},
        {'start': '12:00', 'end': '14:00', 'scheme': '方案3', 'cycle': 150, 'phase_times': {'北向全放': 45, '东向全放': 35, '西向全放': 30, '南向全放': 41}},
        {'start': '14:00', 'end': '16:00', 'scheme': '方案4', 'cycle': 152, 'phase_times': {'北向全放': 42, '东向全放': 35, '西向全放': 30, '南向全放': 45}},
        {'start': '16:00', 'end': '17:00', 'scheme': '方案5', 'cycle': 152, 'phase_times': {'北向全放': 44, '东向全放': 33, '西向全放': 30, '南向全放': 45}},
        {'start': '17:00', 'end': '19:00', 'scheme': '方案17', 'cycle': 164, 'phase_times': {'北向全放': 48, '东向全放': 40, '西向全放': 27, '南向全放': 49}},
        {'start': '19:00', 'end': '20:30', 'scheme': '方案7', 'cycle': 134, 'phase_times': {'北向全放': 34, '东向全放': 37, '西向全放': 26, '南向全放': 37}},
        {'start': '20:30', 'end': '22:00', 'scheme': '方案8', 'cycle': 130, 'phase_times': {'北向全放': 34, '东向全放': 34, '西向全放': 28, '南向全放': 34}},
        {'start': '22:00', 'end': '23:59', 'scheme': '方案9', 'cycle': 115, 'phase_times': {'北向全放': 30, '东向全放': 28, '西向全放': 28, '南向全放': 29}},
    ],
    '相位损失时间': {'北向全放': 6, '东向全放': 6, '西向全放': 6, '南向全放': 6},
    '专属相位映射': {
        '相位1(北向全放)': [('北向', '直行'), ('北向', '左转')],
        '相位2(东向全放)': [('东向', '直行'), ('东向', '左转')],
        '相位3(西向全放)': [('西向', '直行'), ('西向', '左转')],
        '相位4(南向全放)': [('南向', '直行'), ('南向', '左转')]
    }
}

# 2. 非工作日现行方案
CURRENT_SIGNAL_PLAN_WEEKEND = {
    '现行周期方案': [
        {'start': '00:00', 'end': '06:00', 'scheme': '方案11', 'cycle': 100, 'phase_times': {'北向全放': 25, '东向全放': 25, '西向全放': 25, '南向全放': 25}},
        {'start': '06:00', 'end': '07:30', 'scheme': '方案12', 'cycle': 110, 'phase_times': {'北向全放': 27, '东向全放': 28, '西向全放': 27, '南向全放': 28}},
        {'start': '07:30', 'end': '09:00', 'scheme': '方案6', 'cycle': 150, 'phase_times': {'北向全放': 44, '东向全放': 35, '西向全放': 30, '南向全放': 41}},
        {'start': '09:00', 'end': '12:00', 'scheme': '方案13', 'cycle': 158, 'phase_times': {'北向全放': 43, '东向全放': 32, '西向全放': 30, '南向全放': 53}},
        {'start': '12:00', 'end': '14:00', 'scheme': '方案15', 'cycle': 146, 'phase_times': {'北向全放': 42, '东向全放': 35, '西向全放': 30, '南向全放': 39}},
        {'start': '14:00', 'end': '16:00', 'scheme': '方案18', 'cycle': 154, 'phase_times': {'北向全放': 46, '东向全放': 34, '西向全放': 28, '南向全放': 46}},
        {'start': '16:00', 'end': '17:30', 'scheme': '方案19', 'cycle': 158, 'phase_times': {'北向全放': 47, '东向全放': 35, '西向全放': 32, '南向全放': 44}},
        {'start': '17:30', 'end': '19:00', 'scheme': '方案23', 'cycle': 160, 'phase_times': {'北向全放': 47, '东向全放': 34, '西向全放': 31, '南向全放': 48}},
        {'start': '19:00', 'end': '20:30', 'scheme': '方案25', 'cycle': 136, 'phase_times': {'北向全放': 34, '东向全放': 37, '西向全放': 28, '南向全放': 37}},
        {'start': '20:30', 'end': '22:00', 'scheme': '方案27', 'cycle': 128, 'phase_times': {'北向全放': 33, '东向全放': 33, '西向全放': 28, '南向全放': 34}},
        {'start': '22:00', 'end': '23:59', 'scheme': '方案20', 'cycle': 110, 'phase_times': {'北向全放': 28, '东向全放': 28, '西向全放': 26, '南向全放': 28}},
    ],
    '相位损失时间': {'北向全放': 6, '东向全放': 6, '西向全放': 6, '南向全放': 6},
    '专属相位映射': {
        '相位1(北向全放)': [('北向', '直行'), ('北向', '左转')],
        '相位2(东向全放)': [('东向', '直行'), ('东向', '左转')],
        '相位3(西向全放)': [('西向', '直行'), ('西向', '左转')],
        '相位4(南向全放)': [('南向', '直行'), ('南向', '左转')]
    }
}

TRAFFIC_STATE_THRESHOLDS = {
    '极低流量': {'max_flow': 300, 'description': '夜间低峰，流量稀少'},
    '平峰流量': {'min_flow': 300, 'max_flow': 1250, 'description': '日常平峰，流量适中（含早高峰前、白天平峰、晚高峰后）'},
    '高峰流量': {'min_flow': 1250, 'description': '真正早晚高峰，流量拥挤'}
}

STATE_OPTIMIZATION_WEIGHTS = {
    '极低流量': {'delay_weight': 1.0},
    '平峰流量': {'delay_weight': 0.7, 'throughput_weight': 0.3},
    '高峰流量': {'delay_weight': 0.7, 'throughput_weight': 0.0, 'queue_weight': 0.3}
}

CYCLE_BOUNDS = {
    '极低流量': {'min': 90, 'max': 110, 'default': 95},
    '平峰流量': {'min': 90, 'max': 150, 'default': 115},
    '高峰流量': {'min': 140, 'max': 180, 'default': 160}
}

INTER_ID_NAME_MAP = {
    '6caaa6f15b735a': '光明大道与光辉大道',
    '6caa3ec15b566e': '光明大道与光明大街',
    '6caa06b15b50eb': '光明大道与光安路',
    '6ca93d215b3ee3': '光明大道与华夏路',
    '6ca887115b2e22': '光明大道与华裕路'
}

INTER_OFFSET_MAP = {
    '光明大道与光辉大道': 0.0, 
    '光明大道与光明大街': -32.57, 
    '光明大道与光安路': -33.69,
    '光明大道与华夏路': -63.5, 
    '光明大道与华裕路': -63.0,
}

CARDINAL_HANZI = {"E": "东向", "W": "西向", "S": "南向", "N": "北向"}
TURN_HANZI = {1: "左转", 2: "直行"}
_WKT_COORD_PATTERN = re.compile(r'(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)')

# ================= 基础模块 =================
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
    if 'lng_lat_seq' in df.columns: df['lng_lat_seq'] = df['lng_lat_seq'].bfill(limit=3)
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
        else: df['main_direction'] = df['_raw_direction']
    else: df['main_direction'] = df['_raw_direction']
    if '_raw_direction' in df.columns: df.drop(columns=['_raw_direction'], inplace=True)
    return df

# ================= Phase 1: 构建统一的底层DataFrame并分析数据完整性 =================
def build_core_detail_table(file_paths, output_dir):
    global_granular_data = [] 
    date_mapping = {TARGET_DATE_WD: DAY_TYPE_WD, TARGET_DATE_WE: DAY_TYPE_WE}

    for file_path in file_paths:
        try: df_dir = pd.read_csv(file_path, encoding='GBK')
        except: df_dir = pd.read_csv(file_path, encoding='utf-8')
        
        if df_dir.empty or 'inter_id' not in df_dir.columns: continue
        current_inter_id = str(df_dir['inter_id'].iloc[0]).strip().lower()
        
        inter_name = INTER_ID_NAME_MAP.get(current_inter_id, "")
        if inter_name != INTERSECTION: continue
            
        offset = INTER_OFFSET_MAP.get(inter_name, 0.0)
        
        df_dir['create_time'] = pd.to_datetime(df_dir['create_time'])
        
        # 1. 抽取非重叠的5分钟滚动切片
        df_dir = df_dir[df_dir['create_time'].dt.minute % 5 == 0].copy()
        
        # 2. 还原物理时间 (减去10分钟)
        df_dir['业务时间'] = df_dir['create_time'] - pd.Timedelta(minutes=10)
        df_dir['业务日期'] = df_dir['业务时间'].dt.strftime('%Y-%m-%d')
        
        # 3. 过滤出我们需要的两天数据
        df_dir = df_dir[df_dir['业务日期'].isin(date_mapping.keys())].copy()
        if df_dir.empty: continue
            
        df_dir['日期类型'] = df_dir['业务日期'].map(date_mapping)
        df_dir = df_dir.sort_values('业务时间')

        df_dir = enrich_direction_features(df_dir, offset_degree=offset)
        if 'turn_dir_no' in df_dir.columns: df_dir = df_dir[df_dir['turn_dir_no'].isin([1, 2])].copy()
        
        df_dir['路口名称'] = inter_name
        df_dir['进口道方向'] = df_dir['main_direction'].map(CARDINAL_HANZI)
        df_dir['转向'] = df_dir['turn_dir_no'].map({1: '左转', 2: '直行'})
        df_dir['流向'] = df_dir['进口道方向'] + df_dir['转向']
        df_dir['延误指数'] = df_dir.get('delay_index', 0.0)

        pass_flow_cols = [c for c in df_dir.columns if 'pass_flow' in c.lower() or '分均流量' in c]
        if pass_flow_cols:
            df_dir['pass_flow'] = df_dir[pass_flow_cols[0]].astype(str).str.replace(',', '', regex=False)
            df_dir['pass_flow'] = pd.to_numeric(df_dir['pass_flow'].str.extract(r'(\d+\.?\d*)')[0], errors='coerce').fillna(0)
        else: df_dir['pass_flow'] = 0

        hist_max_df = df_dir.groupby('流向')['pass_flow'].max().reset_index()
        hist_max_df.rename(columns={'pass_flow': '历史最大分均流量'}, inplace=True)
        df_dir = pd.merge(df_dir, hist_max_df, on='流向', how='left')

        df_export = df_dir.dropna(subset=['流向']).copy()
        export_cols = ['路口名称', '日期类型', '业务日期', '业务时间', '流向', '延误指数', 'pass_flow', '历史最大分均流量', 'queue_len_avg', 'queue_len_max']
        for c in export_cols: 
            if c not in df_export.columns: df_export[c] = None
        
        df_final = df_export[export_cols].rename(columns={
            'pass_flow': '5分钟车流量', 'queue_len_avg': '平均排队长度', 'queue_len_max': '最大排队长度'
        })
        global_granular_data.append(df_final)

    if global_granular_data:
        df_all_granular = pd.concat(global_granular_data, ignore_index=True)
        
        # ------------------ 新增：数据完整性分析模块 ------------------
        integrity_records = []
        for target_date, day_type in date_mapping.items():
            df_day = df_all_granular[df_all_granular['业务日期'] == target_date]
            if df_day.empty:
                continue
                
            unique_flows = df_day['流向'].nunique()
            actual_count = len(df_day)
            
            # 每天 24小时 * 12个切片(每5分钟1个) = 288 个时间切片
            expected_count = 288 * unique_flows
            integrity_rate = (actual_count / expected_count) * 100 if expected_count > 0 else 0
            
            integrity_records.append({
                '路口名称': INTERSECTION,
                '日期类型': day_type,
                '分析目标日期': target_date,
                '实际有效流向数': unique_flows,
                '每天理论切片数': 288,
                '理论应有数据量(条)': expected_count,
                '实际清洗后数据量(条)': actual_count,
                '数据缺失量(条)': expected_count - actual_count,
                '数据完整率(%)': round(integrity_rate, 2)
            })
            
        integrity_out_path = None
        if integrity_records:
            df_integrity = pd.DataFrame(integrity_records)
            integrity_out_path = os.path.join(output_dir, f"数据完整性分析报告_{INTERSECTION}.csv")
            df_integrity.to_csv(integrity_out_path, index=False, encoding='GBK')
        # -------------------------------------------------------------

        # 仅作保存参考，不作为后续读取媒介
        out_path = os.path.join(output_dir, f"干线全流向_延误与流量占比明细表_{INTERSECTION}.csv")
        df_all_granular.to_csv(out_path, index=False, encoding='GBK')
        
        # ✅ 直接返回内存中的 DataFrame 以及两个输出路径
        return df_all_granular, out_path, integrity_out_path
    
    return None, None, None

# ================= Phase 2: Fisher时段划分 (基于内存DataFrame) =================
def fisher_optimal_partition(data_matrix, k_classes):
    n = len(data_matrix)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            segment = data_matrix[i:j+1]
            D[i, j] = np.sum(np.linalg.norm(segment - np.mean(segment, axis=0), axis=1)**2)
            
    dp = np.full((n, k_classes + 1), np.inf)
    split = np.zeros((n, k_classes + 1), dtype=int)
    for i in range(n): dp[i][1] = D[0, i]
        
    for m in range(2, k_classes + 1):
        for i in range(m - 1, n):
            for j in range(m - 2, i):
                cost = dp[j][m - 1] + D[j + 1, i]
                if cost < dp[i][m]:
                    dp[i][m] = cost; split[i][m] = j
                    
    boundaries, curr = [], n - 1
    for m in range(k_classes, 1, -1):
        curr = split[curr][m]; boundaries.append(curr)
    boundaries.reverse()
    return boundaries, dp[n-1][k_classes]

def run_tod_partition(df_all, output_dir):
    target_flows = ['东向直行', '东向左转', '南向直行', '南向左转', '西向直行', '西向左转', '北向直行', '北向左转']
    all_time_periods = {DAY_TYPE_WD: [], DAY_TYPE_WE: []}
    csv_export_data = []
    
    # 保证时间格式
    df_all['业务时间'] = pd.to_datetime(df_all['业务时间'])
    
    for day_type, target_date in [(DAY_TYPE_WD, TARGET_DATE_WD), (DAY_TYPE_WE, TARGET_DATE_WE)]:
        time_periods = []
        
        # 内存过滤目标日期的数据
        df = df_all[df_all['业务日期'] == target_date].copy()
        if df.empty: continue
            
        time_offsets = df['业务时间'] - df['业务时间'].dt.normalize()
        df['_temp_time'] = pd.Timestamp('2026-01-01') + time_offsets

        df['5分钟车流量'] = pd.to_numeric(df['5分钟车流量'], errors='coerce').fillna(0)
        df['延误指数'] = pd.to_numeric(df['延误指数'], errors='coerce').fillna(0)
        df['最大排队长度'] = pd.to_numeric(df['最大排队长度'], errors='coerce').fillna(0)

        df_flow = df.pivot_table(index='_temp_time', columns='流向', values='5分钟车流量', aggfunc='sum').fillna(0)
        for tf in target_flows: 
            if tf not in df_flow.columns: df_flow[tf] = 0
                
        df_status = df.groupby('_temp_time').agg({'5分钟车流量': 'sum', '延误指数': 'mean', '最大排队长度': 'max'}).fillna(0)
        df_time = df_flow.join(df_status, how='outer').fillna(0).sort_index()

        full_time_range = pd.date_range(start='2026-01-01 00:00:00', end='2026-01-01 23:59:59', freq='15min')
        agg_rules = {'5分钟车流量': 'sum', '延误指数': 'mean', '最大排队长度': 'max'}
        for tf in target_flows: agg_rules[tf] = 'sum'
        df_15min = df_time.resample('15min').agg(agg_rules).reindex(full_time_range).ffill().fillna(0)

        feature_cols = []
        for tf in target_flows:
            feat = f'flow_{tf}'
            df_15min[feat] = df_15min[tf].rolling(3, min_periods=1).mean()
            feature_cols.append(feat)
        df_15min['d_feat'] = df_15min['延误指数'].rolling(4, min_periods=1).mean()
        df_15min['q_feat'] = df_15min['最大排队长度'].rolling(4, min_periods=1).max()
        feature_cols.extend(['d_feat', 'q_feat'])

        raw_matrix = df_15min[feature_cols].values
        data_matrix = (raw_matrix - np.mean(raw_matrix, axis=0)) / (np.std(raw_matrix, axis=0) + 1e-8)

        k_results = {}
        for k in range(MIN_K, MAX_K + 1):
            bounds, cost = fisher_optimal_partition(data_matrix, k)
            k_results[k] = {'boundaries': bounds, 'cost': cost}
            
        optimal_k = MIN_K
        for k in range(MIN_K + 1, MAX_K + 1):
            if (k_results[k-1]['cost'] - k_results[k]['cost']) / k_results[k-1]['cost'] < IMPROVEMENT_THRESHOLD:
                optimal_k = k - 1; break
        else: optimal_k = MAX_K

        best_bounds = k_results[optimal_k]['boundaries']
        start_idx = 0
        for i, split_idx in enumerate(best_bounds + [95]):
            s_time = df_15min.index[start_idx].strftime('%H:%M')
            e_time = (df_15min.index[split_idx] + pd.Timedelta(minutes=15)).strftime('%H:%M')
            if e_time == '00:00': e_time = '23:59'
            period_data = df_15min.iloc[start_idx:split_idx+1]
            
            avg_flow = int(period_data['5分钟车流量'].mean())
            max_queue = int(round(period_data['最大排队长度'].max(), 0))
            
            time_periods.append({
                'id': i + 1, 'start': s_time, 'end': e_time,
                'avg_flow': avg_flow, 'max_queue': max_queue
            })
            
            csv_export_data.append({
                '日期类型': day_type,
                '时段编号': f"时段{i+1:02d}",
                '开始时间': s_time,
                '结束时间': e_time,
                '平均流量(pcu/15min)': avg_flow
            })
            
            start_idx = split_idx + 1
            
        all_time_periods[day_type] = time_periods

    df_periods = pd.DataFrame(csv_export_data)
    out_path = os.path.join(output_dir, f'时段划分结果表_{INTERSECTION}.csv')
    df_periods.to_csv(out_path, index=False, encoding='GBK')
    
    return all_time_periods, out_path

# ================= Phase 3: 周期与绿信比推演 (基于内存DataFrame) =================
def get_current_cycle_for_time(time_str, current_plan):
    for scheme in current_plan['现行周期方案']:
        if time_str >= scheme['start'] and time_str < scheme['end']: return scheme['cycle'], scheme['scheme'], scheme.get('phase_times', {})
    return 99, '未配置', {}

def get_time_strs_in_period(start_str, end_str):
    sh, sm = map(int, start_str.split(':'))
    eh, em = map(int, end_str.split(':'))
    res, ch, cm = [], sh, sm
    while (ch < eh) or (ch == eh and cm < em):
        res.append(f"{ch:02d}:{cm:02d}")
        cm += 5
        if cm >= 60: cm, ch = 0, ch + 1
    return res

def get_current_cycle_for_period(start_str, end_str, current_plan):
    cycles, schemes, pt_list = [], [], []
    for t in get_time_strs_in_period(start_str, end_str):
        c, s, pt = get_current_cycle_for_time(t, current_plan)
        cycles.append(c); schemes.append(s); pt_list.append(pt)
    if cycles:
        most_c = max(set(cycles), key=cycles.count)
        most_s = max(set(schemes), key=schemes.count)
        most_pt = dict(max(set(tuple(p.items()) for p in pt_list), key=pt_list.count)) if pt_list else {}
        return most_c, most_s, most_pt
    return 99, '未知', {}

def calculate_hcm_delay(C, lambda_i, x, c, T=0.25, k=0.5, I=1.0):
    if lambda_i <= 0 or c <= 0: return 9999.0
    d1 = (0.5 * C * (1 - lambda_i)**2) / (1 - min(1.0, x) * lambda_i)
    inside_sqrt = max(0, (x - 1)**2 + (8 * k * I * x) / (c * T))
    return d1 + 900 * T * ((x - 1) + math.sqrt(inside_sqrt))

def classify_period_state(period_avg_flow_15min):
    f = period_avg_flow_15min / 3
    if f <= TRAFFIC_STATE_THRESHOLDS['极低流量']['max_flow']: return '极低流量'
    elif TRAFFIC_STATE_THRESHOLDS['平峰流量']['min_flow'] < f <= TRAFFIC_STATE_THRESHOLDS['平峰流量']['max_flow']: return '平峰流量'
    else: return '高峰流量'

def allocate_green_time(phase_data, C, G, method):
    num_phases = len(phase_data)
    Y_total = sum(d['y_val'] for d in phase_data.values())
    temp_alloc = {}
    
    if method == 'low_peak':
        base_g, surplus_g = 20.0, G - (num_phases * 20.0)
        for p, d in phase_data.items():
            temp_alloc[p] = base_g + (surplus_g * (d['y_val'] / Y_total) if Y_total > 0 else surplus_g / num_phases)
            
    elif method == 'off_peak':
        unallocated, remaining_G = list(phase_data.keys()), G
        while unallocated:
            cY = sum(phase_data[p]['y_val'] for p in unallocated)
            if cY <= 0:
                for p in unallocated: temp_alloc[p] = remaining_G / len(unallocated)
                break
            min_p = min(unallocated, key=lambda p: phase_data[p]['y_val'])
            if remaining_G * (phase_data[min_p]['y_val'] / cY) < 20.0:
                temp_alloc[min_p] = 20.0; remaining_G -= 20.0; unallocated.remove(min_p)
            else:
                for p in unallocated: temp_alloc[p] = remaining_G * (phase_data[p]['y_val'] / cY)
                break
                
    int_g = {p: int(math.floor(g)) for p, g in temp_alloc.items()}
    rems = {p: g - int_g[p] for p, g in temp_alloc.items()}
    for p in sorted(rems.keys(), key=lambda x: rems[x], reverse=True)[:int(round(G - sum(int_g.values())))]:
        int_g[p] += 1
    return int_g

def optimize_cycle(phase_data, L_total, state):
    bounds = CYCLE_BOUNDS[state]; w = STATE_OPTIMIZATION_WEIGHTS[state]
    Y_total = sum([d['y_val'] for d in phase_data.values()])
    tq = sum([d['q_5min'] for d in phase_data.values()])
    min_obj, best_C, best_alloc = float('inf'), bounds['default'], {}
    
    t_cyc = max(bounds['min'], min(bounds['min'] + int(Y_total * 50), bounds['max'])) if state == '高峰流量' else bounds['default']
    
    for C in range(bounds['min'], bounds['max'] + 1):
        G = C - L_total
        if G < len(phase_data) * 20.0: continue
            
        alloc = allocate_green_time(phase_data, C, G, 'low_peak' if state in ['极低流量', '高峰流量'] else 'off_peak')
        cur_alloc, t_del, t_sat, max_sat, valid = {}, 0.0, 0.0, 0.0, True
        
        for p, d in phase_data.items():
            g_i, s_h, q_h = alloc[p], d['s_real_h'], d['q_5min'] * 12
            lam = g_i / C
            if lam <= 0.01 or (state == '高峰流量' and s_h <= 0): valid = False; break
            c = s_h * lam
            x = q_h / c if c > 0 else 9999.0
            if state != '极低流量' and x > 0.92: valid = False; break
            
            d_i = calculate_hcm_delay(C, lam, x, c) if c > 0 else 9999.0
            vw = d['q_5min'] / tq if tq > 0 else 0
            t_del += d_i * vw; t_sat += x * vw; max_sat = max(max_sat, x)
            cur_alloc[p] = {'有效绿灯(s)': int(g_i), '绿信比': round(lam, 3), '饱和度': round(x, 3), '延误(s/veh)': round(d_i, 2)}
            
        if not valid: continue
            
        obj = w.get('delay_weight', 0) * (t_del / 100)
        if state != '极低流量': obj += w.get('throughput_weight', 0) * (abs(t_sat - 0.88) * 10)
        if state == '高峰流量': obj += w.get('queue_weight', 0) * (max(0, max_sat - 0.85) * 10) + 0.2 * (abs(C - t_cyc) / (bounds['max'] - bounds['min']) * 10)
            
        if obj < min_obj: min_obj, best_C, best_alloc = obj, C, cur_alloc
            
    if not best_alloc and state == '极低流量': return optimize_cycle(phase_data, L_total, '平峰流量')
    return best_C, min_obj, best_alloc

def run_period_analysis(df_all, time_periods, output_dir, current_plan, day_type):
    # 内存截取目标标签的数据
    df = df_all[df_all['日期类型'] == day_type].copy()
    df['业务时间'] = pd.to_datetime(df['业务时间'])
    
    df['flow'] = pd.to_numeric(df['5分钟车流量'], errors='coerce').fillna(0)
    df['max_flow'] = pd.to_numeric(df['历史最大分均流量'], errors='coerce').fillna(0)

    print(f"\n\n{'='*80}")
    print(f"【{INTERSECTION}】各时段周期优化分析 —— 📍 [{day_type}方案]")
    print("=" * 80)
    print(f"\n[{day_type}] 现行周期方案:")
    for scheme in current_plan['现行周期方案']:
        print(f"  {scheme['start']}-{scheme['end']}: {scheme['scheme']}, 周期{scheme['cycle']}s")
    
    print("\n流量状态 definition (等效15分钟单位):")
    for st, info in TRAFFIC_STATE_THRESHOLDS.items():
        desc = info.get('description', '')
        if 'max_flow' in info: print(f"  {st}: ≤{info['max_flow'] * 3} pcu/15min | {desc}")
        elif 'min_flow' in info and 'max_flow' in info: print(f"  {st}: {info['min_flow'] * 3}-{info['max_flow'] * 3} pcu/15min | {desc}")
        else: print(f"  {st}: >{info['min_flow'] * 3} pcu/15min | {desc}")
    
    print(f"\n时段划分 (Fisher肘部法则，共{len(time_periods)}段):")
    for p in time_periods: print(f"  时段{p['id']:02d}: {p['start']}-{p['end']} | 平均流量{p['avg_flow']} pcu/15min")
    
    print("\n" + "=" * 80)
    print(f"各时段周期优化结果 ({day_type})")
    print("=" * 80)

    results = []
    L_total = sum(current_plan['相位损失时间'].values())
    
    for p in time_periods:
        st = classify_period_state(p['avg_flow'])
        cc, cs, cpt = get_current_cycle_for_period(p['start'], p['end'], current_plan)
        time_strs = get_time_strs_in_period(p['start'], p['end'])
        
        pdf = df[df['业务时间'].dt.strftime('%H:%M').isin(time_strs)].copy()
        
        phase_data = {}
        for pname, dirs in current_plan['专属相位映射'].items():
            max_y, cq, cs_val = -1, 0, 0
            for d, t in dirs:
                tname = f"{d}{t}"
                q = pdf[pdf['流向'] == tname]['flow'].mean()
                if pd.isna(q): q = df[df['流向'] == tname]['flow'].mean()
                s_raw = df[df['流向'] == tname]['max_flow'].max()
                if pd.isna(s_raw) or s_raw == 0: s_raw = 150
                
                old_lambda = 0.2
                match = re.search(r'\((.*?)\)', pname)
                if match:
                    raw_n = match.group(1)
                    old_g = cpt.get(raw_n, 0) - current_plan['相位损失时间'].get(raw_n, 6)
                    if old_g > 0 and cc > 0: old_lambda = old_g / cc
                        
                s_real = (s_raw * 12) / old_lambda
                y_val = (q * 12) / s_real
                if y_val > max_y: max_y, cq, cs_val = y_val, q, s_real
            if max_y > 0: phase_data[pname] = {'q_5min': cq, 's_real_h': cs_val, 'y_val': max_y}
                
        rec_C, _, alloc = optimize_cycle(phase_data, L_total, st)
        results.append({
            'period_id': p['id'], 'start': p['start'], 'end': p['end'], 'state': st,
            'avg_flow': p['avg_flow'], 'max_queue': p.get('max_queue', 0),
            'current_cycle': cc, 'current_scheme': cs, 'recommended_cycle': rec_C, 'allocation': alloc
        })
        
        diff = rec_C - cc
        diff_pct = (diff / cc) * 100 if cc > 0 else 0
        
        print(f"\n【时段 {p['id']:02d}】{p['start']} - {p['end']}")
        print(f"  流量状态: {st}")
        print(f"  平均流量: {p['avg_flow']} pcu/15min | 最大排队: {p.get('max_queue', 0)}m")
        print(f"  现行方案: {cs} | 周期: {cc}s")
        print(f"  推荐周期: {rec_C}s | 差异: {diff:+d}s ({diff_pct:+.1f}%)")
        print(f"  相位分配:")
        for phase, metrics in alloc.items():
            print(f"    {phase}: 绿灯{metrics['有效绿灯(s)']}s | 绿信比{metrics['绿信比']} | 饱和度{metrics['饱和度']} | 延误{metrics['延误(s/veh)']}s")

    print("\n" + "=" * 80)
    print(f"汇总表 ({day_type})")
    print("=" * 80)
    print(f"{'时段':^6} | {'时间范围':^14} | {'流量状态':^8} | {'平均流量':^10} | {'现行方案':^8} | {'现行周期':^10} | {'推荐周期':^10} | {'差异':^8}")
    print("-" * 90)
    
    for r in results:
        diff = r['recommended_cycle'] - r['current_cycle']
        print(f"{r['period_id']:02d}    | {r['start']}-{r['end']:^7} | {r['state']:^8} | {r['avg_flow']:>8} pcu | {r['current_scheme']:^8} | {r['current_cycle']:>8}s | {r['recommended_cycle']:>8}s | {f'{diff:+d}s':^8}")
    
    c_avg = sum(r['current_cycle'] for r in results) / len(results)
    r_avg = sum(r['recommended_cycle'] for r in results) / len(results)
    print("-" * 90)
    print(f"现行周期均值: {c_avg:.1f}s | 推荐周期均值: {r_avg:.1f}s | 差异: {r_avg - c_avg:+.1f}s")
    
    out_df = pd.DataFrame([{
        '时段编号': r['period_id'], '开始时间': r['start'], '结束时间': r['end'], '流量状态': r['state'],
        '平均流量(pcu/15min)': r['avg_flow'], '现行周期(s)': r['current_cycle'], 
        '推荐周期(s)': r['recommended_cycle'], '周期差异(s)': r['recommended_cycle'] - r['current_cycle'],
        **{f"{ph}_绿灯(s)": m['有效绿灯(s)'] for ph, m in r['allocation'].items()}
    } for r in results])
    
    out_filename = f"各时段周期优化结果_{INTERSECTION}{day_type}.csv"
    out_path = os.path.join(output_dir, out_filename)
    out_df.to_csv(out_path, index=False, encoding='GBK')
    return out_filename, out_path

# ================= 统一启动引擎 =================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = [os.path.join(script_dir, f) for f in os.listdir(script_dir) if "index_view" in f and f.endswith('.csv')]
    
    generated_files = []

    if not csv_files:
        print("❌ 未在当前目录下找到 'index_view' 数据文件！")
    else:
        target_dir = os.path.join(script_dir, f"{INTERSECTION}信控方案")
        os.makedirs(target_dir, exist_ok=True)
        
        # 1. 提取 DataFrame 并生成两个输出文件路径
        df_processed, processed_data_path, integrity_data_path = build_core_detail_table(csv_files, target_dir)
        
        if df_processed is not None:
            # 记录完整性分析报告
            if integrity_data_path:
                generated_files.append(("底层数据完整性分析报告", os.path.basename(integrity_data_path), integrity_data_path))
                
            generated_files.append(("底层聚合清洗数据明细表", os.path.basename(processed_data_path), processed_data_path))
            
            # 2. 将 DataFrame 传给 Phase 2
            dynamic_periods_dict, periods_path = run_tod_partition(df_processed, target_dir)
            
            if dynamic_periods_dict[DAY_TYPE_WD] and dynamic_periods_dict[DAY_TYPE_WE]:
                generated_files.append(("Fisher肘部时段划分结果", os.path.basename(periods_path), periods_path))
                
                # 3. 将 DataFrame 传给 Phase 3 (工作日)
                wd_name, wd_path = run_period_analysis(
                    df_processed, dynamic_periods_dict[DAY_TYPE_WD], target_dir, 
                    current_plan=CURRENT_SIGNAL_PLAN_WEEKDAY, day_type=DAY_TYPE_WD
                )
                generated_files.append(("工作日信控推演方案结果", wd_name, wd_path))
                
                # 4. 将 DataFrame 传给 Phase 3 (非工作日)
                we_name, we_path = run_period_analysis(
                    df_processed, dynamic_periods_dict[DAY_TYPE_WE], target_dir, 
                    current_plan=CURRENT_SIGNAL_PLAN_WEEKEND, day_type=DAY_TYPE_WE
                )
                generated_files.append(("非工作日信控推演方案结果", we_name, we_path))

                print("\n" + "=" * 80)
                print("📁 任务执行完毕！生成文件汇总清单")
                print("=" * 80)
                print(f"本次运行共生成 {len(generated_files)} 个报表文件，均已成功保存至专属目录：\n👉 {target_dir}\n")
                
                for idx, (desc, fname, fpath) in enumerate(generated_files, 1):
                    print(f"  {idx}. 【{desc}】")
                    print(f"     文件名: {fname}")
                    print(f"     绝对路径: {fpath}\n")
                print("=" * 80)