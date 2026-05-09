import pandas as pd
import numpy as np
import os
import math
import re
import matplotlib.pyplot as plt
import gc

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ================= 配置区 =================
TARGET_DATE = '2026-03-25'
INTERSECTIONS = ['光明大道与光明大街']

MIN_K = 5
MAX_K = 20
IMPROVEMENT_THRESHOLD = 0.10  

# ================= 现行方案自定义输入接口 =================
USER_CUSTOM_SCHEME_TEXT = """
00:00 - 06:00, 方案10，周期99s
06:00 - 07:30, 方案2，周期114s
07:30 - 09:00, 方案24，周期167s
09:00 - 12:00, 方案26，周期152s
12:00 - 14:00, 方案3，周期150s
14:00 - 16:00, 方案4，周期152s
16:00 - 17:00, 方案5，周期152s
17:00 - 19:00, 方案17，周期164s
19:00 - 20:30, 方案7，周期134s
20:30 - 22:00, 方案8，周期130s
22:00 - 23:59, 方案9，周期115s
"""

# ================= 基础拓扑配置 =================
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

# ================= 智能文本解析器 =================
def parse_custom_scheme(text):
    periods = []
    matches = re.findall(r'(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})', text)
    for s, e in matches:
        periods.append((s, e))
    return periods

# ================= 预处理与绘图辅助函数 =================
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
        else: df['main_direction'] = df['_raw_direction']
    else: df['main_direction'] = df['_raw_direction']
    if '_raw_direction' in df.columns: df.drop(columns=['_raw_direction'], inplace=True)
    return df

# 👇 修改后的绘图函数：加入 8 维流向堆叠面积图 👇
def plot_tod_gantt(inter_name, current_periods, opt_periods, target_date, output_dir, df_15min, target_flows):
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS'] 
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, (ax_flow, ax_gantt) = plt.subplots(2, 1, figsize=(16, 9), sharex=True, gridspec_kw={'height_ratios': [2, 1.5]})
    
    def t2h(t_str):
        if t_str in ['23:59', '24:00']: return 23.99
        h, m = map(int, t_str.split(':'))
        return h + m / 60.0
        
    colors_cycle = plt.cm.tab20.colors 
    
    # 1. 绘制流向堆叠面积图 (上层子图)
    time_hours = df_15min.index.hour + df_15min.index.minute / 60.0
    flow_stacks = [df_15min[tf].values for tf in target_flows]
    
    ax_flow.stackplot(time_hours, flow_stacks, labels=target_flows, alpha=0.8, colors=colors_cycle[:8])
    ax_flow.plot(time_hours, df_15min['5分钟车流量'], color='black', linewidth=1.5, linestyle='--', label='总流量包络线', alpha=0.6)
    
    ax_flow.set_ylabel('车流量 (pcu/15min)', fontsize=12, fontweight='bold')
    ax_flow.set_title(f'【{inter_name}】多维流向结构与时段对比图 ({target_date})', fontsize=16, fontweight='bold', pad=15)
    ax_flow.legend(loc='upper left', bbox_to_anchor=(1, 1), title="交通流向", fontsize=9)
    ax_flow.grid(axis='y', linestyle=':', alpha=0.5)

    # 2. 绘制时段甘特图 (下层子图)
    # 绘制现行方案
    for i, (s, e) in enumerate(current_periods):
        sh, eh = t2h(s), t2h(e)
        ax_gantt.broken_barh([(sh, eh-sh)], (10, 4), facecolors=colors_cycle[i%20], edgecolor='white', linewidth=1, alpha=0.7)
        if eh - sh > 0.7: 
            ax_gantt.text(sh + (eh-sh)/2, 12, f"{s}-{e}", ha='center', va='center', color='black', fontsize=9, fontweight='bold')
    
    # 绘制优化方案
    for i, (s, e) in enumerate(opt_periods):
        sh, eh = t2h(s), t2h(e)
        ax_gantt.broken_barh([(sh, eh-sh)], (2, 4), facecolors=colors_cycle[(i+5)%20], edgecolor='white', linewidth=1, alpha=0.7)
        if eh - sh > 0.7:
            ax_gantt.text(sh + (eh-sh)/2, 4, f"{s}-{e}", ha='center', va='center', color='black', fontsize=9, fontweight='bold')
            
    ax_gantt.set_ylim(0, 16)
    ax_gantt.set_xlim(0, 24)
    ax_gantt.set_yticks([4, 12])
    ax_gantt.set_yticklabels(['8维数据驱动方案', '现行时段方案'], fontsize=12, fontweight='bold')
    
    ax_gantt.set_xticks(np.arange(0, 25, 1))
    ax_gantt.set_xticklabels([f"{int(x):02d}:00" for x in np.arange(0, 25, 1)], fontsize=10)
    ax_gantt.set_xlabel('时间（24小时制）', fontsize=12, fontweight='bold')
    
    # 辅助网格
    ax_gantt.set_xticks(np.arange(0, 24.1, 0.5), minor=True)
    ax_flow.set_xticks(np.arange(0, 24.1, 0.5), minor=True)
    ax_flow.grid(axis='x', which='minor', linestyle='-', color='gray', alpha=0.1)
    ax_gantt.grid(axis='x', which='minor', linestyle='-', color='gray', alpha=0.1)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{inter_name}_多维流向时段对比图_{target_date}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    # 👇 新增：强制清理画布内存 👇
    fig.clf()                # 清空当前画布内容
    plt.close(fig)           # 关闭指定画布
    plt.close('all')         # 关闭后台所有潜在未关闭的图表
    # 👆 新增结束 👆
    
    print(f"  📊 8维流向对比图已生成: {save_path}")

# ================= Fisher 最优分割法核心引擎 =================
def fisher_optimal_partition(data_matrix, k_classes):
    n = len(data_matrix)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            segment = data_matrix[i:j+1]
            mean_vector = np.mean(segment, axis=0)
            D[i, j] = np.sum(np.linalg.norm(segment - mean_vector, axis=1)**2)
            
    dp = np.full((n, k_classes + 1), np.inf)
    split = np.zeros((n, k_classes + 1), dtype=int)
    
    for i in range(n):
        dp[i][1] = D[0, i]
        
    for m in range(2, k_classes + 1):
        for i in range(m - 1, n):
            for j in range(m - 2, i):
                cost = dp[j][m - 1] + D[j + 1, i]
                if cost < dp[i][m]:
                    dp[i][m] = cost
                    split[i][m] = j
                    
    boundaries = []
    curr = n - 1
    for m in range(k_classes, 1, -1):
        curr = split[curr][m]
        boundaries.append(curr)
        
    boundaries.reverse()
    min_cost = dp[n-1][k_classes] 
    return boundaries, min_cost

# ================= 数据转换与自动搜索主流程 =================
def run_tod_partition(file_paths):
    print("================ Fisher 肘部法则：全自动时段划分 (分流向数据驱动) ================")
    
    parsed_current_periods = parse_custom_scheme(USER_CUSTOM_SCHEME_TEXT)
    target_flows = ['东向直行', '东向左转', '南向直行', '南向左转', '西向直行', '西向左转', '北向直行', '北向左转']

    for file_path in file_paths:
        try: df = pd.read_csv(file_path, encoding='GBK')
        except: df = pd.read_csv(file_path, encoding='utf-8')
            
        if df.empty or 'inter_id' not in df.columns: continue
        current_inter_id = str(df['inter_id'].iloc[0]).strip().lower()
        if current_inter_id not in INTER_ID_NAME_MAP: continue
        inter_name = INTER_ID_NAME_MAP[current_inter_id]
        if inter_name not in INTERSECTIONS: continue
            
        print(f"\n📂 处理中: {inter_name}")
        offset = INTER_OFFSET_MAP.get(inter_name, 0.0)

        df['create_time'] = pd.to_datetime(df['create_time'], errors='coerce')
        df = df[df['create_time'].dt.strftime('%Y-%m-%d') == TARGET_DATE].copy()
        if df.empty: continue
            
        df['_temp_time'] = pd.to_datetime('2026-01-01 ' + df['create_time'].dt.strftime('%H:%M:%S'))
        df = enrich_direction_features(df, offset_degree=offset)
        df = df[df['turn_dir_no'].isin([1, 2])].copy()
        df['流向'] = df['main_direction'].map(CARDINAL_HANZI) + df['turn_dir_no'].map(TURN_HANZI)

        p_col = next((c for c in df.columns if 'pass_flow' in c.lower() or '分均流量' in c), None)
        df['5分钟车流量'] = pd.to_numeric(df[p_col].astype(str).str.extract(r'(\d+)')[0], errors='coerce').fillna(0)
        df['延误指数'] = pd.to_numeric(df.get('delay_index', 0), errors='coerce').fillna(0)
        df['最大排队长度'] = pd.to_numeric(df.get('queue_len_max', 0), errors='coerce').fillna(0)

        df_flow = df.pivot_table(index='_temp_time', columns='流向', values='5分钟车流量', aggfunc='sum').fillna(0)
        for tf in target_flows:
            if tf not in df_flow.columns: df_flow[tf] = 0
                
        df_status = df.groupby('_temp_time').agg({'5分钟车流量': 'sum', '延误指数': 'mean', '最大排队长度': 'max'}).fillna(0)
        df_time = df_flow.join(df_status, how='outer').fillna(0).sort_index()

        full_time_range = pd.date_range(start='2026-01-01 00:00:00', end='2026-01-01 23:59:59', freq='15min')
        agg_rules = {tf: 'sum' for tf in target_flows}
        agg_rules.update({'5分钟车流量': 'sum', '延误指数': 'mean', '最大排队长度': 'max'})
        df_15min = df_time.resample('15min').agg(agg_rules).reindex(full_time_range).ffill().fillna(0)

        # 核心算法处理 (保持原样)
        feature_cols = []
        for tf in target_flows:
            feat_name = f'flow_{tf}'
            df_15min[feat_name] = df_15min[tf].rolling(window=3, min_periods=1).mean()
            feature_cols.append(feat_name)
        df_15min['delay_f'] = df_15min['延误指数'].rolling(window=4, min_periods=1).mean()
        df_15min['queue_f'] = df_15min['最大排队长度'].rolling(window=4, min_periods=1).max()
        feature_cols.extend(['delay_f', 'queue_f'])

        raw_matrix = df_15min[feature_cols].values
        data_matrix = (raw_matrix - np.mean(raw_matrix, axis=0)) / (np.std(raw_matrix, axis=0) + 1e-8)

        k_results = {k: fisher_optimal_partition(data_matrix, k) for k in range(MIN_K, MAX_K + 1)}
        
        optimal_k = MIN_K
        for k in range(MIN_K + 1, MAX_K + 1):
            improvement = (k_results[k-1][1] - k_results[k][1]) / k_results[k-1][1]
            if improvement < IMPROVEMENT_THRESHOLD:
                optimal_k = k - 1
                break
        else: optimal_k = MAX_K

        best_boundaries = k_results[optimal_k][0]
        start_idx = 0
        opt_periods_for_plot = []
        for split_idx in best_boundaries + [95]:
            s_t = df_15min.index[start_idx].strftime('%H:%M')
            e_t = (df_15min.index[split_idx] + pd.Timedelta(minutes=15)).strftime('%H:%M')
            if e_t == '00:00': e_t = '23:59'
            opt_periods_for_plot.append((s_t, e_t))
            start_idx = split_idx + 1
            
        output_dir = os.path.dirname(file_path)
        # 👇 调用更新后的绘图函数，传入多维数据 👇
        plot_tod_gantt(inter_name, parsed_current_periods, opt_periods_for_plot, TARGET_DATE, output_dir, df_15min, target_flows)

        # 👇 新增：每次处理完一个表格后，暴力清空内存池 👇
        try:
            del df, df_flow, df_status, df_time, df_15min, raw_matrix, data_matrix
        except NameError:
            pass # 如果有因 continue 跳过而未定义的变量，直接忽略
        gc.collect() 
        # 👆 新增结束 👆
    print("\n" + "=" * 60)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = [os.path.join(script_dir, f) for f in os.listdir(script_dir) if "index_view" in f and f.endswith('.csv')]
    if csv_files: run_tod_partition(csv_files)