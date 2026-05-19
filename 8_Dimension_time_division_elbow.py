import pandas as pd
import numpy as np
import os
import math
import re
import matplotlib.pyplot as plt

# ================= 配置区 =================
TARGET_DATE = '2026-03-25'
INTERSECTIONS = ['光明大道与光明大街']

MIN_K = 5
MAX_K = 20
IMPROVEMENT_THRESHOLD = 0.10  

# ================= 现行方案自定义输入接口 =================
CURRENT_SIGNAL_PLAN = {
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

# ================= 智能数据解析器 =================
def parse_custom_scheme(plan_dict):
    periods = []
    if '现行周期方案' in plan_dict:
        for item in plan_dict['现行周期方案']:
            periods.append((item['start'], item['end']))
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

def plot_tod_gantt(inter_name, current_periods, opt_periods, target_date, output_dir):
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS'] 
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(16, 5))
    
    def t2h(t_str):
        if t_str in ['23:59', '24:00']: return 24.0
        h, m = map(int, t_str.split(':'))
        return h + m / 60.0
        
    colors = plt.cm.tab20.colors 
    
    # 绘制现行方案 (上层)
    for i, (s, e) in enumerate(current_periods):
        sh, eh = t2h(s), t2h(e)
        ax.broken_barh([(sh, eh-sh)], (15, 6), facecolors=colors[i%20], edgecolor='white', linewidth=1.5, alpha=0.85)
        if eh - sh > 0.8: 
            ax.text(sh + (eh-sh)/2, 18, f"{s}-{e}", ha='center', va='center', color='black', fontsize=9.5, fontweight='bold')
    
    # 绘制优化方案 (下层)
    for i, (s, e) in enumerate(opt_periods):
        sh, eh = t2h(s), t2h(e)
        ax.broken_barh([(sh, eh-sh)], (5, 6), facecolors=colors[i%20], edgecolor='white', linewidth=1.5, alpha=0.85)
        if eh - sh > 0.8:
            ax.text(sh + (eh-sh)/2, 8, f"{s}-{e}", ha='center', va='center', color='black', fontsize=9.5, fontweight='bold')
            
    ax.set_ylim(0, 25)
    ax.set_xlim(0, 24)
    ax.set_yticks([8, 18])
    ax.set_yticklabels(['分流向数据驱动方案', '现行时段方案'], fontsize=12, fontweight='bold')
    
    ax.set_xticks(np.arange(0, 25, 1))
    ax.set_xticklabels([f"{int(x):02d}:00" for x in np.arange(0, 25, 1)], rotation=0, fontsize=10)
    ax.set_xlabel('时间（24小时制）', fontsize=12, fontweight='bold')
    
    ax.set_xticks(np.arange(0, 24.1, 0.5), minor=True)
    ax.grid(axis='x', which='major', linestyle='-', color='gray', alpha=0.4)
    ax.grid(axis='x', which='minor', linestyle=':', color='gray', alpha=0.2)
    
    ax.set_title(f'【{inter_name}】时段划分对比图 ({target_date})', fontsize=16, fontweight='bold', pad=15)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{inter_name}_时段划分对比甘特图_{target_date}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  📊 对比甘特图已生成: {save_path}")

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
    
    parsed_current_periods = parse_custom_scheme(CURRENT_SIGNAL_PLAN)
    if parsed_current_periods:
        print(f"✅ 成功从配置区解析出 {len(parsed_current_periods)} 个现行时段设置。")
    else:
        print("⚠️ 未从 CURRENT_SIGNAL_PLAN 提取到有效的时段格式，请检查。")

    target_flows = ['东向直行', '东向左转', '南向直行', '南向左转', '西向直行', '西向左转', '北向直行', '北向左转']

    for file_path in file_paths:
        try: df = pd.read_csv(file_path, encoding='GBK')
        except: df = pd.read_csv(file_path, encoding='utf-8')
            
        if df.empty or 'inter_id' not in df.columns: continue
            
        current_inter_id = str(df['inter_id'].iloc[0]).strip().lower()
        if current_inter_id not in INTER_ID_NAME_MAP: continue
        inter_name = INTER_ID_NAME_MAP[current_inter_id]
        if inter_name not in INTERSECTIONS: continue
            
        print(f"\n📂 正在处理路口【{inter_name}】的底层表: {os.path.basename(file_path)}")
        offset = INTER_OFFSET_MAP.get(inter_name, 0.0)

        df['create_time'] = pd.to_datetime(df['create_time'], errors='coerce')
        df = df.dropna(subset=['create_time']).copy()
        
        df = df[df['create_time'].dt.strftime('%Y-%m-%d') == TARGET_DATE].copy()
        if df.empty:
            print(f"⚠️ 路口【{inter_name}】在指定日期 {TARGET_DATE} 无数据，跳过。")
            continue
            
        df['_temp_time'] = pd.to_datetime('2026-01-01 ' + df['create_time'].dt.strftime('%H:%M:%S'))

        df = enrich_direction_features(df, offset_degree=offset)
        if 'turn_dir_no' in df.columns:
            df = df[df['turn_dir_no'].isin([1, 2])].copy()
        else: continue
            
        df['流向'] = df['main_direction'].map(CARDINAL_HANZI) + df['turn_dir_no'].map(TURN_HANZI)

        p_col = next((c for c in df.columns if 'pass_flow' in c.lower() or '分均流量' in c), None)
        if p_col:
            df['5分钟车流量'] = df[p_col].astype(str).str.replace(',', '', regex=False).str.extract(r'(\d+\.?\d*)')[0]
            df['5分钟车流量'] = pd.to_numeric(df['5分钟车流量'], errors='coerce').fillna(0)
        else: df['5分钟车流量'] = 0

        df['延误指数'] = pd.to_numeric(df.get('delay_index', 0), errors='coerce').fillna(0)
        df['最大排队长度'] = pd.to_numeric(df.get('queue_len_max', 0), errors='coerce').fillna(0)

        df_flow = df.pivot_table(index='_temp_time', columns='流向', values='5分钟车流量', aggfunc='sum').fillna(0)
        for tf in target_flows:
            if tf not in df_flow.columns: df_flow[tf] = 0
                
        df_status = df.groupby('_temp_time').agg({
            '5分钟车流量': 'sum', 
            '延误指数': 'mean',
            '最大排队长度': 'max'
        }).fillna(0)

        df_time = df_flow.join(df_status, how='outer').fillna(0).sort_index()

        full_time_range = pd.date_range(start='2026-01-01 00:00:00', end='2026-01-01 23:59:59', freq='15min')
        agg_rules = {'5分钟车流量': 'sum', '延误指数': 'mean', '最大排队长度': 'max'}
        for tf in target_flows: agg_rules[tf] = 'sum'
            
        df_15min = df_time.resample('15min').agg(agg_rules).reindex(full_time_range).ffill().fillna(0)

        feature_cols = []
        for tf in target_flows:
            feat_name = f'flow_feature_{tf}'
            df_15min[feat_name] = df_15min[tf].rolling(window=3, min_periods=1, center=False).mean()
            feature_cols.append(feat_name)
            
        df_15min['delay_feature'] = df_15min['延误指数'].rolling(window=4, min_periods=1, center=False).mean()
        df_15min['queue_feature'] = df_15min['最大排队长度'].rolling(window=4, min_periods=1, center=False).max()
        feature_cols.extend(['delay_feature', 'queue_feature'])

        raw_matrix = df_15min[feature_cols].values
        data_matrix = (raw_matrix - np.mean(raw_matrix, axis=0)) / (np.std(raw_matrix, axis=0) + 1e-8)

        print(f"🎯 开始为【{inter_name}】探索最优切分段数 (k ∈ [{MIN_K}, {MAX_K}])，基于指定日期：{TARGET_DATE}")
        k_results = {}
        for k in range(MIN_K, MAX_K + 1):
            boundaries, cost = fisher_optimal_partition(data_matrix, k)
            k_results[k] = {'boundaries': boundaries, 'cost': cost}
            
        # 👇 恢复的打印输出部分 👇
        optimal_k = MIN_K
        print(f"  [推演记录] k={MIN_K} | 总方差误差: {k_results[MIN_K]['cost']:.2f}")
        for k in range(MIN_K + 1, MAX_K + 1):
            prev_cost = k_results[k-1]['cost']
            curr_cost = k_results[k]['cost']
            improvement_ratio = (prev_cost - curr_cost) / prev_cost
            print(f"  [推演记录] k={k} | 总方差误差: {curr_cost:.2f} | 误差下降比例: {improvement_ratio*100:.1f}%")
            if improvement_ratio < IMPROVEMENT_THRESHOLD:
                print(f"  💡 触发边际收益衰减！从 {k-1} 段增加到 {k} 段带来的优化已不足 {IMPROVEMENT_THRESHOLD*100}%。")
                optimal_k = k - 1
                break
        else:
            optimal_k = MAX_K
            print(f"  💡 交通流极度剧烈波动，误差持续大幅下降，采用设定的上限切分段数。")
        # 👆 恢复的打印输出部分 👆

        print(f"\n✅ 算法最终判定最优划分段数：{optimal_k} 段")
        print("-" * 60)
        
        best_boundaries = k_results[optimal_k]['boundaries']
        start_idx = 0
        opt_periods_for_plot = []  
        
        for i, split_idx in enumerate(best_boundaries + [95]):
            start_time = df_15min.index[start_idx].strftime('%H:%M')
            end_time = (df_15min.index[split_idx] + pd.Timedelta(minutes=15)).strftime('%H:%M')
            if end_time == '00:00': end_time = '23:59'
            
            opt_periods_for_plot.append((start_time, end_time))
            
            period_data = df_15min.iloc[start_idx:split_idx+1]
            avg_flow = int(period_data['5分钟车流量'].mean())
            max_q = round(period_data['最大排队长度'].max(), 1)
            
            print(f"  [时段 {i+1:02d}] {start_time} - {end_time} | 期间平均15min总流量: {avg_flow} pcu | 期间最大排队极值: {max_q} m")
            start_idx = split_idx + 1
            
        print("-" * 60)
        
        if parsed_current_periods:
            output_dir = os.path.dirname(file_path)
            plot_tod_gantt(inter_name, parsed_current_periods, opt_periods_for_plot, TARGET_DATE, output_dir)

    print("\n" + "=" * 60)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = [os.path.join(script_dir, f) for f in os.listdir(script_dir) if "index_view" in f and f.endswith('.csv')]
    
    if not csv_files:
        print("❌ 当前目录下未找到包含 'index_view' 的数据表文件！")
    else:
        try:
            run_tod_partition(csv_files)
        except Exception as e:
            print(f"脚本执行异常: {e}")