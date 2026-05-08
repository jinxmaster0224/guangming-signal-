import pandas as pd
import numpy as np
import os
import math
import re

# ================= 配置区 =================
TARGET_DATE = '2026-03-25'

# 
INTERSECTIONS = ['光明大道与光明大街']

# 自动肘部法则搜索范围
MIN_K = 5
MAX_K = 20
# 边际收益阈值：增加一个时段，如果方差下降不足 10% (0.1)，则停止切分
IMPROVEMENT_THRESHOLD = 0.10  

# 【新增】路口 ID 与名称映射字典
INTER_ID_NAME_MAP = {
    '6caaa6f15b735a': '光明大道与光辉大道',
    '6caa3ec15b566e': '光明大道与光明大街',
    '6caa06b15b50eb': '光明大道与光安路',
    '6ca93d215b3ee3': '光明大道与华夏路',
    '6ca887115b2e22': '光明大道与华裕路'
}

# 【新增】路口偏移角配置
INTER_OFFSET_MAP = {
    '光明大道与光辉大道': 0.0,
    '光明大道与光明大街': -32.57,
    '光明大道与光安路': -33.69,
    '光明大道与华夏路': -63.5,
    '光明大道与华裕路': -63.0,
}

CARDINAL_HANZI = {"E": "东向", "W": "西向", "S": "南向", "N": "北向"}
TURN_HANZI = {1: "左转", 2: "直行"}

# ================= 预处理辅助函数 =================
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

# ================= Fisher 最优分割法核心引擎 =================
def fisher_optimal_partition(data_matrix, k_classes):
    """
    返回最优边界，以及当前 k 划分下的最小总离差平方和 (cost)
    """
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
    print("================ Fisher 肘部法则：全自动时段划分 ================")
    
    # 预设的 8 个核心维度流向
    target_flows = ['东向直行', '东向左转', '南向直行', '南向左转', '西向直行', '西向左转', '北向直行', '北向左转']

    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path, encoding='GBK')
        except:
            df = pd.read_csv(file_path, encoding='utf-8')
            
        if df.empty or 'inter_id' not in df.columns:
            continue
            
        # 1. 匹配底层表与路口名称
        current_inter_id = str(df['inter_id'].iloc[0]).strip().lower()
        if current_inter_id not in INTER_ID_NAME_MAP:
            continue
        inter_name = INTER_ID_NAME_MAP[current_inter_id]
        if inter_name not in INTERSECTIONS:
            continue
            
        print(f"\n📂 正在处理路口【{inter_name}】的底层表: {os.path.basename(file_path)}")
        offset = INTER_OFFSET_MAP.get(inter_name, 0.0)

        # 2. 截取特定时长的数据 (按 TARGET_DATE 剥离单日数据)
        df['create_time'] = pd.to_datetime(df['create_time'], errors='coerce')
        df = df.dropna(subset=['create_time']).copy()
        
        # 过滤出定义的日期
        df = df[df['create_time'].dt.strftime('%Y-%m-%d') == TARGET_DATE].copy()
        
        if df.empty:
            print(f"⚠️ 路口【{inter_name}】在指定日期 {TARGET_DATE} 无数据，跳过。")
            continue
            
        # 统一映射到基准时间以便重采样
        df['_temp_time'] = pd.to_datetime('2026-01-01 ' + df['create_time'].dt.strftime('%H:%M:%S'))

        # 3. 物理方向解析与转向过滤
        df = enrich_direction_features(df, offset_degree=offset)
        
        # 只保留 1-左转 和 2-直行 (忽略0和3)
        if 'turn_dir_no' in df.columns:
            df = df[df['turn_dir_no'].isin([1, 2])].copy()
        else:
            print("⚠️ 数据中缺失 'turn_dir_no' 列，无法区分转向，跳过该表。")
            continue
            
        # 拼接物理流向中文名
        df['流向'] = df['main_direction'].map(CARDINAL_HANZI) + df['turn_dir_no'].map(TURN_HANZI)

        # 提取底层基础指标
        p_col = next((c for c in df.columns if 'pass_flow' in c.lower() or '分均流量' in c), None)
        if p_col:
            df['5分钟车流量'] = df[p_col].astype(str).str.replace(',', '', regex=False).str.extract(r'(\d+\.?\d*)')[0]
            df['5分钟车流量'] = pd.to_numeric(df['5分钟车流量'], errors='coerce').fillna(0)
        else:
            df['5分钟车流量'] = 0

        df['延误指数'] = pd.to_numeric(df.get('delay_index', 0), errors='coerce').fillna(0)
        df['最大排队长度'] = pd.to_numeric(df.get('queue_len_max', 0), errors='coerce').fillna(0)

        # 4. 空间聚合 - 剥离 8 维独立流量特征
        df_flow = df.pivot_table(index='_temp_time', columns='流向', values='5分钟车流量', aggfunc='sum').fillna(0)
        
        # 补齐缺漏的流向 (防止某天缺数导致矩阵维度不满 8 维)
        for tf in target_flows:
            if tf not in df_flow.columns:
                df_flow[tf] = 0
                
        # 提取交叉口宏观状态用于综合评判与打印
        df_status = df.groupby('_temp_time').agg({
            '5分钟车流量': 'sum', 
            '延误指数': 'mean',
            '最大排队长度': 'max'
        }).fillna(0)

        # 拼接完整的宽表
        df_time = df_flow.join(df_status, how='outer').fillna(0).sort_index()

        # 5. 时间重采样 (强制对齐为96个15分钟切片) 
        full_time_range = pd.date_range(start='2026-01-01 00:00:00', end='2026-01-01 23:59:59', freq='15min')
        
        agg_rules = {
            '5分钟车流量': 'sum',
            '延误指数': 'mean',
            '最大排队长度': 'max'
        }
        for tf in target_flows:
            agg_rules[tf] = 'sum'
            
        df_15min = df_time.resample('15min').agg(agg_rules).reindex(full_time_range).ffill().fillna(0)

        # 6. 特征处理 - 对 8 个流向分别做趋势平滑
        feature_cols = []
        for tf in target_flows:
            feat_name = f'flow_feature_{tf}'
            df_15min[feat_name] = df_15min[tf].rolling(window=3, min_periods=1, center=False).mean()
            feature_cols.append(feat_name)
            
        df_15min['delay_feature'] = df_15min['延误指数'].rolling(window=4, min_periods=1, center=False).mean()
        df_15min['queue_feature'] = df_15min['最大排队长度'].rolling(window=4, min_periods=1, center=False).max()
        
        # 将延误和排队压入特征矩阵，总计 10 个特征
        feature_cols.extend(['delay_feature', 'queue_feature'])

        # 核心：10维矩阵的 Z-score 无量纲化
        raw_matrix = df_15min[feature_cols].values
        data_matrix = (raw_matrix - np.mean(raw_matrix, axis=0)) / (np.std(raw_matrix, axis=0) + 1e-8)

        print(f"🎯 开始为【{inter_name}】探索最优切分段数 (k ∈ [{MIN_K}, {MAX_K}])，基于指定日期：{TARGET_DATE}")
        
        # 记录不同 k 值下的运算结果
        k_results = {}
        for k in range(MIN_K, MAX_K + 1):
            boundaries, cost = fisher_optimal_partition(data_matrix, k)
            k_results[k] = {'boundaries': boundaries, 'cost': cost}
            
        # 寻找肘部：计算边际收益 (方差下降百分比)
        optimal_k = MIN_K
        print(f"  [推演记录] k={MIN_K} | 总方差误差: {k_results[MIN_K]['cost']:.2f}")
        
        for k in range(MIN_K + 1, MAX_K + 1):
            prev_cost = k_results[k-1]['cost']
            curr_cost = k_results[k]['cost']
            improvement_ratio = (prev_cost - curr_cost) / prev_cost
            
            print(f"  [推演记录] k={k} | 总方差误差: {curr_cost:.2f} | 误差下降比例: {improvement_ratio*100:.1f}%")
            
            # 如果下降比例小于阈值，说明上一轮的 k 就是“肘部”
            if improvement_ratio < IMPROVEMENT_THRESHOLD:
                print(f"  💡 触发边际收益衰减！从 {k-1} 段增加到 {k} 段带来的优化已不足 {IMPROVEMENT_THRESHOLD*100}%。")
                optimal_k = k - 1
                break
        else:
            optimal_k = MAX_K
            print(f"  💡 交通流极度剧烈波动，误差持续大幅下降，采用设定的上限切分段数。")

        print(f"\n✅ 算法最终判定最优划分段数：{optimal_k} 段")
        print("-" * 60)
        
        # 提取最优 k 对应的边界进行时间表输出
        best_boundaries = k_results[optimal_k]['boundaries']
        start_idx = 0
        for i, split_idx in enumerate(best_boundaries + [95]):
            start_time = df_15min.index[start_idx].strftime('%H:%M')
            end_time = (df_15min.index[split_idx] + pd.Timedelta(minutes=15)).strftime('%H:%M')
            if end_time == '00:00': end_time = '23:59'
            
            period_data = df_15min.iloc[start_idx:split_idx+1]
            avg_flow = int(period_data['5分钟车流量'].mean())
            max_q = round(period_data['最大排队长度'].max(), 1)
            
            print(f"  [时段 {i+1:02d}] {start_time} - {end_time} | 期间平均15min总流量: {avg_flow} pcu | 期间最大排队极值: {max_q} m")
            
            start_idx = split_idx + 1
    print("\n" + "=" * 60)

if __name__ == "__main__":
    # 自动检索当前目录下的所有 index_view 文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = [os.path.join(script_dir, f) for f in os.listdir(script_dir) if "index_view" in f and f.endswith('.csv')]
    
    if not csv_files:
        print("❌ 当前目录下未找到包含 'index_view' 的数据表文件！")
    else:
        try:
            run_tod_partition(csv_files)
        except Exception as e:
            print(f"脚本执行异常: {e}")