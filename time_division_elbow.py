import pandas as pd
import numpy as np
import os

# ================= 配置区 =================
# 只研究光明大街
INTERSECTIONS = ['光明大道与光明大街']

# 自动肘部法则搜索范围
MIN_K = 5
MAX_K = 20
# 边际收益阈值：增加一个时段，如果方差下降不足 10% (0.1)，则停止切分
IMPROVEMENT_THRESHOLD = 0.10  

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
    
    # 最小总离差平方和即为 dp 矩阵的右下角
    min_cost = dp[n-1][k_classes] 
    return boundaries, min_cost

# ================= 数据转换与自动搜索主流程 =================
def run_tod_partition(csv_path):
    print("================ Fisher 肘部法则：全自动时段划分 ================")
    
    df = pd.read_csv(csv_path, encoding='GBK')
    df.columns = df.columns.str.strip()
    
    required_cols = ['路口名称', 'create_time', '延误指数', '5分钟车流量', '最大排队长度']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"致命错误：CSV 中缺少必需的列 '{col}'")
            
    df['create_time'] = pd.to_datetime(df['create_time'], errors='coerce')
    df = df.dropna(subset=['create_time']).copy()
    
    target_date = df['create_time'].dt.date.mode()[0]
    print(f"[自动识别评估基准日]: {target_date}")
    
    df = df[df['create_time'].dt.date == target_date].copy()
    df['_temp_time'] = pd.to_datetime('2026-01-01 ' + df['create_time'].dt.strftime('%H:%M:%S'))

    df['延误指数'] = pd.to_numeric(df['延误指数'], errors='coerce').fillna(0)
    df['5分钟车流量'] = pd.to_numeric(df['5分钟车流量'], errors='coerce').fillna(0)
    df['最大排队长度'] = pd.to_numeric(df['最大排队长度'], errors='coerce').fillna(0)

    for inter in INTERSECTIONS:
        df_inter = df[df['路口名称'] == inter].copy()
        
        if df_inter.empty:
            continue
            
        # 空间聚合
        df_time = df_inter.groupby('_temp_time').agg({
            '5分钟车流量': 'sum',
            '延误指数': 'mean',
            '最大排队长度': 'max'
        }).sort_index()

        # 时间重采样 (96个15分钟切片)
        full_time_range = pd.date_range(start='2026-01-01 00:00:00', end='2026-01-01 23:59:59', freq='15min')
        df_15min = df_time.resample('15min').agg({
            '5分钟车流量': 'sum',           
            '延误指数': 'mean',      
            '最大排队长度': 'max'    
        }).reindex(full_time_range).ffill().fillna(0)

        # 差异化指标处理
        df_15min['flow_feature'] = df_15min['5分钟车流量'].rolling(window=3, min_periods=1, center=False).mean()
        df_15min['delay_feature'] = df_15min['延误指数'].rolling(window=4, min_periods=1, center=False).mean()
        df_15min['queue_feature'] = df_15min['最大排队长度'].rolling(window=4, min_periods=1, center=False).max()

        feature_cols = ['flow_feature', 'delay_feature', 'queue_feature']
        raw_matrix = df_15min[feature_cols].values
        data_matrix = (raw_matrix - np.mean(raw_matrix, axis=0)) / (np.std(raw_matrix, axis=0) + 1e-8)

        print(f"\n🎯 开始为交叉口【{inter}】探索最优段数 (k ∈ [{MIN_K}, {MAX_K}]) ...")
        
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
            # 如果循环一直没被 break，说明哪怕切 8 段依然有很大收益，默认取最大边界
            optimal_k = MAX_K
            print(f"  💡 交通流剧烈波动，误差持续大幅下降，采用设定的上限切分段数。")

        print(f"\n✅ 算法最终判定最优划分段数：{optimal_k} 段")
        print("-" * 50)
        
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
            
            print(f"  [时段 {i+1}] {start_time} - {end_time} | 期间平均15min总流量: {avg_flow} pcu | 期间最大排队极值: {max_q} m")
            
            start_idx = split_idx + 1
    print("\n" + "=" * 60)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_dir, "干线全流向_延误与流量占比明细表.csv") 
    try:
        run_tod_partition(csv_file_path)
    except Exception as e:
        print(f"脚本执行异常: {e}")