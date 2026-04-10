import pandas as pd
import os

def health_check_v4(csv_path):
    print("=" * 60)
    print(f"正在读取数据进行深度健康诊断: {os.path.basename(csv_path)}")
    
    # 兼容多种编码读取
    try: 
        df = pd.read_csv(csv_path, encoding='GBK')
    except:
        try: 
            df = pd.read_csv(csv_path, encoding='utf-8')
        except: 
            df = pd.read_csv(csv_path, encoding='latin1')

    print(f"数据总行数: {len(df)}")

    # 核心字段依赖检查
    required_cols = ['create_time', 'turn_dir_no']
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ 严重错误: 缺失必须字段 '{col}'，无法进行诊断！")
            return

    # 时间预处理
    df['create_time'] = pd.to_datetime(df['create_time'], errors='coerce')
    df = df.dropna(subset=['create_time'])  # 去除解析失败的脏时间数据
    df['minute_time'] = df['create_time'].dt.floor('min')
    df['date'] = df['minute_time'].dt.date

    # 获取全集时间范围
    min_date = df['date'].min()
    max_date = df['date'].max()
    all_dates = pd.date_range(start=min_date, end=max_date, freq='D').date
    total_days = len(all_dates)
    expected_turns = [0, 1, 2, 3]

    print(f"当前数据集跨度: {min_date} 至 {max_date} (共 {total_days} 天)")

    # ================= 0. 宏观整体数据缺失率 =================
    print("\n--- [0] 宏观整体数据缺失率诊断 ---")
    # 理论总记录数 = 总天数 * 1440分钟 * 16行 (4进口道 x 4指标)
    theoretical_total_records = total_days * 1440 * 16
    # 实际有效记录数 = 过滤出合法的转向指标(0,1,2,3)后的行数
    actual_valid_records = len(df[df['turn_dir_no'].isin(expected_turns)])
    
    print(f"理论应有总记录数: {theoretical_total_records} ( {total_days}天 x 1440分钟 x 16行/分钟 )")
    print(f"实际有效总记录数: {actual_valid_records}")
    
    if actual_valid_records <= theoretical_total_records:
        overall_missing_rate = (theoretical_total_records - actual_valid_records) / theoretical_total_records
        if overall_missing_rate == 0:
            print(f"🌟 整体数据缺失率: 0.00% (完美全量数据)")
        else:
            print(f"🚨 整体数据缺失率: {overall_missing_rate:.2%}")
    else:
        expansion_rate = (actual_valid_records - theoretical_total_records) / theoretical_total_records
        print(f"⚠️ 整体数据缺失率: 0.00% (但数据量超出理论值，存在重复膨胀，超出率: {expansion_rate:.2%})")

    # ================= 1. 时间序列诊断 =================
    print(f"\n--- [1] 时间序列完整性诊断 (按天) ---")
    daily_minutes = df.groupby('date')['minute_time'].nunique()

    missing_time_found = False
    for d in all_dates:
        actual_mins = daily_minutes.get(d, 0)
        missing_rate = (1440 - actual_mins) / 1440
        
        if missing_rate > 0:
            missing_time_found = True
            if actual_mins == 0:
                print(f"  🚨 [ {d} ] 数据完全缺失 (缺失分钟率 100.00%)")
            else:
                print(f"  ⚠️ [ {d} ] 缺失分钟率: {missing_rate:.2%} | 已存分钟数: {actual_mins}/1440")
    
    if not missing_time_found:
        print("  ✅ 完美！所有日期均具备完整的 1440 分钟数据，无时间断层。")

    # ================= 2. 转向结构诊断 =================
    print("\n--- [2] 转向数据(turn_dir_no) 结构完整性诊断 ---")
    print("规则判定: 对于存在的每一分钟，单种转向指标(如左转)应具备 4 行数据(对应4个进口道)")
    turn_mapping = {0: '0-方向聚合', 1: '1-左转', 2: '2-直行', 3: '3-右转'}
    
    pivot_turn = df.pivot_table(
        index='date', 
        columns='turn_dir_no', 
        values='create_time', 
        aggfunc='count', 
        fill_value=0
    )

    issue_found = False
    for d in all_dates:
        actual_mins = daily_minutes.get(d, 0)
        if actual_mins == 0:
            continue
        
        # 动态基准：当前实际存在的分钟数 * 4 个进口道
        # (即：如果这一天只有 1000 分钟有数据，那么“1-左转”这个指标理论上应该有 1000 * 4 = 4000 行)
        expected_rows_per_turn = actual_mins * 4 
        missing_turns_for_day = []
        
        for t_no in expected_turns:
            actual_rows = pivot_turn.loc[d, t_no] if (d in pivot_turn.index and t_no in pivot_turn.columns) else 0
            
            if actual_rows < expected_rows_per_turn:
                missing_turns_for_day.append(f"{turn_mapping[t_no]:<10} (应有: {expected_rows_per_turn:<5} | 实际: {actual_rows:<5})")
            elif actual_rows > expected_rows_per_turn:
                missing_turns_for_day.append(f"{turn_mapping[t_no]:<10} (⚠️重复冗余, 预期 {expected_rows_per_turn}, 实际 {actual_rows})")

        if missing_turns_for_day:
            issue_found = True
            print(f"  [ 日期: {d} ] 发现转向结构不完整/异常:")
            for msg in missing_turns_for_day:
                print(f"      -> {msg}")

    if not issue_found:
        print("  ✅ 完美！现有时间戳内，每个转向指标均具备完整的 4 个进口道数据，无缺失或重复。")
        
    print("=" * 60)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = [f for f in os.listdir(script_dir) if f.endswith('.csv')]
    
    dir_csv = next((os.path.join(script_dir, f) for f in csv_files if "index_view" in f), None)
    
    if dir_csv:
        health_check_v4(dir_csv)
    else:
        print("未找到包含 'index_view' 的 CSV 文件，请检查同一目录下是否存在。")