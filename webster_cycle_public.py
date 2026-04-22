import pandas as pd
import numpy as np
import os
import math
import re

# ================= 配置区 =================
# 1. 自定义分析日期接口
TARGET_DATE = '2026-03-25'  

# 2. 五路口信控方案接口
CURRENT_SIGNAL_PLANS = {
    '光明大道与华夏路': {
        '现行周期': 163,
        '相位时间分配': {'北向全放': 49, '南向全放': 49, '东向全放': 33, '西向全放': 32},
        '相位损失时间': {'北向全放': 9, '南向全放': 9, '东向全放': 7, '西向全放': 7},
        '专属相位映射': {
            '相位1(北向全放)': [('北向', '直行'), ('北向', '左转')],
            '相位2(南向全放)': [('南向', '直行'), ('南向', '左转')],
            '相位3(东向全放)': [('东向', '直行'), ('东向', '左转')],
            '相位4(西向全放)': [('西向', '直行'), ('西向', '左转')]
        }
    },
    '光明大道与华裕路': {
        '现行周期': 176,
        '相位时间分配': {'东向全放': 35, '东西直行同放': 29, '西向全放': 46, '南向全放': 38, '北向全放': 28},
        '相位损失时间': {'东向全放': 7, '东西直行同放': 5, '西向全放': 7, '南向全放': 9, '北向全放': 9},
        '专属相位映射': {
            '相位1(东向全放)': [('东向', '左转')],
            '相位2(东西直行同放)': [('东向', '直行'), ('西向', '直行')],
            '相位3(西向全放)': [('西向', '左转')],
            '相位4(南向全放)': [('南向', '直行'), ('南向', '左转')],
            '相位5(北向全放)': [('北向', '直行'), ('北向', '左转')]
        }
    }
}

# 3. 早高峰研究时段切片接口 (15分钟步长)
TIME_SLICES = {
    1: ('07:30', '07:45'),
    2: ('07:45', '08:00'),
    3: ('08:00', '08:15'),
    4: ('08:15', '08:30'),
    5: ('08:30', '08:45'),
    6: ('08:45', '09:00')
}

# [操作区]：输入你想研究的时段编号 (1-6)
TARGET_SLICE_ID = 3

# ================= 核心算法模块 =================

def calculate_hcm_delay(C, lambda_i, x, c, T=0.25, k=0.5, I=1.0):
    """预测单个相位的 HCM 近饱和延误公式 (T=0.25h)"""
    if lambda_i <= 0 or c <= 0:
        return 9999.0
    d1 = (0.5 * C * (1 - lambda_i)**2) / (1 - min(1.0, x) * lambda_i)
    inside_sqrt = max(0, (x - 1)**2 + (8 * k * I * x) / (c * T))
    d2 = 900 * T * ((x - 1) + math.sqrt(inside_sqrt))
    return d1 + d2

def brute_force_optimal_cycle(L_total, phase_data, T=0.25):
    """全局暴力搜索最优周期 (等饱和度分配法)"""
    Y_total = sum([data['y_val'] for data in phase_data.values()])
    total_q_5min = sum([data['q_5min'] for data in phase_data.values()])
    
    if Y_total <= 0:
        return None, None, {}

    min_total_delay = float('inf')
    best_C = None
    best_allocation = {}

    # 如果需要强制拉长干线周期，可修改为 range(120, 181)
    for C in range(60, 181):
        G = C - L_total
        if G <= 0:
            continue
            
        current_cycle_delay = 0
        current_allocation = {}
        valid_cycle = True
        
        for phase, data in phase_data.items():
            # 使用需求比 y 分配绿灯
            y_ratio = data['y_val'] / Y_total
            g_i = G * y_ratio
            lambda_i = g_i / C
            
            # 最低安全绿灯时间防线
            if g_i < 12.0 or lambda_i <= 0.01:
                current_cycle_delay += 9999.0
                continue
                
            s_real_h = data['s_real_h']
            q_h = data['q_5min'] * 12
            
            if s_real_h <= 0:
                valid_cycle = False
                break
                
            c = s_real_h * lambda_i
            x = q_h / c if c > 0 else 9999.0
            
            d_i = calculate_hcm_delay(C, lambda_i, x, c, T)
            
            # 使用绝对流量比加权评估路口综合延误
            volume_weight = data['q_5min'] / total_q_5min if total_q_5min > 0 else 0
            current_cycle_delay += d_i * volume_weight
            
            current_allocation[phase] = {
                '有效绿灯(s)': round(g_i, 1),
                '绿信比': round(lambda_i, 3),
                '饱和度': round(x, 3),
                '延误(s/veh)': round(d_i, 2)
            }
            
        if not valid_cycle:
            continue
            
        if current_cycle_delay < min_total_delay:
            min_total_delay = current_cycle_delay
            best_C = C
            best_allocation = current_allocation

    return best_C, min_total_delay, best_allocation

# ================= 数据预处理 =================

def load_and_preprocess_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到输入文件: {csv_path}")
    df = pd.read_csv(csv_path, encoding='GBK')
    df.columns = df.columns.str.strip()
    
    if 'create_time' in df.columns:
        df['_parsed_time'] = pd.to_datetime(df['create_time'], errors='coerce')
        df['_date_str'] = df['_parsed_time'].dt.strftime('%Y-%m-%d')
        df = df[df['_date_str'] == TARGET_DATE].copy()
        df['_temp_time'] = pd.to_datetime('2026-01-01 ' + df['_parsed_time'].dt.strftime('%H:%M:%S'))
    elif '时间' in df.columns:
        df['_temp_time'] = pd.to_datetime('2026-01-01 ' + df['时间'].astype(str).str.strip())

    df['延误指数'] = pd.to_numeric(df['延误指数'], errors='coerce').fillna(0)
    p_col = next((c for c in df.columns if 'pass_flow' in c.lower() or '分均流量' in c), None)
    if p_col:
        df['pass_flow'] = pd.to_numeric(df[p_col].astype(str).str.replace(',', '', regex=False).str.extract(r'(\d+\.?\d*)')[0], errors='coerce').fillna(0)
    else:
        df['pass_flow'] = 0

    if '历史最大分均流量' in df.columns:
        df['历史最大分均流量'] = pd.to_numeric(df['历史最大分均流量'], errors='coerce').fillna(0)
    else:
        df['历史最大分均流量'] = 0

    df['进口道方向'] = df['进口道方向'].astype(str).str.strip()
    df['转向'] = df['转向'].astype(str).str.strip()
    return df[~df['进口道方向'].str.contains('聚合') & ~df['转向'].str.contains('聚合')]

# ================= 主流程 =================

def run_analysis(df):
    if TARGET_SLICE_ID not in TIME_SLICES:
        print(f"[错误] 无效的时段编号: {TARGET_SLICE_ID}。请在配置区输入 1-6 之间的整数。")
        return
        
    start_time, end_time = TIME_SLICES[TARGET_SLICE_ID]
    period_name = f"切片 {TARGET_SLICE_ID} ({start_time} - {end_time})"
    
    print(f"\n[评估基准日]: {TARGET_DATE}")
    print(f"[研究时段]: {period_name}")
    print("-" * 60)

    active_intersections = list(CURRENT_SIGNAL_PLANS.keys())

    start_dt = pd.to_datetime(f'2026-01-01 {start_time}')
    end_dt = pd.to_datetime(f'2026-01-01 {end_time}')
    
    df_period = df[(df['_temp_time'] >= start_dt) & 
                   (df['_temp_time'] <= end_dt) & 
                   (df['路口名称'].isin(active_intersections))]
                   
    if df_period.empty: 
        print(f"[异常] 当前时段 {period_name} 无数据记录。")
        return
        
    inter_evals = []
    for inter, group_inter in df_period.groupby('路口名称'):
        flow_stats = group_inter.groupby(['进口道方向', '转向']).agg(
            平均延误指数=('延误指数', 'mean'),  
            平均分均流量=('pass_flow', 'mean'),
            历史最大分均流量=('历史最大分均流量', 'first') 
        ).reset_index()
        
        total_flow = flow_stats['平均分均流量'].sum()
        flow_stats['流量权重'] = flow_stats['平均分均流量'] / total_flow if total_flow > 0 else 0.0
        flow_stats['加权延误'] = flow_stats['平均延误指数'] * flow_stats['流量权重']
        
        inter_evals.append({
            '路口名称': inter,
            '加权总延误': flow_stats['加权延误'].sum(),
            '_flow_stats': flow_stats
        })
        
    if not inter_evals: 
        return
        
    df_eval = pd.DataFrame(inter_evals).sort_values(by=['加权总延误'], ascending=[False])
    critical_inter = df_eval.iloc[0]['路口名称']
    
    current_plan = CURRENT_SIGNAL_PLANS.get(critical_inter, {})
    cur_cycle = current_plan.get('现行周期', 120)
    cur_alloc = current_plan.get('相位时间分配', {})
    cur_loss = current_plan.get('相位损失时间', {})
    cur_phase_mapping = current_plan.get('专属相位映射', {})
    
    L_total = sum(cur_loss.values()) if cur_loss else 16
    critical_flow_stats = df_eval.iloc[0]['_flow_stats']
    phase_data = {}
    
    for phase_name, directions in cur_phase_mapping.items():
        max_y_val = -1
        crit_q = 0
        crit_s_real = 0
        
        for d, t in directions:
            mov = critical_flow_stats[(critical_flow_stats['进口道方向'] == d) & (critical_flow_stats['转向'] == t)]
            if not mov.empty:
                q = mov['平均分均流量'].iloc[0]
                s = mov['历史最大分均流量'].iloc[0]
                
                if s > 0:
                    # 循环外解封还原物理饱和流率
                    match = re.search(r'\((.*?)\)', phase_name)
                    old_lambda = 0.2  
                    if match:
                        raw_name = match.group(1)
                        alloc_t = cur_alloc.get(raw_name, 0)
                        loss_t = cur_loss.get(raw_name, 0)
                        old_g = alloc_t - loss_t
                        if old_g > 0 and cur_cycle > 0:
                            old_lambda = old_g / cur_cycle
                    
                    s_real_h = (s * 12) / old_lambda
                    q_h = q * 12
                    
                    # 计算需求流量比 y
                    y_val = q_h / s_real_h
                    
                    if y_val > max_y_val:
                        max_y_val = y_val
                        crit_q = q
                        crit_s_real = s_real_h
        
        if crit_s_real > 0:
            phase_data[phase_name] = {
                'q_5min': crit_q, 
                's_real_h': crit_s_real,
                'y_val': max_y_val
            }
            
    if phase_data:
        best_c, min_delay, allocation = brute_force_optimal_cycle(L_total, phase_data)
        
        print(f"🎯 关键路口定位: {critical_inter}")
        print(f"\n[输出产物]: 最小延误-周期方案")
        print(f"最小加权预测延误: {min_delay:.2f} s/veh")
        print(f"最优公共周期: {best_c} s")
        print("各相位绿信比分配:")
        for p, metrics in allocation.items():
            print(f"  - {p} -> 绿灯: {metrics['有效绿灯(s)']}s | 绿信比: {metrics['绿信比']} | 饱和度: {metrics['饱和度']}")
            
        # ================= 新增代码：计算并输出最终带损失时间的完整方案 =================
        print("\n最终周期各相位方案 (包含损失时间，总计=公共周期):")
        plan_str = []
        actual_total_time = 0
        phases_list = list(allocation.keys())
        
        for i, p in enumerate(phases_list):
            # 提取如 '北向全放' 这样的基础相位名，以匹配 cur_loss 字典
            match = re.search(r'\((.*?)\)', p)
            raw_name = match.group(1) if match else p
            loss_t = cur_loss.get(raw_name, 0)
            green_t = allocation[p]['有效绿灯(s)']
            
            # 使用四舍五入计算当前相位方案总时间（有效绿灯+损失时间）
            if i == len(phases_list) - 1:
                # 为了防止四舍五入引起的总和不等于最佳周期，最后一个相位用倒扣的方法计算
                phase_total = best_c - actual_total_time
            else:
                phase_total = round(green_t + loss_t)
                actual_total_time += phase_total
                
            plan_str.append(f"【{raw_name}】{phase_total}s")
            
        print(" -> ".join(plan_str) + f"  (总周期: {best_c}s)")
        # ========================================================================
        
    else:
        print("[异常] 数据不足，无法执行周期推演。")
    print("-" * 60)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_dir, "干线全流向_延误与流量占比明细表.csv") 
    try:
        df_details = load_and_preprocess_data(csv_file_path)
        run_analysis(df_details)
    except Exception as e:
        print(f"脚本执行异常: {e}")