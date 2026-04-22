import pandas as pd
import numpy as np
import os
import math
import re

# ================= 配置区 =================
# 1. 自定义分析日期接口
TARGET_DATE = '2026-03-25'  

# 2. 五路口信控方案接口
# （已补全另外3个路口的通用四相位模板，确保代码能计算全部路口的 Y 值。请按需替换为真实配时）
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
        '相位损失时间': {'东向全放': 8, '东西直行同放': 5, '西向全放': 5, '南向全放': 5, '北向全放': 5},
        '专属相位映射': {
            '相位1(东向全放)': [('东向', '左转')],
            '相位2(东西直行同放)': [('东向', '直行'), ('西向', '直行')],
            '相位3(西向全放)': [('西向', '左转')],
            '相位4(南向全放)': [('南向', '直行'), ('南向', '左转')],
            '相位5(北向全放)': [('北向', '直行'), ('北向', '左转')]
        }
    },
    '光明大道与光安路': {
        '现行周期': 163,
        '相位时间分配': {'北向全放': 35, '南向全放': 35, '东向全放': 35, '西向全放': 35},
        '相位损失时间': {'北向全放': 7, '南向全放': 7, '东向全放': 7, '西向全放': 7},
        '专属相位映射': {
            '相位1(北向全放)': [('北向', '直行'), ('北向', '左转')],
            '相位2(南向全放)': [('南向', '直行'), ('南向', '左转')],
            '相位3(东向全放)': [('东向', '直行'), ('东向', '左转')],
            '相位4(西向全放)': [('西向', '直行'), ('西向', '左转')]
        }
    },
    '光明大道与光明大街': {
        '现行周期': 167,
        '相位时间分配': {'北向全放': 46, '东向全放': 40, '西向全放': 30, '南向全放': 51},
        '相位损失时间': {'北向全放': 6, '东向全放': 6, '西向全放': 6, '南向全放': 6},
        '专属相位映射': {
            '相位1(北向全放)': [('北向', '直行'), ('北向', '左转')],
            '相位2(东向全放)': [('东向', '直行'), ('东向', '左转')],
            '相位3(西向全放)': [('西向', '直行'), ('西向', '左转')],
            '相位4(南向全放)': [('南向', '直行'), ('南向', '左转')]
        }
    },
    '光明大道与光辉大道': {
        '现行周期': 120,
        '相位时间分配': {'东向全放': 25, '北向全放': 35, '西向全放': 25, '南向全放': 35},
        '相位损失时间': {'东向全放': 6, '北向全放': 6, '西向全放': 6, '南向全放': 6},
        '专属相位映射': {
            '相位1(东向全放)': [('东向', '直行'), ('东向', '左转')],
            '相位2(北向全放)': [('北向', '直行'), ('北向', '左转')],
            '相位3(西向全放)': [('西向', '直行'), ('西向', '左转')],
            '相位4(南向全放)': [('南向', '直行'), ('南向', '左转')]
        }
    }
}

# 3. 指定平峰时段切片接口 (15分钟步长，共41个切片)
TIME_SLICES = {
    # ... (时段配置保持不变) ...
    10: ('02:15', '02:30'), 
    29: ('07:00', '07:15')
}

# [操作区]：输入你想研究的时段编号
TARGET_SLICE_ID = 29

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
    
    if Y_total <= 0: return None, None, {}

    min_total_delay = float('inf')
    best_C = None
    best_allocation = {}

    for C in range(60, 181):
        G = C - L_total
        if G <= 0: continue
            
        current_cycle_delay = 0
        current_allocation = {}
        valid_cycle = True
        
        for phase, data in phase_data.items():
            y_ratio = data['y_val'] / Y_total
            g_i = G * y_ratio
            lambda_i = g_i / C
            
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
            volume_weight = data['q_5min'] / total_q_5min if total_q_5min > 0 else 0
            current_cycle_delay += d_i * volume_weight
            
            current_allocation[phase] = {
                '有效绿灯(s)': round(g_i, 1),
                '绿信比': round(lambda_i, 3),
                '饱和度': round(x, 3),
                '延误(s/veh)': round(d_i, 2)
            }
            
        if not valid_cycle: continue
            
        if current_cycle_delay < min_total_delay:
            min_total_delay = current_cycle_delay
            best_C = C
            best_allocation = current_allocation

    return best_C, min_total_delay, best_allocation

# ================= 数据预处理 =================

def load_and_preprocess_data(csv_path):
    if not os.path.exists(csv_path): raise FileNotFoundError(f"未找到输入文件: {csv_path}")
    df = pd.read_csv(csv_path, encoding='GBK')
    df.columns = df.columns.str.strip()
    
    if 'create_time' in df.columns:
        df['_parsed_time'] = pd.to_datetime(df['create_time'], errors='coerce')
        df['_date_str'] = df['_parsed_time'].dt.strftime('%Y-%m-%d')
        df = df[df['_date_str'] == TARGET_DATE].copy()
        df['_temp_time'] = pd.to_datetime('2026-01-01 ' + df['_parsed_time'].dt.strftime('%H:%M:%S'))

    df['延误指数'] = pd.to_numeric(df['延误指数'], errors='coerce').fillna(0)
    
    # 支持 'pass_flow' 或 '分均流量' 列名
    p_col = next((c for c in df.columns if 'pass_flow' in c.lower() or '分均流量' in c), None)
    if p_col:
        df['分均流量'] = pd.to_numeric(df[p_col].astype(str).str.replace(',', '', regex=False).str.extract(r'(\d+\.?\d*)')[0], errors='coerce').fillna(0)
    else:
        df['分均流量'] = 0

    df['历史最大分均流量'] = pd.to_numeric(df.get('历史最大分均流量', 0), errors='coerce').fillna(0)
    
    # 读取排队长度 (容错处理)
    df['平均排队长度'] = pd.to_numeric(df.get('平均排队长度', 0), errors='coerce').fillna(0)

    df['进口道方向'] = df['进口道方向'].astype(str).str.strip()
    df['转向'] = df['转向'].astype(str).str.strip()
    return df[~df['进口道方向'].str.contains('聚合') & ~df['转向'].str.contains('聚合')]

# ================= 主流程 =================

def run_analysis(df):
    if TARGET_SLICE_ID not in TIME_SLICES: return
    start_time, end_time = TIME_SLICES[TARGET_SLICE_ID]
    period_name = f"切片 {TARGET_SLICE_ID} ({start_time} - {end_time})"
    
    print(f"\n[评估基准日]: {TARGET_DATE}")
    print(f"[研究时段]: {period_name}")
    print("-" * 60)

    active_intersections = list(CURRENT_SIGNAL_PLANS.keys())
    start_dt, end_dt = pd.to_datetime(f'2026-01-01 {start_time}'), pd.to_datetime(f'2026-01-01 {end_time}')
    
    df_period = df[(df['_temp_time'] >= start_dt) & (df['_temp_time'] <= end_dt) & (df['路口名称'].isin(active_intersections))]
    if df_period.empty: return
        
    inter_evals = []
    
    # 遍历计算每个路口的三大核心验证指标
    for inter, group_inter in df_period.groupby('路口名称'):
        flow_stats = group_inter.groupby(['进口道方向', '转向']).agg(
            平均延误指数=('延误指数', 'mean'),  
            平均分均流量=('分均流量', 'mean'),
            历史最大分均流量=('历史最大分均流量', 'first'),
            平均排队长度=('平均排队长度', 'mean') 
        ).reset_index()
        
        total_flow = flow_stats['平均分均流量'].sum()
        flow_stats['流量权重'] = flow_stats['平均分均流量'] / total_flow if total_flow > 0 else 0.0
        
        # 指标1：加权延误 Dt
        flow_stats['加权延误'] = flow_stats['平均延误指数'] * flow_stats['流量权重']
        # 指标2：加权排队
        flow_stats['加权排队'] = flow_stats['平均排队长度'] * flow_stats['流量权重']
        
        # 指标3：提取该路口的信控方案并计算全路口关键流量比 Y
        cur_plan = CURRENT_SIGNAL_PLANS.get(inter, {})
        cur_cycle = cur_plan.get('现行周期', 120)
        cur_alloc = cur_plan.get('相位时间分配', {})
        cur_loss = cur_plan.get('相位损失时间', {})
        cur_phase_mapping = cur_plan.get('专属相位映射', {})
        
        Y_total = 0.0
        phase_data_cache = {} # 缓存以便后续给关键路口算周期用
        
        for phase_name, directions in cur_phase_mapping.items():
            max_y_val = -1
            crit_q, crit_s_real = 0, 0
            
            for d, t in directions:
                mov = flow_stats[(flow_stats['进口道方向'] == d) & (flow_stats['转向'] == t)]
                if not mov.empty:
                    q = mov['平均分均流量'].iloc[0]
                    s = mov['历史最大分均流量'].iloc[0]
                    
                    if s > 0:
                        match = re.search(r'\((.*?)\)', phase_name)
                        old_lambda = 0.2  
                        if match:
                            raw_name = match.group(1)
                            old_g = cur_alloc.get(raw_name, 0) - cur_loss.get(raw_name, 0)
                            if old_g > 0 and cur_cycle > 0:
                                old_lambda = old_g / cur_cycle
                        
                        s_real_h = (s * 12) / old_lambda
                        y_val = (q * 12) / s_real_h
                        
                        if y_val > max_y_val:
                            max_y_val = y_val
                            crit_q, crit_s_real = q, s_real_h
                            
            if max_y_val > 0:
                Y_total += max_y_val
                phase_data_cache[phase_name] = {'q_5min': crit_q, 's_real_h': crit_s_real, 'y_val': max_y_val}
                
        inter_evals.append({
            '路口名称': inter,
            '加权总延误': flow_stats['加权延误'].sum(),
            '加权平均排队': flow_stats['加权排队'].sum(),
            '关键流量比Y': Y_total,
            '_phase_data': phase_data_cache,
            '_loss_time': sum(cur_loss.values()) if cur_loss else 16
        })
        
    if not inter_evals: return
        
    df_eval = pd.DataFrame(inter_evals).sort_values(by=['加权总延误'], ascending=[False])
    
    print("🚦 [核心指标交叉佐证榜单] (按加权延误降序排列):")
    for idx, row in df_eval.iterrows():
        print(f"  {idx+1}. {row['路口名称']}")
        print(f"     -> 加权总延误(Dt): {row['加权总延误']:.4f} | 关键流量比(Y): {row['关键流量比Y']:.4f} | 加权平均排队: {row['加权平均排队']:.1f} m")
    print("-" * 60)
    
    # 取 Top 1 作为关键路口推演
    critical_inter = df_eval.iloc[0]['路口名称']
    phase_data = df_eval.iloc[0]['_phase_data']
    L_total = df_eval.iloc[0]['_loss_time']
            
    if phase_data:
        best_c, min_delay, allocation = brute_force_optimal_cycle(L_total, phase_data)
        print(f"\n🎯 锁定全线关键瓶颈: {critical_inter}")
        print(f"[周期推演产物]: 最小延误-公共周期方案")
        print(f"理论最小加权预测延误: {min_delay:.2f} s/veh")
        print(f"最优公共周期 (Common Cycle): {best_c} s")
        # ... (后续打印逻辑不变) ...