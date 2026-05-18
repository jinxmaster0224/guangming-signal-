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

# 3. 全天24小时切片接口 (15分钟步长，共96个切片)
TIME_SLICES = {}
for i in range(96):
    start_hour = (i * 15) // 60
    start_min = (i * 15) % 60
    end_hour = ((i + 1) * 15) // 60
    end_min = ((i + 1) * 15) % 60
    start_str = f"{start_hour:02d}:{start_min:02d}"
    end_str = "23:59" if end_hour == 24 else f"{end_hour:02d}:{end_min:02d}"
    TIME_SLICES[i + 1] = (start_str, end_str)

# ================= 核心算法模块 =================

def calculate_hcm_delay(C, lambda_i, x, c, T=0.25, k=0.5, I=1.0):
    if lambda_i <= 0 or c <= 0: return 9999.0
    d1 = (0.5 * C * (1 - lambda_i)**2) / (1 - min(1.0, x) * lambda_i)
    inside_sqrt = max(0, (x - 1)**2 + (8 * k * I * x) / (c * T))
    d2 = 900 * T * ((x - 1) + math.sqrt(inside_sqrt))
    return d1 + d2

def brute_force_optimal_cycle(L_total, phase_data, T=0.25):
    Y_total = sum([data['y_val'] for data in phase_data.values()])
    total_q_5min = sum([data['q_5min'] for data in phase_data.values()])
    if Y_total <= 0: return None, None, {}
    min_total_delay = float('inf')
    best_C, best_allocation = None, {}

    for C in range(60, 181):
        G = C - L_total
        if G <= 0: continue
        current_cycle_delay, current_allocation, valid_cycle = 0, {}, True
        for phase, data in phase_data.items():
            y_ratio = data['y_val'] / Y_total
            g_i = G * y_ratio
            lambda_i = g_i / C
            if g_i < 12.0 or lambda_i <= 0.01:
                current_cycle_delay += 9999.0
                continue
            s_real_h, q_h = data['s_real_h'], data['q_5min'] * 12
            if s_real_h <= 0:
                valid_cycle = False
                break
            c = s_real_h * lambda_i
            x = q_h / c if c > 0 else 9999.0
            d_i = calculate_hcm_delay(C, lambda_i, x, c, T)
            current_cycle_delay += d_i * (data['q_5min'] / total_q_5min if total_q_5min > 0 else 0)
            current_allocation[phase] = {
                '有效绿灯(s)': round(g_i, 1), '绿信比': round(lambda_i, 3),
                '饱和度': round(x, 3), '延误(s/veh)': round(d_i, 2)
            }
        if valid_cycle and current_cycle_delay < min_total_delay:
            min_total_delay, best_C, best_allocation = current_cycle_delay, C, current_allocation
    return best_C, min_total_delay, best_allocation

# ================= 数据预处理 (严格字段映射) =================

def load_and_preprocess_data(csv_path):
    if not os.path.exists(csv_path): raise FileNotFoundError(f"未找到输入文件: {csv_path}")
    df = pd.read_csv(csv_path, encoding='GBK')
    df.columns = df.columns.str.strip()
    
    if 'create_time' in df.columns:
        df['_parsed_time'] = pd.to_datetime(df['create_time'], errors='coerce')
        df = df[df['_parsed_time'].dt.strftime('%Y-%m-%d') == TARGET_DATE].copy()
        df['_temp_time'] = pd.to_datetime('2026-01-01 ' + df['_parsed_time'].dt.strftime('%H:%M:%S'))

    # 严格读取新列名
    df['延误指数'] = pd.to_numeric(df['延误指数'], errors='coerce').fillna(0)
    df['平均分均流量'] = pd.to_numeric(df['5分钟车流量'], errors='coerce').fillna(0)
    df['历史最大分均流量'] = pd.to_numeric(df['历史最大5分钟流量'], errors='coerce').fillna(0)
    df['平均排队长度'] = pd.to_numeric(df['平均排队长度'], errors='coerce').fillna(0)
    df['最大排队长度'] = pd.to_numeric(df['最大排队长度'], errors='coerce').fillna(0)

    # 进口与转向清洗
    df['进口道方向'] = df['进口道方向'].astype(str).str.strip()
    df['转向'] = df['转向'].astype(str).str.strip()
    
    return df[~df['进口道方向'].str.contains('聚合') & ~df['转向'].str.contains('聚合')]

# ================= 主流程 =================

def run_analysis(df, slice_id):
    if slice_id not in TIME_SLICES: return
    start_time, end_time = TIME_SLICES[slice_id]
    start_dt, end_dt = pd.to_datetime(f'2026-01-01 {start_time}'), pd.to_datetime(f'2026-01-01 {end_time}')
    
    df_period = df[(df['_temp_time'] >= start_dt) & (df['_temp_time'] <= end_dt) & (df['路口名称'].isin(CURRENT_SIGNAL_PLANS.keys()))]
    if df_period.empty: return
        
    print(f"\n[切片 {slice_id:02d}] {start_time} - {end_time}")
    inter_evals = []
    
    for inter, group_inter in df_period.groupby('路口名称'):
        flow_stats = group_inter.groupby(['进口道方向', '转向']).agg(
            平均延误指数=('延误指数', 'mean'), 平均分均流量=('平均分均流量', 'mean'),
            历史最大分均流量=('历史最大分均流量', 'first'), 最大排队长度=('最大排队长度', 'max')
        ).reset_index()
        
        cur_plan = CURRENT_SIGNAL_PLANS[inter]
        Y_total, phase_data_cache = 0.0, {} 
        
        for phase_name, directions in cur_plan['专属相位映射'].items():
            max_y_val, crit_q, crit_s_real = -1, 0, 0
            for d, t in directions:
                mov = flow_stats[(flow_stats['进口道方向'] == d) & (flow_stats['转向'] == t)]
                if not mov.empty:
                    q, s = mov['平均分均流量'].iloc[0], mov['历史最大分均流量'].iloc[0]
                    if s > 0:
                        match = re.search(r'\((.*?)\)', phase_name)
                        old_lambda = 0.2
                        if match:
                            raw_name = match.group(1)
                            old_g = cur_plan['相位时间分配'].get(raw_name, 0) - cur_plan['相位损失时间'].get(raw_name, 0)
                            if old_g > 0: old_lambda = old_g / cur_plan['现行周期']
                        s_real_h = (s * 12) / old_lambda
                        y_val = (q * 12) / s_real_h
                        if y_val > max_y_val: max_y_val, crit_q, crit_s_real = y_val, q, s_real_h
            if max_y_val > 0:
                Y_total += max_y_val
                phase_data_cache[phase_name] = {'q_5min': crit_q, 's_real_h': crit_s_real, 'y_val': max_y_val}
        
        inter_evals.append({
            '路口名称': inter, '关键流量比Y': Y_total, 
            '交叉口最大排队': flow_stats['最大排队长度'].max(),
            '_phase_data': phase_data_cache,
            '_loss_time': sum(cur_plan['相位损失时间'].values())
        })
        
    for row in inter_evals:
        if row['_phase_data']:
            best_c, min_delay, allocation = brute_force_optimal_cycle(row['_loss_time'], row['_phase_data'])
            print(f"  路口: {row['路口名称']} | Y={row['关键流量比Y']:.3f} | 建议周期: {best_c}s | 预测延误: {min_delay:.1f}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 直接指向特定的 CSV 文件
    csv_file_path = os.path.join(script_dir, "干线全流向_延误与流量占比明细表.csv") 
    
    try:
        df_details = load_and_preprocess_data(csv_file_path)
        print(f"✅ 成功读取数据表，开始执行全天 96 切片周期推演...\n")
        
        for slice_id in TIME_SLICES.keys():
            run_analysis(df_details, slice_id)
            
    except Exception as e:
        print(f"脚本执行异常: {e}")