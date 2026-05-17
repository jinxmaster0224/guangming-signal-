import pandas as pd
import numpy as np
import os
import math
import re
from datetime import datetime, time

TARGET_DATE = '2026-03-25'
INTERSECTION = '光明大道与光明大街'

TRAFFIC_STATE_THRESHOLDS = {
    '极低流量': {'max_flow': 300, 'description': '夜间低峰，流量稀少'},
    '平峰流量': {'min_flow': 300, 'max_flow': 1250, 'description': '日常平峰，流量适中（含早高峰前、白天平峰、晚高峰后）'},
    '高峰流量': {'min_flow': 1250, 'description': '真正早晚高峰，流量拥挤'}
}

STATE_OPTIMIZATION_WEIGHTS = {
    '极低流量': {
        'delay_weight': 1.0        
    },
    '平峰流量': {
        'delay_weight': 0.7,      
        'throughput_weight': 0.3  
    },
    '高峰流量': {
        'delay_weight': 0.3,      
        'throughput_weight': 0.4,
        'queue_weight': 0.3
    }
}

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

CYCLE_BOUNDS = {
    '极低流量': {'min': 90, 'max': 110, 'default': 95},
    '平峰流量': {'min': 90, 'max': 150, 'default': 115},
    '高峰流量': {'min': 140, 'max': 180, 'default': 160}
}

TIME_PERIODS = [
    {'id': 1, 'start': '00:00', 'end': '06:30', 'avg_flow': 393, 'max_queue': 51},
    {'id': 2, 'start': '06:30', 'end': '07:30', 'avg_flow': 1977, 'max_queue': 171},
    {'id': 3, 'start': '07:30', 'end': '08:15', 'avg_flow': 4002, 'max_queue': 110},
    {'id': 4, 'start': '08:15', 'end': '09:15', 'avg_flow': 4323, 'max_queue': 274},
    {'id': 5, 'start': '09:15', 'end': '10:00', 'avg_flow': 3954, 'max_queue': 116},
    {'id': 6, 'start': '10:00', 'end': '11:45', 'avg_flow': 3459, 'max_queue': 138},
    {'id': 7, 'start': '11:45', 'end': '13:30', 'avg_flow': 3321, 'max_queue': 114},
    {'id': 8, 'start': '13:30', 'end': '14:15', 'avg_flow': 3906, 'max_queue': 167},
    {'id': 9, 'start': '14:15', 'end': '17:30', 'avg_flow': 4386, 'max_queue': 194},
    {'id': 10, 'start': '17:30', 'end': '20:00', 'avg_flow': 4068, 'max_queue': 234},
    {'id': 11, 'start': '20:00', 'end': '22:30', 'avg_flow': 2748, 'max_queue': 208},
    {'id': 12, 'start': '22:30', 'end': '23:59', 'avg_flow': 1656, 'max_queue': 146},
]

def get_current_cycle_for_time(time_str):
    for scheme in CURRENT_SIGNAL_PLAN['现行周期方案']:
        start = scheme['start']
        end = scheme['end']
        if time_str >= start and time_str < end:
            return scheme['cycle'], scheme['scheme'], scheme.get('phase_times', {})
    return 99, '方案10', {}

def get_current_cycle_for_period(start_str, end_str):
    cycles = []
    schemes = []
    phase_times_list = []
    time_strs = get_time_strs_in_period(start_str, end_str)
    for t in time_strs:
        cycle, scheme, phase_times = get_current_cycle_for_time(t)
        cycles.append(cycle)
        schemes.append(scheme)
        phase_times_list.append(phase_times)
    if cycles:
        most_common_cycle = max(set(cycles), key=cycles.count)
        most_common_scheme = max(set(schemes), key=schemes.count)
        most_common_phase_times = {}
        if phase_times_list:
            most_common_phase_times = max(set(tuple(p.items()) for p in phase_times_list), key=phase_times_list.count)
            most_common_phase_times = dict(most_common_phase_times)
        return most_common_cycle, most_common_scheme, most_common_phase_times
    return 99, '方案10', {}

def calculate_hcm_delay(C, lambda_i, x, c, T=0.25, k=0.5, I=1.0):
    if lambda_i <= 0 or c <= 0:
        return 9999.0
    d1 = (0.5 * C * (1 - lambda_i)**2) / (1 - min(1.0, x) * lambda_i)
    inside_sqrt = max(0, (x - 1)**2 + (8 * k * I * x) / (c * T))
    d2 = 900 * T * ((x - 1) + math.sqrt(inside_sqrt))
    return d1 + d2

def classify_traffic_state(total_flow_5min):
    if total_flow_5min <= TRAFFIC_STATE_THRESHOLDS['极低流量']['max_flow']:
        return '极低流量'
    elif TRAFFIC_STATE_THRESHOLDS['平峰流量']['min_flow'] < total_flow_5min <= TRAFFIC_STATE_THRESHOLDS['平峰流量']['max_flow']:
        return '平峰流量'
    else:
        return '高峰流量'

def classify_period_state(period_avg_flow_15min):
    flow_5min_equiv = period_avg_flow_15min / 3
    return classify_traffic_state(flow_5min_equiv)

def calculate_state_objective(C, phase_data, state, L_total, target_cycle=None):
    weights = STATE_OPTIMIZATION_WEIGHTS[state]
    bounds = CYCLE_BOUNDS[state]
    
    if C < bounds['min'] or C > bounds['max']:
        return float('inf')
    
    G = C - L_total
    if G <= 0:
        return float('inf')
    
    Y_total = sum([data['y_val'] for data in phase_data.values()])
    total_q_5min = sum([data['q_5min'] for data in phase_data.values()])
    
    if Y_total <= 0 or total_q_5min <= 0:
        return float('inf')
    
    total_delay = 0.0
    total_saturation = 0.0
    min_green_ratio = float('inf')
    max_saturation = 0.0
    
    for phase, data in phase_data.items():
        y_ratio = data['y_val'] / Y_total
        g_i = G * y_ratio
        lambda_i = g_i / C
        
        if g_i < 20.0 or lambda_i <= 0.01:
            return float('inf')
        
        s_real_h, q_h = data['s_real_h'], data['q_5min'] * 12
        c = s_real_h * lambda_i
        x = q_h / c if c > 0 else 9999.0
        
        if x > 0.92:
            return float('inf')
        
        d_i = calculate_hcm_delay(C, lambda_i, x, c)
        total_delay += d_i * (data['q_5min'] / total_q_5min)
        total_saturation += x * (data['q_5min'] / total_q_5min)
        max_saturation = max(max_saturation, x)
        min_green_ratio = min(min_green_ratio, lambda_i)
    
    delay_score = total_delay / 100
    cycle_diff = abs(C - bounds['default'])
    stability_score = cycle_diff / bounds['max']
    min_green_score = max(0, 0.12 - min_green_ratio) * 10
    throughput_score = abs(total_saturation - 0.88) * 10
    queue_score = max(0, max_saturation - 0.85) * 10
    
    objective = (
        weights.get('delay_weight', 0) * delay_score +
        weights.get('cycle_stability_weight', 0) * stability_score +
        weights.get('min_green_weight', 0) * min_green_score +
        weights.get('throughput_weight', 0) * throughput_score +
        weights.get('queue_weight', 0) * queue_score
    )
    
    return objective

def optimize_cycle_low_flow(phase_data, L_total, bounds):
    weights = STATE_OPTIMIZATION_WEIGHTS['极低流量']
    total_q_5min = sum([data['q_5min'] for data in phase_data.values()])
    Y_total = sum([data['y_val'] for data in phase_data.values()])
    
    min_objective = float('inf')
    best_C = bounds['default']
    best_allocation = {}
    
    for C in range(bounds['min'], bounds['max'] + 1):
        G = C - L_total
        if G <= 0:
            continue
            
        num_phases = len(phase_data)
        base_g = 20.0  
        total_base_g = num_phases * base_g
        
        if G < total_base_g:
            continue
            
        surplus_g = G - total_base_g
        
        current_allocation = {}
        total_delay = 0.0
        valid_cycle = True
        
        for phase, data in phase_data.items():
            if Y_total > 0:
                g_i = base_g + surplus_g * (data['y_val'] / Y_total)
            else:
                g_i = base_g + surplus_g / num_phases
            
            lambda_i = g_i / C
            
            if lambda_i <= 0.01:
                valid_cycle = False
                break
                
            s_real_h = data['s_real_h']
            q_h = data['q_5min'] * 12
            c = s_real_h * lambda_i
            x = q_h / c if c > 0 else 9999.0
            
            d_i = calculate_hcm_delay(C, lambda_i, x, c) if c > 0 else 9999.0
            
            volume_weight = data['q_5min'] / total_q_5min if total_q_5min > 0 else 0
            total_delay += d_i * volume_weight
            
            current_allocation[phase] = {
                '有效绿灯(s)': round(g_i, 1),
                '绿信比': round(lambda_i, 3),
                '饱和度': round(x, 3),
                '延误(s/veh)': round(d_i, 2)
            }
            
        if not valid_cycle:
            continue
            
        delay_score = total_delay / 100
        objective = weights.get('delay_weight', 1.0) * delay_score
        
        if objective < min_objective:
            min_objective = objective
            best_C = C
            best_allocation = current_allocation

    if not best_allocation:
        return optimize_cycle_off_peak(phase_data, L_total, bounds)
        
    return best_C, min_objective, best_allocation

def optimize_cycle_off_peak(phase_data, L_total, bounds):
    weights = STATE_OPTIMIZATION_WEIGHTS['平峰流量']
    total_q_5min = sum([data['q_5min'] for data in phase_data.values()])
    
    min_objective = float('inf')
    best_C = bounds['default']
    best_allocation = {}
    
    for C in range(bounds['min'], bounds['max'] + 1):
        G = C - L_total
        # 物理限制：至少保证所有相位都有20秒
        if G < len(phase_data) * 20.0:
            continue
        
        # =========================================================
        # 核心修改：带极值修正的迭代分配算法 (防止小流量绑架大周期)
        # =========================================================
        unallocated_phases = list(phase_data.keys())
        temp_allocation = {}
        remaining_G = G
        
        while unallocated_phases:
            current_Y_total = sum([phase_data[p]['y_val'] for p in unallocated_phases])
            
            if current_Y_total <= 0:
                # 如果没有流量，剩余时间均分
                for p in unallocated_phases:
                    temp_allocation[p] = remaining_G / len(unallocated_phases)
                break
                
            # 找到当前按比例分配中，分得最少的那个相位
            min_p = min(unallocated_phases, key=lambda p: phase_data[p]['y_val'])
            g_min_req = remaining_G * (phase_data[min_p]['y_val'] / current_Y_total)
            
            # 极值修正判断
            if g_min_req < 20.0:
                # 给最弱势群体强行保底20秒，并从总池子踢出
                temp_allocation[min_p] = 20.0
                remaining_G -= 20.0
                unallocated_phases.remove(min_p)
            else:
                # 如果连最弱势群体都能分到>=20秒，说明池子很充裕，大家放心按比例分
                for p in unallocated_phases:
                    temp_allocation[p] = remaining_G * (phase_data[p]['y_val'] / current_Y_total)
                break
        # =========================================================
        
        current_allocation = {}
        total_delay = 0.0
        total_saturation = 0.0
        valid_cycle = True
        
        for phase, data in phase_data.items():
            g_i = temp_allocation[phase]
            lambda_i = g_i / C
            
            if lambda_i <= 0.01:
                valid_cycle = False
                break
            
            s_real_h = data['s_real_h']
            q_h = data['q_5min'] * 12
            c = s_real_h * lambda_i
            x = q_h / c if c > 0 else 9999.0
            
            if x > 0.92:
                valid_cycle = False
                break
            
            d_i = calculate_hcm_delay(C, lambda_i, x, c)
            total_delay += d_i * (data['q_5min'] / total_q_5min) if total_q_5min > 0 else d_i
            total_saturation += x * (data['q_5min'] / total_q_5min) if total_q_5min > 0 else x
            
            current_allocation[phase] = {
                '有效绿灯(s)': round(g_i, 1),
                '绿信比': round(lambda_i, 3),
                '饱和度': round(x, 3),
                '延误(s/veh)': round(d_i, 2)
            }
        
        if not valid_cycle:
            continue
        
        delay_score = total_delay / 100
        throughput_score = abs(total_saturation - 0.88) * 10
        
        objective = (
            weights.get('delay_weight', 0) * delay_score +
            weights.get('throughput_weight', 0) * throughput_score
        )
        
        if objective < min_objective:
            min_objective = objective
            best_C = C
            best_allocation = current_allocation
    
    return best_C, min_objective, best_allocation

def optimize_cycle_peak(phase_data, L_total, bounds):
    weights = STATE_OPTIMIZATION_WEIGHTS['高峰流量']
    total_q_5min = sum([data['q_5min'] for data in phase_data.values()])
    Y_total = sum([data['y_val'] for data in phase_data.values()])
    
    min_objective = float('inf')
    best_C = bounds['default']
    best_allocation = {}
    
    target_cycle = bounds['min'] + int(Y_total * 50)
    target_cycle = max(bounds['min'], min(target_cycle, bounds['max']))
    
    for C in range(bounds['min'], bounds['max'] + 1):
        G = C - L_total
        if G <= 0:
            continue
        
        coord_phases = {p: d for p, d in phase_data.items() if '南' in p or '北' in p}
        non_coord_phases = {p: d for p, d in phase_data.items() if '南' not in p and '北' not in p}
        
        target_x_non_coord = 0.70
        non_coord_g_sum = 0
        temp_allocation = {}
        
        for p, d in non_coord_phases.items():
            g_req = (d['y_val'] * C) / target_x_non_coord
            g_actual = max(20.0, g_req)
            non_coord_g_sum += g_actual
            temp_allocation[p] = g_actual
        
        surplus_g = G - non_coord_g_sum
        
        if surplus_g <= 0 or not coord_phases:
            if Y_total <= 0:
                continue
            for p, d in phase_data.items():
                temp_allocation[p] = max(20.0, G * (d['y_val'] / Y_total))
        else:
            y_coord_total = sum([d['y_val'] for d in coord_phases.values()])
            for p, d in coord_phases.items():
                if y_coord_total > 0:
                    g_actual = max(20.0, surplus_g * (d['y_val'] / y_coord_total))
                else:
                    g_actual = max(20.0, surplus_g / len(coord_phases))
                temp_allocation[p] = g_actual
        
        current_allocation = {}
        total_delay = 0.0
        total_saturation = 0.0
        max_saturation = 0.0
        valid_cycle = True
        
        for phase, data in phase_data.items():
            g_i = temp_allocation[phase]
            lambda_i = g_i / C
            
            if lambda_i <= 0.01:
                valid_cycle = False
                break
            
            s_real_h = data['s_real_h']
            q_h = data['q_5min'] * 12
            
            if s_real_h <= 0:
                valid_cycle = False
                break
            
            c = s_real_h * lambda_i
            x = q_h / c if c > 0 else 9999.0
            
            if x > 0.92:
                valid_cycle = False
                break
            
            d_i = calculate_hcm_delay(C, lambda_i, x, c)
            
            volume_weight = data['q_5min'] / total_q_5min if total_q_5min > 0 else 0
            total_delay += d_i * volume_weight
            total_saturation += x * volume_weight
            max_saturation = max(max_saturation, x)
            
            current_allocation[phase] = {
                '有效绿灯(s)': round(g_i, 1),
                '绿信比': round(lambda_i, 3),
                '饱和度': round(x, 3),
                '延误(s/veh)': round(d_i, 2)
            }
        
        if not valid_cycle:
            continue
        
        delay_score = total_delay / 100
        throughput_score = abs(total_saturation - 0.88) * 10
        queue_score = max(0, max_saturation - 0.85) * 10
        cycle_deviation_score = abs(C - target_cycle) / (bounds['max'] - bounds['min']) * 10
        
        objective = (
            weights.get('delay_weight', 0) * delay_score +
            weights.get('throughput_weight', 0) * throughput_score +
            weights.get('queue_weight', 0) * queue_score +
            0.2 * cycle_deviation_score
        )
        
        if objective < min_objective:
            min_objective = objective
            best_C = C
            best_allocation = current_allocation
    
    return best_C, min_objective, best_allocation

def optimize_cycle_by_state(phase_data, state, L_total, target_cycle=None):
    bounds = CYCLE_BOUNDS[state]
    
    if state == '极低流量':
        return optimize_cycle_low_flow(phase_data, L_total, bounds)
    elif state == '平峰流量':
        return optimize_cycle_off_peak(phase_data, L_total, bounds)
    elif state == '高峰流量':
        return optimize_cycle_peak(phase_data, L_total, bounds)
    else:
        return optimize_cycle_off_peak(phase_data, L_total, bounds)

def load_and_preprocess_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到输入文件: {csv_path}")
    
    df = pd.read_csv(csv_path, encoding='GBK')
    df.columns = df.columns.str.strip()
    
    df['create_time'] = pd.to_datetime(df['create_time'], errors='coerce')
    df['true_start_time'] = df['create_time'] - pd.Timedelta(minutes=10)
    df = df[df['true_start_time'].dt.strftime('%Y-%m-%d') == TARGET_DATE].copy()

    df['进口道方向'] = df['进口道方向'].astype(str).str.strip()
    df['转向'] = df['转向'].astype(str).str.strip()
    df = df[~df['进口道方向'].str.contains('聚合') & ~df['转向'].str.contains('聚合')].copy()
    df = df[df['转向'].isin(['直行', '左转'])].copy()
    df['流向'] = df['进口道方向'] + df['转向']
    
    df['flow'] = pd.to_numeric(df['5分钟车流量'], errors='coerce').fillna(0)
    df['max_flow'] = pd.to_numeric(df['历史最大5分钟流量'], errors='coerce').fillna(0)
    df['delay'] = pd.to_numeric(df['延误指数'], errors='coerce').fillna(0)
    
    return df

def get_time_strs_in_period(start_str, end_str):
    start_h, start_m = map(int, start_str.split(':'))
    end_h, end_m = map(int, end_str.split(':'))
    
    result = []
    current_h, current_m = start_h, start_m
    
    while (current_h < end_h) or (current_h == end_h and current_m < end_m):
        result.append(f"{current_h:02d}:{current_m:02d}")
        current_m += 5
        if current_m >= 60:
            current_m = 0
            current_h += 1
    
    return result

def analyze_period(df, period):
    period_id = period['id']
    start = period['start']
    end = period['end']
    period_avg_flow = period['avg_flow']
    max_queue = period['max_queue']
    
    state = classify_period_state(period_avg_flow)
    bounds = CYCLE_BOUNDS[state]
    
    current_cycle, current_scheme, current_phase_times = get_current_cycle_for_period(start, end)
    
    time_strs = get_time_strs_in_period(start, end)
    
    period_df = df[df['true_start_time'].dt.strftime('%H:%M').isin(time_strs)].copy()
    
    L_total = sum(CURRENT_SIGNAL_PLAN['相位损失时间'].values())
    
    phase_data = {}
    for phase_name, directions in CURRENT_SIGNAL_PLAN['专属相位映射'].items():
        max_y, crit_q, crit_s = -1, 0, 0
        
        for d, t in directions:
            target_name = f"{d}{t}"
            q = period_df[period_df['流向'] == target_name]['flow'].mean()
            if pd.isna(q):
                q = df[df['流向'] == target_name]['flow'].mean()
            
            s_raw = df[df['流向'] == target_name]['max_flow'].max()
            if pd.isna(s_raw) or s_raw == 0:
                s_raw = 150
            
            match = re.search(r'\((.*?)\)', phase_name)
            old_lambda = 0.2
            if match:
                raw_name = match.group(1)
                phase_time = current_phase_times.get(raw_name, 0) if current_phase_times else 0
                lost_time = CURRENT_SIGNAL_PLAN['相位损失时间'].get(raw_name, 6)
                old_g = phase_time - lost_time
                if old_g > 0 and current_cycle > 0:
                    old_lambda = old_g / current_cycle
            
            s_real_h = (s_raw * 12) / old_lambda
            y_val = (q * 12) / s_real_h
            
            if y_val > max_y:
                max_y, crit_q, crit_s = y_val, q, s_real_h
        
        if max_y > 0:
            phase_data[phase_name] = {'q_5min': crit_q, 's_real_h': crit_s, 'y_val': max_y}
    
    best_C, obj_val, allocation = optimize_cycle_by_state(phase_data, state, L_total, current_cycle)
    
    return {
        'period_id': period_id,
        'start': start,
        'end': end,
        'state': state,
        'avg_flow': period_avg_flow,
        'max_queue': max_queue,
        'current_cycle': current_cycle,
        'current_scheme': current_scheme,
        'recommended_cycle': best_C,
        'allocation': allocation,
        'cycle_bounds': bounds
    }

def run_period_analysis(df):
    print(f"=" * 80)
    print(f"【{INTERSECTION}】各时段周期优化分析")
    print(f"分析日期: {TARGET_DATE}")
    print(f"=" * 80)
    
    print("\n现行周期方案:")
    for scheme in CURRENT_SIGNAL_PLAN['现行周期方案']:
        print(f"  {scheme['start']}-{scheme['end']}: {scheme['scheme']}, 周期{scheme['cycle']}s")
    
    print("\n流量状态定义 (等效15分钟单位):")
    for state, info in TRAFFIC_STATE_THRESHOLDS.items():
        desc = info.get('description', '')
        if 'max_flow' in info:
            print(f"  {state}: ≤{info['max_flow'] * 3} pcu/15min | {desc}")
        elif 'min_flow' in info and 'max_flow' in info:
            print(f"  {state}: {info['min_flow'] * 3}-{info['max_flow'] * 3} pcu/15min | {desc}")
        else:
            print(f"  {state}: >{info['min_flow'] * 3} pcu/15min | {desc}")
    
    print(f"\n时段划分 (Fisher肘部法则，共{len(TIME_PERIODS)}段):")
    for p in TIME_PERIODS:
        print(f"  时段{p['id']:02d}: {p['start']}-{p['end']} | 平均流量{p['avg_flow']} pcu/15min")
    
    print("\n" + "=" * 80)
    print("各时段周期优化结果")
    print("=" * 80)
    
    results = []
    for period in TIME_PERIODS:
        result = analyze_period(df, period)
        results.append(result)
        
        diff = result['recommended_cycle'] - result['current_cycle']
        diff_pct = diff / result['current_cycle'] * 100
        
        print(f"\n【时段 {result['period_id']:02d}】{result['start']} - {result['end']}")
        print(f"  流量状态: {result['state']}")
        print(f"  平均流量: {result['avg_flow']} pcu/15min | 最大排队: {result['max_queue']}m")
        print(f"  现行方案: {result['current_scheme']} | 周期: {result['current_cycle']}s")
        print(f"  推荐周期: {result['recommended_cycle']}s | 差异: {diff:+d}s ({diff_pct:+.1f}%)")
        print(f"  相位分配:")
        for phase, metrics in result['allocation'].items():
            print(f"    {phase}: 绿灯{metrics['有效绿灯(s)']}s | 绿信比{metrics['绿信比']} | 饱和度{metrics['饱和度']} | 延误{metrics['延误(s/veh)']}s")
    
    print("\n" + "=" * 80)
    print("汇总表")
    print("=" * 80)
    print(f"{'时段':^6} | {'时间范围':^14} | {'流量状态':^8} | {'平均流量':^10} | {'现行方案':^8} | {'现行周期':^10} | {'推荐周期':^10} | {'差异':^8}")
    print("-" * 90)
    
    for r in results:
        diff = r['recommended_cycle'] - r['current_cycle']
        diff_str = f"{diff:+d}s"
        print(f"{r['period_id']:02d}    | {r['start']}-{r['end']:^7} | {r['state']:^8} | {r['avg_flow']:>8} pcu | {r['current_scheme']:^8} | {r['current_cycle']:>8}s | {r['recommended_cycle']:>8}s | {diff_str:^8}")
    
    current_avg = sum([r['current_cycle'] for r in results]) / len(results)
    rec_avg = sum([r['recommended_cycle'] for r in results]) / len(results)
    print("-" * 90)
    print(f"现行周期均值: {current_avg:.1f}s | 推荐周期均值: {rec_avg:.1f}s | 差异: {rec_avg - current_avg:+.1f}s")
    
    return results

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "干线全流向_延误与流量占比明细表.csv")
    
    try:
        df = load_and_preprocess_data(csv_path)
        results = run_period_analysis(df)
        
        output_data = []
        for r in results:
            row = {
                '时段编号': r['period_id'],
                '开始时间': r['start'],
                '结束时间': r['end'],
                '流量状态': r['state'],
                '平均流量(pcu/15min)': r['avg_flow'],
                '最大排队(m)': r['max_queue'],
                '现行方案': r['current_scheme'],
                '现行周期(s)': r['current_cycle'],
                '推荐周期(s)': r['recommended_cycle'],
                '周期差异(s)': r['recommended_cycle'] - r['current_cycle']
            }
            for phase, metrics in r['allocation'].items():
                row[f'{phase}_绿灯(s)'] = metrics['有效绿灯(s)']
                row[f'{phase}_绿信比'] = metrics['绿信比']
                row[f'{phase}_饱和度'] = metrics['饱和度']
                row[f'{phase}_延误(s/veh)'] = metrics['延误(s/veh)']
            output_data.append(row)
        
        output_df = pd.DataFrame(output_data)
        output_path = os.path.join(script_dir, "各时段周期优化结果_主干道偏心.csv")
        output_df.to_csv(output_path, index=False, encoding='GBK')
        print(f"\n结果已保存至: {output_path}")
        
    except Exception as e:
        print(f"执行异常: {e}")
        import traceback
        traceback.print_exc()