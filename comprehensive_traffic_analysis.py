#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import os
import math
import re
import gc
import numpy as np

# ==========================================
# ⚙️ 核心配置区：时间段与图表开关
# ==========================================
# 开关：是否在最终的 3x4 综合看板(大总图)中也显示红色周期分割线？
# True -> 显示； False -> 保持纯净(默认)
SHOW_OVERLAY_ON_COMBINED_CHART = True  

# 自定义时间段及对应值配置
# start: 开始时间, end: 结束时间, value: 右侧Y轴的值（如：周期时间）
CUSTOM_PERIODS = [
    {"start": "00:00:00", "end": "06:00:00", "value": 84},
    {"start": "06:00:00", "end": "07:00:00", "value": 105},
    {"start": "07:00:00", "end": "09:00:00", "value": 120},
    {"start": "09:00:00", "end": "10:30:00", "value": 113},
    {"start": "10:30:00", "end": "12:00:00", "value": 110},
    {"start": "12:00:00", "end": "15:00:00", "value": 113},
    {"start": "15:00:00", "end": "17:30:00", "value": 115},
    {"start": "17:30:00", "end": "19:30:00", "value": 113},
    {"start": "19:30:00", "end": "21:30:00", "value": 106},
    {"start": "21:30:00", "end": "23:59:59", "value": 106},

]

def add_custom_periods_overlay(ax, periods_config, right_ylabel='周期时间 (秒)', base_date='2026-01-01'):
    """在指定的坐标轴上添加红色虚线分割和黑色阶梯线（双Y轴）"""
    if not periods_config:
        return
        
    times = []
    values = []
    
    for period in periods_config:
        start_time = pd.to_datetime(f"{base_date} {period['start']}")
        times.append(start_time)
        values.append(period['value'])
        # 绘制红色竖直虚线
        ax.axvline(x=start_time, color='red', linestyle='--', alpha=0.6, linewidth=1)
        
    # 添加最后一段的结束时间点，让阶梯图画到坐标轴边缘
    last_end_time = pd.to_datetime(f"{base_date} {periods_config[-1]['end']}")
    times.append(last_end_time)
    values.append(periods_config[-1]['value'])
    ax.axvline(x=last_end_time, color='red', linestyle='--', alpha=0.6, linewidth=1)
    
    # 创建双Y轴并绘制阶梯图
    ax2 = ax.twinx()
    ax2.step(times, values, where='post', color='black', linestyle='--', linewidth=2.5, label='周期变化')
    ax2.set_ylabel(right_ylabel)
    ax2.legend(loc='upper right')

# ==========================================
# 🎨 全局图表设置
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

CARDINAL_HANZI = {"E": "东向", "W": "西向", "S": "南向", "N": "北向"}
DIR_COLORS = {'E': '#1f77b4', 'S': '#2ca02c', 'W': '#d62728', 'N': '#9467bd'}
_WKT_COORD_PATTERN = re.compile(r'(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)')

# ==========================================
# ⚙️ 空间与流向分析模块
# ==========================================
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
    print("  -> [进度] 正在计算 60w+ 坐标偏转角")
    mask = df['lng_lat_seq'].notna() & (df['lng_lat_seq'] != '')
    df.loc[mask, '_raw_direction'] = df.loc[mask, 'lng_lat_seq'].apply(
        lambda x: extract_direction_from_coords(x, offset_degree)
    )
    
    if "frid" in df.columns:
        print("  -> [进度] 正在对齐轨迹流向")
        valid_dirs = df[['frid', '_raw_direction']].dropna()
        if not valid_dirs.empty:
            counts = valid_dirs.groupby(['frid', '_raw_direction']).size().reset_index(name='count')
            counts = counts.sort_values(by=['frid', 'count'], ascending=[True, False])
            best_dirs = counts.drop_duplicates(subset=['frid']).set_index('frid')['_raw_direction']
            df['main_direction'] = df['frid'].map(best_dirs)
            del valid_dirs, counts, best_dirs
        else: df['main_direction'] = df['_raw_direction']
    else: df['main_direction'] = df['_raw_direction']
    
    if '_raw_direction' in df.columns: df.drop(columns=['_raw_direction'], inplace=True)
    gc.collect()
    print("  -> [进度] 流向对齐完成，内存已释放！")
    return df

# ==========================================
# 📊 第一部分：转向流量专项分析
# ==========================================
def generate_turning_flow_charts(df, intersection_name, output_dir, interval):
    try:
        df_turn = df[df['turn_dir_no'].isin([1, 2])].dropna(subset=['main_direction'])
        if df_turn.empty: return
            
        grouped = df_turn.groupby([
            pd.Grouper(key='create_time', freq=f'{interval}min'), 
            'main_direction', 'turn_dir_no'
        ])['pass_flow'].sum().reset_index()
        
        grouped['date'] = grouped['create_time'].dt.date
        dates = sorted(grouped['date'].unique())
        directions = ['E', 'S', 'W', 'N']
        
        # --- 1. 2x2 子区图 ---
        fig_sub, axes_sub = plt.subplots(2, 2, figsize=(16, 12))
        fig_sub.suptitle(f'{intersection_name} - 各进口道转向流量 ({interval}分钟聚合)', fontsize=18, y=0.98)
        axes_sub = axes_sub.flatten()
        
        for i, direction in enumerate(directions):
            ax = axes_sub[i]
            dir_data = grouped[grouped['main_direction'] == direction]
            base_color = DIR_COLORS.get(direction, 'black') 
            
            for d in dates:
                date_data = dir_data[dir_data['date'] == d]
                left_data = date_data[date_data['turn_dir_no'] == 1]
                straight_data = date_data[date_data['turn_dir_no'] == 2]
                
                if not left_data.empty:
                    norm_time = pd.to_datetime('2026-01-01 ' + left_data['create_time'].dt.time.astype(str))
                    ax.plot(norm_time, left_data['pass_flow'], label=f'{d} 左转' if len(dates)>1 else '左转', 
                            color=base_color, linestyle='--', marker='o', markersize=3, alpha=0.8)
                if not straight_data.empty:
                    norm_time = pd.to_datetime('2026-01-01 ' + straight_data['create_time'].dt.time.astype(str))
                    ax.plot(norm_time, straight_data['pass_flow'], label=f'{d} 直行' if len(dates)>1 else '直行', 
                            color=base_color, linestyle='-', marker='s', markersize=3, alpha=0.8)
            
            ax.set_title(f'{CARDINAL_HANZI.get(direction, direction)}进口道', fontsize=14, color=base_color, fontweight='bold')
            ax.set_xlabel('时间', fontsize=12)
            ax.set_ylabel('车流量 (辆)', fontsize=12)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, linestyle='--', alpha=0.7)
            if not dir_data.empty: ax.legend()
            
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(output_dir, f'1_转向流量_分图_{interval}min.png'), dpi=150)
        plt.close(fig_sub)

        # --- 2. 全景同框图 ---
        fig_all, ax_all = plt.subplots(figsize=(14, 8))
        fig_all.suptitle(f'{intersection_name} - 全进口道转向流量对比 ({interval}分钟聚合)', fontsize=18)
        
        for direction in directions:
            dir_data = grouped[grouped['main_direction'] == direction]
            if dir_data.empty: continue
            base_color = DIR_COLORS.get(direction, 'black')
            dir_cn = CARDINAL_HANZI.get(direction, direction)
            
            for d in dates:
                date_data = dir_data[dir_data['date'] == d]
                left_data = date_data[date_data['turn_dir_no'] == 1]
                straight_data = date_data[date_data['turn_dir_no'] == 2]
                date_prefix = f"{d} " if len(dates) > 1 else ""
                
                if not left_data.empty:
                    norm_time = pd.to_datetime('2026-01-01 ' + left_data['create_time'].dt.time.astype(str))
                    ax_all.plot(norm_time, left_data['pass_flow'], label=f'{date_prefix}{dir_cn}左转', 
                                color=base_color, linestyle='--', marker='o', markersize=3, alpha=0.8)
                if not straight_data.empty:
                    norm_time = pd.to_datetime('2026-01-01 ' + straight_data['create_time'].dt.time.astype(str))
                    ax_all.plot(norm_time, straight_data['pass_flow'], label=f'{date_prefix}{dir_cn}直行', 
                                color=base_color, linestyle='-', marker='s', markersize=3, alpha=0.8)

        ax_all.set_xlabel('时间', fontsize=12)
        ax_all.set_ylabel('车流量 (辆)', fontsize=12)
        ax_all.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax_all.tick_params(axis='x', rotation=45)
        ax_all.grid(True, linestyle='--', alpha=0.7)
        ax_all.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
        
        plt.tight_layout(rect=[0, 0, 0.85, 0.96]) 
        plt.savefig(os.path.join(output_dir, f'2_转向流量_全同框_{interval}min.png'), dpi=150)
        plt.close(fig_all)
        
    except Exception as e: print(f"转向单图时序图出错: {e}")

# ==========================================
# 📊 第二部分：整体指标分析 (包含 3x4 终极总图)
# ==========================================
def generate_charts(df_interval, df_raw, intersection_name, output_dir, interval):
    if df_interval.empty: return
    df_interval['date'] = df_interval['create_time'].dt.date
    dates = sorted(df_interval['date'].dropna().unique())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    metrics = [
        ('queue_len_max', '最大排队长度', '排队长度(米)'),
        ('queue_len_avg', '平均排队长度', '排队长度(米)'),
        ('pass_flow', '总车流量', '车流量(辆)'),
        ('stop_time', '车均灯前停车时间', '停车时间(秒)'),
        ('stop_times', '车均停车次数', '停车次数(次)'),
        ('pass_speed', '车均通过速度', '速度(m/s)'),
        ('no_stop_pass_speed', '不停车通过速度', '速度(m/s)'),
        ('delay_index', '延误指数', '延误指数'),
        ('idx_state', '路口状态', '状态标识'),
        ('los', '延误指数评级', '评级(A-F)')
    ]

    # --- 准备 3x4 看板画布 ---
    fig_comb, axes_comb = plt.subplots(3, 4, figsize=(24, 16))
    fig_comb.suptitle(f'{intersection_name} - 交通运行综合指标 ({interval}分钟聚合)', fontsize=24, fontweight='bold', y=0.98)
    axes_comb = axes_comb.flatten()
    
    plot_idx = 0 
    los_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
    
    for col, title, ylabel in metrics:
        if col not in df_interval.columns or df_interval[col].dropna().empty: 
            plot_idx += 1
            continue
            
        fig_single, ax_single = plt.subplots(figsize=(10, 5))
        ax_comb = axes_comb[plot_idx] 
        plot_idx += 1
        
        for i, date in enumerate(dates):
            date_df = df_interval[df_interval['date'] == date]
            if date_df.empty: continue
            norm_time = pd.to_datetime('2026-01-01 ' + date_df['create_time'].dt.time.astype(str))
            
            y_data = date_df[col]
            if col == 'los':
                clean_los = date_df['los'].astype(str).str.strip().str.upper()
                y_data = clean_los.map(los_mapping).fillna(0)

            ax_single.plot(norm_time, y_data, marker='o', markersize=2, color=colors[i%len(colors)], label=str(date))
            ax_comb.plot(norm_time, y_data, marker='o', markersize=2, color=colors[i%len(colors)])
        
        # === 针对单张大图的设置 ===
        ax_single.set_title(f'{intersection_name} - {title} ({interval}分钟)')
        ax_single.set_ylabel(ylabel)
        ax_single.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax_single.tick_params(axis='x', rotation=45)
        ax_single.grid(True, linestyle='--', alpha=0.6)
        if col == 'los': ax_single.set_yticks([0, 1, 2, 3, 4, 5], ['A', 'B', 'C', 'D', 'E', 'F'])
        
        # 1. 强制给单图加上周期配置虚线
        add_custom_periods_overlay(ax_single, CUSTOM_PERIODS, right_ylabel='周期时间 (秒)')
        ax_single.legend(loc='upper left', fontsize=10) 
        
        fig_single.tight_layout()
        fig_single.savefig(os.path.join(output_dir, f'3_{col}_{interval}min.png'), dpi=120)
        plt.close(fig_single)
        
        # === 针对 3x4 总图中的小格子的设置 ===
        ax_comb.set_title(title, fontsize=14, fontweight='bold')
        ax_comb.set_ylabel(ylabel, fontsize=10)
        ax_comb.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax_comb.tick_params(axis='x', rotation=45)
        ax_comb.grid(True, linestyle='--', alpha=0.6)
        if col == 'los': ax_comb.set_yticks([0, 1, 2, 3, 4, 5], ['A', 'B', 'C', 'D', 'E', 'F'])
        
        # 2. 根据开关决定是否给总图加上周期配置虚线
        if SHOW_OVERLAY_ON_COMBINED_CHART:
            add_custom_periods_overlay(ax_comb, CUSTOM_PERIODS, right_ylabel='周期时间')

    # --- 第 11 个格子：全同框转向流量 ---
    ax_turn = axes_comb[10]
    turn_handles, turn_labels = [], [] 
    
    if 'turn_dir_no' in df_raw.columns:
        df_turn = df_raw[df_raw['turn_dir_no'].isin([1, 2])].dropna(subset=['main_direction'])
        if not df_turn.empty:
            grouped = df_turn.groupby([
                pd.Grouper(key='create_time', freq=f'{interval}min'), 
                'main_direction', 'turn_dir_no'
            ])['pass_flow'].sum().reset_index()
            grouped['date'] = grouped['create_time'].dt.date
            dates_turn = sorted(grouped['date'].unique())
            
            for direction in ['E', 'S', 'W', 'N']:
                dir_data = grouped[grouped['main_direction'] == direction]
                if dir_data.empty: continue
                base_color = DIR_COLORS.get(direction, 'black')
                dir_cn = CARDINAL_HANZI.get(direction, direction)
                
                for d in dates_turn:
                    date_data = dir_data[dir_data['date'] == d]
                    left_data = date_data[date_data['turn_dir_no'] == 1]
                    straight_data = date_data[date_data['turn_dir_no'] == 2]
                    date_prefix = f"{d} " if len(dates_turn) > 1 else ""
                    
                    if not left_data.empty:
                        norm_time = pd.to_datetime('2026-01-01 ' + left_data['create_time'].dt.time.astype(str))
                        line, = ax_turn.plot(norm_time, left_data['pass_flow'], label=f'{date_prefix}{dir_cn}左转', 
                                     color=base_color, linestyle='--', marker='o', markersize=2, alpha=0.8)
                        turn_handles.append(line)
                        turn_labels.append(f'{date_prefix}{dir_cn}左转')
                    if not straight_data.empty:
                        norm_time = pd.to_datetime('2026-01-01 ' + straight_data['create_time'].dt.time.astype(str))
                        line, = ax_turn.plot(norm_time, straight_data['pass_flow'], label=f'{date_prefix}{dir_cn}直行', 
                                     color=base_color, linestyle='-', marker='s', markersize=2, alpha=0.8)
                        turn_handles.append(line)
                        turn_labels.append(f'{date_prefix}{dir_cn}直行')
                        
            ax_turn.set_title('各进口道转向流量', fontsize=14, fontweight='bold')
            ax_turn.set_ylabel('车流量(辆)', fontsize=10)
            ax_turn.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax_turn.tick_params(axis='x', rotation=45)
            ax_turn.grid(True, linestyle='--', alpha=0.6)
        else: ax_turn.axis('off')
    else: ax_turn.axis('off')

    # --- 第 12 个格子：图例控制台 ---
    ax_leg = axes_comb[11]
    ax_leg.axis('off') 
    
    date_handles = [mlines.Line2D([], [], color=colors[i%len(colors)], marker='o', markersize=6, lw=2, label=str(d)) for i, d in enumerate(dates)]
    
    if turn_handles:
        by_label = dict(zip(turn_labels, turn_handles))
        leg_turn = ax_leg.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, 0.95), 
                                 fontsize=10, ncol=2, title="全进口道转向流量 (实线直行 / 虚线左转)", title_fontsize=12)
        leg_turn.get_title().set_fontweight('bold')
        leg_date = ax_leg.legend(handles=date_handles, loc='lower center', bbox_to_anchor=(0.5, 0.05), 
                                 fontsize=11, ncol=1, title="常规指标日期对比", title_fontsize=12)
        leg_date.get_title().set_fontweight('bold')
        ax_leg.add_artist(leg_turn) 
    else:
        leg_date = ax_leg.legend(handles=date_handles, loc='center', fontsize=12, ncol=1, title="常规指标日期对比", title_fontsize=14)
        leg_date.get_title().set_fontweight('bold')

    fig_comb.tight_layout(rect=[0, 0, 1, 0.96])
    fig_comb.savefig(os.path.join(output_dir, f'0_综合看板_3x4_{interval}min.png'), dpi=200)
    plt.close(fig_comb)

# ==========================================
# ⚙️ 底层文件读取与时间配置
# ==========================================
def get_date_ranges():
    time_periods = [
        (True,  '2026-03-02 00:00:00', '2026-03-02 23:59:59'), 
        (False, '2026-03-09 00:00:00', '2026-03-09 23:59:59'), 
        (False, '2026-03-11 00:00:00', '2026-03-11 23:59:59'),
        (False, '2026-03-16 00:00:00', '2026-03-16 23:59:59'),
        (False, '2026-03-23 00:00:00', '2026-03-23 23:59:59')
    ]
    date_ranges = []
    for i, (enable, start_date, end_date) in enumerate(time_periods, 1):
        if enable:
            date_ranges.append((pd.Timestamp(start_date), pd.Timestamp(end_date)))
            print(f"[INFO] 已启用: {start_date[:10]}")
    return date_ranges if date_ranges else [None]

def load_csv_safe(csv_path, target_cols):
    """带强力容错和内存保护的底层读取器"""
    print(f"  -> 尝试加载: {os.path.basename(csv_path)}")
    try:
        with open(csv_path, 'r', encoding='GBK', errors='replace') as f:
            df_head = pd.read_csv(f, nrows=0)
        all_cols = df_head.columns.tolist()
    except Exception as e:
        print(f"  [错误] 表头解析失败: {e}")
        return pd.DataFrame()

    use_cols = [c for c in target_cols if c in all_cols]
    if not use_cols: return pd.DataFrame()
    
    try: df = pd.read_csv(csv_path, encoding='GBK', usecols=use_cols)
    except Exception:
        try: df = pd.read_csv(csv_path, encoding='utf-8', usecols=use_cols)
        except Exception:
            with open(csv_path, 'r', encoding='GBK', errors='replace') as f:
                df = pd.read_csv(f, usecols=use_cols)
                
    if 'create_time' in df.columns:
        df['create_time'] = pd.to_datetime(df['create_time'])
    return df

# ==========================================
# 🚀 主控处理管线：支持双表融合查询
# ==========================================
def analyze_traffic_data(dir_csv_path, info_csv_path, date_ranges, offset_degree=0.0):
    print("\n[INFO] 正在启动数据管线 (启用内存保护与双表合并)...")
    
    # 1. 提取主数据
    target_dir_cols = [
        'create_time', 'inter_name', 'lng_lat_seq', 'turn_dir_no', 'pass_flow', 
        'queue_len_max', 'queue_len_avg', 'stop_time', 'stop_times', 'pass_speed', 
        'no_stop_pass_speed', 'delay_index', 'frid'
    ]
    df = load_csv_safe(dir_csv_path, target_dir_cols)
    if df.empty:
        print("[错误] 核心方向数据读取失败！")
        return
        
    intersection_name_val = ""
    if 'inter_name' in df.columns and not df['inter_name'].dropna().empty:
        intersection_name_val = str(df['inter_name'].dropna().iloc[0])

    # 2. 提取并横向拼接路口信息补充数据
    if info_csv_path and os.path.exists(info_csv_path):
        target_info_cols = ['create_time', 'idx_state', 'los', 'inter_name']
        df_info = load_csv_safe(info_csv_path, target_info_cols)
        if not df_info.empty:
            if 'inter_name' in df_info.columns and not df_info['inter_name'].dropna().empty:
                intersection_name_val = str(df_info['inter_name'].dropna().iloc[0])
            
            df_info = df_info.drop_duplicates(subset=['create_time'])
            
            if 'inter_name' in df_info.columns:
                df_info = df_info.drop(columns=['inter_name'])
                
            df = pd.merge(df, df_info, on='create_time', how='left')
            print("  -> [成功] 已融合路口状态(idx_state)与延误评级(los)指标。")
            del df_info
            gc.collect()

    print(f"\n[INFO] 成功加载并整合 {len(df)} 条有效数据记录！")
    df = df.sort_values('create_time')
    
    if intersection_name_val:
        intersection_name = f"{intersection_name_val}  "
    else:
        intersection_name = ""

    # 3. 空间打标
    if 'lng_lat_seq' in df.columns: df = enrich_direction_features(df, offset_degree)
    else: df['main_direction'] = 'Unknown'

    # 4. 日期过滤
    if date_ranges and date_ranges[0] is not None:
        mask = False
        for start_time, end_time in date_ranges:
            mask |= (df['create_time'] >= start_time) & (df['create_time'] <= end_time)
        df = df[mask]
    
    if df.empty:
        print("[错误] 过滤后数据为空，请检查时间段配置。")
        return
    
    # 5. 多维度聚合绘图
    if 'no_stop_pass_speed' in df.columns:
        df['no_stop_pass_speed'] = df['no_stop_pass_speed'].replace(0, np.nan)
        
    intervals = [1, 5, 15, 30, 60]
    for interval in intervals:
        print(f"\n========== 开始生成 {interval}分钟 聚合图表 ==========")
        output_dir = os.path.join(os.path.dirname(dir_csv_path), f'分析结果_{interval}min')
        os.makedirs(output_dir, exist_ok=True)
        
        if interval == 1:
            numeric_cols_mean = [c for c in ['queue_len_max', 'queue_len_avg', 'stop_time', 'stop_times', 'pass_speed', 'no_stop_pass_speed', 'delay_index'] if c in df.columns]
            df_interval_mean = df[numeric_cols_mean + ['create_time']].groupby(pd.Grouper(key='create_time', freq='1min')).mean().reset_index()
            
            if 'pass_flow' in df.columns:
                df_interval_sum = df[['pass_flow', 'create_time']].groupby(pd.Grouper(key='create_time', freq='1min')).sum().reset_index()
                df_interval = pd.merge(df_interval_mean, df_interval_sum, on='create_time', how='left')
            else:
                df_interval = df_interval_mean
                
        else:
            numeric_cols_mean = [c for c in ['queue_len_max', 'queue_len_avg', 'stop_time', 'stop_times', 'pass_speed', 'no_stop_pass_speed', 'delay_index'] if c in df.columns]
            df_interval_mean = df[numeric_cols_mean + ['create_time']].groupby(pd.Grouper(key='create_time', freq=f'{interval}min')).mean().reset_index()
            
            if 'pass_flow' in df.columns:
                df_interval_sum = df[['pass_flow', 'create_time']].groupby(pd.Grouper(key='create_time', freq=f'{interval}min')).sum().reset_index()
                df_interval = pd.merge(df_interval_mean, df_interval_sum, on='create_time', how='left')
            else:
                df_interval = df_interval_mean
            
            if 'idx_state' in df.columns:
                state_vals = df.groupby(pd.Grouper(key='create_time', freq=f'{interval}min'))['idx_state'].first().reset_index()
                df_interval = df_interval.merge(state_vals, on='create_time', how='left')
                
            if 'los' in df.columns:
                los_vals = df.groupby(pd.Grouper(key='create_time', freq=f'{interval}min'))['los'].apply(
                    lambda x: x.mode().iloc[0] if not x.mode().empty else None
                ).reset_index()
                df_interval = df_interval.merge(los_vals, on='create_time', how='left')
                
        generate_charts(df_interval, df, intersection_name, output_dir, interval)
        if 'turn_dir_no' in df.columns:
            generate_turning_flow_charts(df, intersection_name, output_dir, interval)
            
    print("\n🎉 全部图表生成完毕！请去脚本所在目录查看 [分析结果] 文件夹。")

if __name__ == "__main__":
    OFFSET_DEGREE = 0.0  
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = [f for f in os.listdir(script_dir) if f.endswith('.csv')]
    
    dir_csv, info_csv = None, None
    for f in csv_files:
        if "direction_index_view" in f: dir_csv = os.path.join(script_dir, f)
        elif "intersection_info_view" in f: info_csv = os.path.join(script_dir, f)
        
    if not dir_csv:
        fallback = next((f for f in csv_files if "ods_gaode" in f), csv_files[0] if csv_files else None)
        if fallback: dir_csv = os.path.join(script_dir, fallback)
        
    if dir_csv:
        analyze_traffic_data(dir_csv, info_csv, get_date_ranges(), OFFSET_DEGREE)
    else:
        print(f"错误：目录 {script_dir} 中没有找到CSV文件！")