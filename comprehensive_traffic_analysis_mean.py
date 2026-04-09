import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import math
import re
import gc
import numpy as np

# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  

# ================= 核心配置区：时间段与图表开关 =================
# 控制是否在 4x3 多指标对比总图 中显示周期配置虚线
SHOW_OVERLAY_ON_COMBINED_CHART = True  

# 自定义时间段及对应值配置
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

# 转向流量图表的颜色与文字配置
CARDINAL_HANZI = {"E": "东向", "W": "西向", "S": "南向", "N": "北向"}
DIR_COLORS = {'E': '#1f77b4', 'S': '#2ca02c', 'W': '#d62728', 'N': '#9467bd'}
_WKT_COORD_PATTERN = re.compile(r'(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)')

# ================= 辅助工具函数 =================
def format_date_str(dates):
    """根据日期数量智能格式化日期字符串"""
    if not dates: return "未知日期"
    if len(dates) == 1:
        return str(dates[0])
    elif len(dates) <= 3:
        return ", ".join([str(d) for d in dates])
    else:
        return f"{dates[0]} 至 {dates[-1]}"

def add_custom_periods_overlay(ax, periods_config, right_ylabel='周期时间 (秒)', base_date='2026-01-01'):
    """在指定的坐标轴上添加红色虚线分割和黑色阶梯线（双Y轴）"""
    if not periods_config: return
    times, values = [], []
    for period in periods_config:
        start_time = pd.to_datetime(f"{base_date} {period['start']}")
        times.append(start_time)
        values.append(period['value'])
        ax.axvline(x=start_time, color='red', linestyle='--', alpha=0.6, linewidth=1)
        
    last_end_time = pd.to_datetime(f"{base_date} {periods_config[-1]['end']}")
    times.append(last_end_time)
    values.append(periods_config[-1]['value']) 
    ax.axvline(x=last_end_time, color='red', linestyle='--', alpha=0.6, linewidth=1)
    
    ax2 = ax.twinx()
    ax2.step(times, values, where='post', color='black', linestyle='--', linewidth=2.5, label='周期变化')
    ax2.set_ylabel(right_ylabel)
    ax2.legend(loc='upper right', fontsize=8)  
# =================================================================

# ================= 空间与流向打标模块 (专供转向分图使用) =================
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
    print("  -> [副线进度] 正在为转向流量图计算坐标偏转角与轨迹流向...")
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
            del valid_dirs, counts, best_dirs
        else: df['main_direction'] = df['_raw_direction']
    else: df['main_direction'] = df['_raw_direction']
    if '_raw_direction' in df.columns: df.drop(columns=['_raw_direction'], inplace=True)
    gc.collect()
    return df

def generate_turning_flow_charts(df, intersection_name, output_dir, interval):
    """独立模块：生成各个方向的转向流量图 (分图与同框)"""
    try:
        df_turn = df[df['turn_dir_no'].isin([1, 2])].dropna(subset=['main_direction'])
        if df_turn.empty: return
            
        grouped = df_turn.groupby([
            pd.Grouper(key='create_time', freq=f'{interval}min'), 
            'main_direction', 'turn_dir_no'
        ])['pass_flow'].mean().reset_index()
        
        grouped['date'] = grouped['create_time'].dt.date
        dates = sorted(grouped['date'].unique())
        date_str = format_date_str(dates)
        directions = ['E', 'S', 'W', 'N']
        
        # --- 1. 2x2 子区图 ---
        fig_sub, axes_sub = plt.subplots(2, 2, figsize=(16, 12))
        fig_sub.suptitle(f'{intersection_name} [{date_str}] - 各进口道转向分均车流量 ({interval}分钟聚合)', fontsize=18, y=0.98)
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
            ax.set_ylabel('分均车流量 (辆)', fontsize=12)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, linestyle='--', alpha=0.7)
            if not dir_data.empty: ax.legend(loc='upper left', fontsize=9)
            add_custom_periods_overlay(ax, CUSTOM_PERIODS, right_ylabel='周期 (秒)')
            
        plt.tight_layout(rect=[0, 0, 1, 0.96], w_pad=3.0)
        plt.savefig(os.path.join(output_dir, f'1_转向流量_分图_{interval}min.png'), dpi=150, bbox_inches='tight')
        plt.close(fig_sub)

        # --- 2. 全景同框图 ---
        fig_all, ax_all = plt.subplots(figsize=(14, 8))
        fig_all.suptitle(f'{intersection_name} [{date_str}] - 全进口道转向分均流量对比 ({interval}分钟聚合)', fontsize=18)
        
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
        ax_all.set_ylabel('分均车流量 (辆)', fontsize=12)
        ax_all.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax_all.tick_params(axis='x', rotation=45)
        ax_all.grid(True, linestyle='--', alpha=0.7)
        
        ax_all.legend(loc='center left', bbox_to_anchor=(1.06, 0.5), fontsize=10)
        add_custom_periods_overlay(ax_all, CUSTOM_PERIODS, right_ylabel='周期时间 (秒)')
        
        plt.tight_layout(rect=[0, 0, 0.90, 0.96]) 
        plt.savefig(os.path.join(output_dir, f'2_转向流量_全同框_{interval}min.png'), dpi=150, bbox_inches='tight')
        plt.close(fig_all)
        print(f"  -> {interval}分钟单独的转向流量图表生成完成")
    except Exception as e: 
        print(f"转向单图生成出错: {e}")

# ================= 新增模块：其他指标的四方向分解图 =================
def generate_directional_metrics_charts(df_dir, intersection_name, output_dir, interval):
    """提取副表中的各个可用指标，像转向流量一样生成四方向子图和全同框总图"""
    if df_dir is None or df_dir.empty or 'main_direction' not in df_dir.columns:
        return
        
    try:
        metrics_def = [
            {'col': 'queue_len_max', 'name': '最大排队长度', 'ylabel': '最大排队长度 (米)'},
            {'col': 'queue_len_avg', 'name': '平均排队长度', 'ylabel': '平均排队长度 (米)'},
            {'col': 'stop_time', 'name': '停车时间', 'ylabel': '停车时间 (秒)'},
            {'col': 'stop_times', 'name': '停车次数', 'ylabel': '停车次数 (次)'},
            {'col': 'pass_speed', 'name': '通过速度', 'ylabel': '通过速度 (m/s)'},
            {'col': 'no_stop_pass_speed', 'name': '不停车通过速度', 'ylabel': '不停车速度 (m/s)'},
            {'col': 'delay_index', 'name': '延误指数', 'ylabel': '延误指数'}
        ]
        
        available_metrics = [m for m in metrics_def if m['col'] in df_dir.columns]
        if not available_metrics: return
            
        print(f"  -> [新增功能] 发现可用方向指标：{[m['name'] for m in available_metrics]}，准备生成各自的四方向分解图...")

        has_turn = 'turn_dir_no' in df_dir.columns
        if has_turn:
            df_target = df_dir[df_dir['turn_dir_no'].isin([1, 2])].dropna(subset=['main_direction'])
            group_cols = [pd.Grouper(key='create_time', freq=f'{interval}min'), 'main_direction', 'turn_dir_no']
        else:
            df_target = df_dir.dropna(subset=['main_direction'])
            group_cols = [pd.Grouper(key='create_time', freq=f'{interval}min'), 'main_direction']

        if df_target.empty: return
        directions = ['E', 'S', 'W', 'N']

        for metric in available_metrics:
            col = metric['col']
            name = metric['name']
            ylabel = metric['ylabel']

            grouped = df_target.groupby(group_cols)[col].mean().reset_index()
            grouped['date'] = grouped['create_time'].dt.date
            dates = sorted(grouped['date'].unique())
            date_str = format_date_str(dates)

            # --- 1. 每个指标生成 2x2 四方向分图 ---
            fig_sub, axes_sub = plt.subplots(2, 2, figsize=(16, 12))
            fig_sub.suptitle(f'{intersection_name} [{date_str}] - 各进口道{name}对比 ({interval}分钟聚合)', fontsize=18, y=0.98)
            axes_sub = axes_sub.flatten()

            for i, direction in enumerate(directions):
                ax = axes_sub[i]
                dir_data = grouped[grouped['main_direction'] == direction]
                base_color = DIR_COLORS.get(direction, 'black')

                for d in dates:
                    date_data = dir_data[dir_data['date'] == d]
                    
                    if has_turn:
                        left_data = date_data[date_data['turn_dir_no'] == 1]
                        straight_data = date_data[date_data['turn_dir_no'] == 2]

                        if not left_data.empty:
                            norm_time = pd.to_datetime('2026-01-01 ' + left_data['create_time'].dt.time.astype(str))
                            ax.plot(norm_time, left_data[col], label=f'{d} 左转' if len(dates)>1 else '左转',
                                    color=base_color, linestyle='--', marker='o', markersize=3, alpha=0.8)
                        if not straight_data.empty:
                            norm_time = pd.to_datetime('2026-01-01 ' + straight_data['create_time'].dt.time.astype(str))
                            ax.plot(norm_time, straight_data[col], label=f'{d} 直行' if len(dates)>1 else '直行',
                                    color=base_color, linestyle='-', marker='s', markersize=3, alpha=0.8)
                    else:
                        if not date_data.empty:
                            norm_time = pd.to_datetime('2026-01-01 ' + date_data['create_time'].dt.time.astype(str))
                            ax.plot(norm_time, date_data[col], label=str(d),
                                    color=base_color, linestyle='-', marker='s', markersize=3, alpha=0.8)

                ax.set_title(f'{CARDINAL_HANZI.get(direction, direction)}进口道', fontsize=14, color=base_color, fontweight='bold')
                ax.set_xlabel('时间', fontsize=12)
                ax.set_ylabel(ylabel, fontsize=12)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, linestyle='--', alpha=0.7)
                if not dir_data.empty: ax.legend(loc='upper left', fontsize=9)
                add_custom_periods_overlay(ax, CUSTOM_PERIODS, right_ylabel='周期 (秒)')

            plt.tight_layout(rect=[0, 0, 1, 0.96], w_pad=3.0)
            plt.savefig(os.path.join(output_dir, f'3_{name}_四方向分图_{interval}min.png'), dpi=150, bbox_inches='tight')
            plt.close(fig_sub)

            # --- 2. 每个指标生成全景同框总图 ---
            fig_all, ax_all = plt.subplots(figsize=(14, 8))
            fig_all.suptitle(f'{intersection_name} [{date_str}] - 全进口道{name}总对比 ({interval}分钟聚合)', fontsize=18)

            for direction in directions:
                dir_data = grouped[grouped['main_direction'] == direction]
                if dir_data.empty: continue
                base_color = DIR_COLORS.get(direction, 'black')
                dir_cn = CARDINAL_HANZI.get(direction, direction)

                for d in dates:
                    date_data = dir_data[dir_data['date'] == d]
                    date_prefix = f"{d} " if len(dates) > 1 else ""

                    if has_turn:
                        left_data = date_data[date_data['turn_dir_no'] == 1]
                        straight_data = date_data[date_data['turn_dir_no'] == 2]

                        if not left_data.empty:
                            norm_time = pd.to_datetime('2026-01-01 ' + left_data['create_time'].dt.time.astype(str))
                            ax_all.plot(norm_time, left_data[col], label=f'{date_prefix}{dir_cn}左转',
                                        color=base_color, linestyle='--', marker='o', markersize=3, alpha=0.8)
                        if not straight_data.empty:
                            norm_time = pd.to_datetime('2026-01-01 ' + straight_data['create_time'].dt.time.astype(str))
                            ax_all.plot(norm_time, straight_data[col], label=f'{date_prefix}{dir_cn}直行',
                                        color=base_color, linestyle='-', marker='s', markersize=3, alpha=0.8)
                    else:
                        if not date_data.empty:
                            norm_time = pd.to_datetime('2026-01-01 ' + date_data['create_time'].dt.time.astype(str))
                            ax_all.plot(norm_time, date_data[col], label=f'{date_prefix}{dir_cn}',
                                        color=base_color, linestyle='-', marker='s', markersize=3, alpha=0.8)

            ax_all.set_xlabel('时间', fontsize=12)
            ax_all.set_ylabel(ylabel, fontsize=12)
            ax_all.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax_all.tick_params(axis='x', rotation=45)
            ax_all.grid(True, linestyle='--', alpha=0.7)
            
            ax_all.legend(loc='center left', bbox_to_anchor=(1.06, 0.5), fontsize=10)
            add_custom_periods_overlay(ax_all, CUSTOM_PERIODS, right_ylabel='周期时间 (秒)')
            
            plt.tight_layout(rect=[0, 0, 0.90, 0.96])
            plt.savefig(os.path.join(output_dir, f'4_{name}_全同框_{interval}min.png'), dpi=150, bbox_inches='tight')
            plt.close(fig_all)
            
        print(f"  -> {interval}分钟: 所有可用指标的方向分解图生成完成")
    except Exception as e:
        print(f"生成指标方向分解图出错: {e}")
# =================================================================

def generate_charts(df, intersection_name, output_dir, interval, df_dir=None):
    """
    生成指定时间间隔的折线图 (主线指标)
    并在最后合并入转向全图
    """
    try:
        print(f"开始生成{interval}分钟聚合的主图表...")
        
        # 按日期分组
        df['date'] = df['create_time'].dt.date
        dates = sorted(df['date'].unique())
        date_str = format_date_str(dates)
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'gray']
        
        # ================= 1. 生成最大排队长度折线图 =================
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            if 'queue_len_max' in date_df.columns:
                ax.plot(normalized_time, date_df['queue_len_max'], marker='o', markersize=3, linewidth=1, 
                         color=colors[i % len(colors)], label=str(date))
        ax.set_title(f'{intersection_name} [{date_str}] - 最大排队长度变化趋势 ({interval}分钟)')
        ax.set_xlabel('时间')
        ax.set_ylabel('最大排队长度')
        ax.grid(True)
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        add_custom_periods_overlay(ax, CUSTOM_PERIODS, right_ylabel='周期时间 (秒)')
        ax.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'queue_len_max_trend_{interval}min.png'), bbox_inches='tight')
        plt.close()
        
        # ================= 2. 生成平均排队长度折线图 =================
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            if 'queue_len_avg' in date_df.columns:
                ax.plot(normalized_time, date_df['queue_len_avg'], marker='o', markersize=3, linewidth=1, 
                         color=colors[i % len(colors)], label=str(date))
        ax.set_title(f'{intersection_name} [{date_str}] - 平均排队长度变化趋势 ({interval}分钟)')
        ax.set_xlabel('时间')
        ax.set_ylabel('平均排队长度')
        ax.grid(True)
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        add_custom_periods_overlay(ax, CUSTOM_PERIODS, right_ylabel='周期时间 (秒)')
        ax.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'queue_len_avg_trend_{interval}min.png'), bbox_inches='tight')
        plt.close()
        
        # ================= 3. 生成车流量折线图 (变更为：分均车流量) =================
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            if 'pass_flow' in date_df.columns:
                ax.plot(normalized_time, date_df['pass_flow'], marker='o', markersize=3, linewidth=1, 
                         color=colors[i % len(colors)], label=str(date))
        ax.set_title(f'{intersection_name} [{date_str}] - 分均车流量变化趋势 ({interval}分钟)')
        ax.set_xlabel('时间')
        ax.set_ylabel('分均车流量 (辆)')
        ax.grid(True)
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        add_custom_periods_overlay(ax, CUSTOM_PERIODS, right_ylabel='周期时间 (秒)')
        ax.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'pass_flow_trend_{interval}min.png'), bbox_inches='tight')
        plt.close()
        
        # ================= 4. 生成车均灯前停车时间折线图 =================
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            if 'stop_time' in date_df.columns:
                ax.plot(normalized_time, date_df['stop_time'], marker='o', markersize=3, linewidth=1, 
                         color=colors[i % len(colors)], label=str(date))
        ax.set_title(f'{intersection_name} [{date_str}] - 车均灯前停车时间变化趋势 ({interval}分钟)')
        ax.set_xlabel('时间')
        ax.set_ylabel('停车时间 (秒)')
        ax.grid(True)
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        add_custom_periods_overlay(ax, CUSTOM_PERIODS, right_ylabel='周期时间 (秒)')
        ax.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'stop_time_trend_{interval}min.png'), bbox_inches='tight')
        plt.close()

        # ================= 5. 生成车均停车次数折线图 =================
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            if 'stop_times' in date_df.columns:
                ax.plot(normalized_time, date_df['stop_times'], marker='o', markersize=3, linewidth=1, 
                         color=colors[i % len(colors)], label=str(date))
        ax.set_title(f'{intersection_name} [{date_str}] - 车均停车次数变化趋势 ({interval}分钟)')
        ax.set_xlabel('时间')
        ax.set_ylabel('停车次数')
        ax.grid(True)
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        add_custom_periods_overlay(ax, CUSTOM_PERIODS, right_ylabel='周期时间 (秒)')
        ax.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'stop_times_trend_{interval}min.png'), bbox_inches='tight')
        plt.close()
        
        # ================= 6. 生成车均通过速度折线图 =================
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            if 'pass_speed' in date_df.columns:
                ax.plot(normalized_time, date_df['pass_speed'], marker='o', markersize=3, linewidth=1, 
                         color=colors[i % len(colors)], label=str(date))
        ax.set_title(f'{intersection_name} [{date_str}] - 车均通过速度变化趋势 ({interval}分钟)')
        ax.set_xlabel('时间')
        ax.set_ylabel('通过速度 (m/s)')
        ax.grid(True)
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        add_custom_periods_overlay(ax, CUSTOM_PERIODS, right_ylabel='周期时间 (秒)')
        ax.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'pass_speed_trend_{interval}min.png'), bbox_inches='tight')
        plt.close()
        
        # ================= 7. 生成车均不停车通过速度折线图 =================
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            if 'no_stop_pass_speed' in date_df.columns:
                ax.plot(normalized_time, date_df['no_stop_pass_speed'], marker='o', markersize=3, linewidth=1, 
                         color=colors[i % len(colors)], label=str(date))
        ax.set_title(f'{intersection_name} [{date_str}] - 车均不停车通过速度变化趋势 ({interval}分钟)')
        ax.set_xlabel('时间')
        ax.set_ylabel('不停车通过速度 (m/s)')
        ax.grid(True)
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        add_custom_periods_overlay(ax, CUSTOM_PERIODS, right_ylabel='周期时间 (秒)')
        ax.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'no_stop_pass_speed_trend_{interval}min.png'), bbox_inches='tight')
        plt.close()
        
        # ================= 8. 生成延误指数折线图 =================
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            if 'delay_index' in date_df.columns:
                ax.plot(normalized_time, date_df['delay_index'], marker='o', markersize=3, linewidth=1, 
                         color=colors[i % len(colors)], label=str(date))
        ax.set_title(f'{intersection_name} [{date_str}] - 延误指数变化趋势 ({interval}分钟)')
        ax.set_xlabel('时间')
        ax.set_ylabel('延误指数')
        ax.grid(True)
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        add_custom_periods_overlay(ax, CUSTOM_PERIODS, right_ylabel='周期时间 (秒)')
        ax.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'delay_index_trend_{interval}min.png'), bbox_inches='tight')
        plt.close()
        
        # ================= 9. 生成路口状态折线图 =================
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            if 'idx_state' in date_df.columns:
                ax.plot(normalized_time, date_df['idx_state'], marker='o', markersize=3, linewidth=1, 
                         color=colors[i % len(colors)], label=str(date))
        ax.set_title(f'{intersection_name} [{date_str}] - 路口状态变化趋势 ({interval}分钟)')
        ax.set_xlabel('时间')
        ax.set_ylabel('路口状态')
        ax.grid(True)
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        add_custom_periods_overlay(ax, CUSTOM_PERIODS, right_ylabel='周期时间 (秒)')
        ax.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'idx_state_trend_{interval}min.png'), bbox_inches='tight')
        plt.close()
        
        # ================= 10. 生成延误指数评级折线图 =================
        if 'los' in df.columns:
            plt.figure(figsize=(12, 6))
            ax = plt.gca()
            los_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'nan': 0}
            for i, date in enumerate(dates):
                date_df = df[df['date'] == date]
                los_values = date_df['los'].map(los_mapping).fillna(0)
                time_only = date_df['create_time'].dt.time
                normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
                ax.plot(normalized_time, los_values, marker='o', markersize=3, linewidth=1, 
                         color=colors[i % len(colors)], label=str(date))
            
            ax.set_title(f'{intersection_name} [{date_str}] - 延误指数评级变化趋势 ({interval}分钟)')
            ax.set_xlabel('时间')
            ax.set_ylabel('延误指数评级')
            ax.grid(True)
            plt.setp(ax.get_xticklabels(), rotation=45)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.set_yticks([0, 1, 2, 3, 4, 5])
            ax.set_yticklabels(['A', 'B', 'C', 'D', 'E', 'F'])
            add_custom_periods_overlay(ax, CUSTOM_PERIODS, right_ylabel='周期时间 (秒)')
            ax.legend(loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'los_trend_{interval}min.png'), bbox_inches='tight')
            plt.close()
        
        # ================= 生成多指标对比图（4x3布局） =================
        plt.figure(figsize=(16, 12))
        plt.suptitle(f'{intersection_name} [{date_str}] - 多指标综合对比图 ({interval}分钟聚合)', fontsize=20)
        
        metrics_plot = [
            ('queue_len_max', '最大排队长度（米）', '最大排队长度趋势图'),
            ('queue_len_avg', '平均排队长度（米）', '平均排队长度趋势图'),
            ('pass_flow', '分均车流量（辆）', '分均车流量趋势图'),
            ('stop_time', '停车时间 (秒)', '停车时间趋势图'),
            ('stop_times', '停车次数（次）', '停车次数趋势图'),
            ('pass_speed', '通过速度 (m/s)', '通过速度趋势图'),
            ('no_stop_pass_speed', '不停车通过速度 (m/s)', '不停车通过速度趋势图'),
            ('delay_index', '延误指数', '延误指数趋势图'),
            ('idx_state', '路口状态', '路口状态趋势图')
        ]
        
        for idx, (col, ylabel, title) in enumerate(metrics_plot, 1):
            plt.subplot(4, 3, idx)
            for i, date in enumerate(dates):
                date_df = df[df['date'] == date]
                time_only = date_df['create_time'].dt.time
                normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
                if col in date_df.columns:
                    plt.plot(normalized_time, date_df[col], marker='o', markersize=2, linewidth=1, 
                             color=colors[i % len(colors)], label=str(date))
            plt.title(title)
            plt.ylabel(ylabel)
            plt.xticks(rotation=45)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.legend(loc='upper left', fontsize=8)
            plt.grid(True, linestyle='--', alpha=0.6)
            if SHOW_OVERLAY_ON_COMBINED_CHART:
                add_custom_periods_overlay(plt.gca(), CUSTOM_PERIODS, right_ylabel='周期时间 (秒)')
        
        # 第10个格子：延误指数评级
        if 'los' in df.columns:
            plt.subplot(4, 3, 10)
            los_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'nan': 0}
            for i, date in enumerate(dates):
                date_df = df[df['date'] == date]
                los_values = date_df['los'].map(los_mapping).fillna(0)
                time_only = date_df['create_time'].dt.time
                normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
                plt.plot(normalized_time, los_values, marker='o', markersize=2, linewidth=1, 
                         color=colors[i % len(colors)], label=str(date))
            plt.title('延误指数评级趋势图')
            plt.ylabel('延误指数评级')
            plt.xticks(rotation=45)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.yticks([0, 1, 2, 3, 4, 5], ['A', 'B', 'C', 'D', 'E', 'F'])
            plt.legend(loc='upper left', fontsize=8)
            plt.grid(True, linestyle='--', alpha=0.6)
            if SHOW_OVERLAY_ON_COMBINED_CHART:
                add_custom_periods_overlay(plt.gca(), CUSTOM_PERIODS, right_ylabel='周期时间 (秒)')

        # === 强制显示 第11个格子：全同框转向流量 ===
        ax_turn = plt.subplot(4, 3, 11)
        turn_handles, turn_labels = [], []
        
        if df_dir is not None and not df_dir.empty and 'turn_dir_no' in df_dir.columns and 'main_direction' in df_dir.columns:
            grouped = df_dir.groupby([
                pd.Grouper(key='create_time', freq=f'{interval}min'), 
                'main_direction', 'turn_dir_no'
            ])['pass_flow'].mean().reset_index()
            
            grouped['date'] = grouped['create_time'].dt.date
            dates_turn = sorted(grouped['date'].unique())
            directions = ['E', 'S', 'W', 'N']
            
            for direction in directions:
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
                        
            ax_turn.set_title('全进口道转向分均流量')
            ax_turn.set_ylabel('分均车流量(辆)')
            ax_turn.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.setp(ax_turn.get_xticklabels(), rotation=45)
            ax_turn.grid(True, linestyle='--', alpha=0.6)
            
            if SHOW_OVERLAY_ON_COMBINED_CHART:
                add_custom_periods_overlay(ax_turn, CUSTOM_PERIODS, right_ylabel='周期时间(秒)')
        else:
            ax_turn.text(0.5, 0.5, '未找到转向流量数据\n(需保证目录下有 index_view 文件)', 
                         ha='center', va='center', fontsize=12, color='gray', weight='bold')
            ax_turn.set_title('全进口道转向分均流量 (无数据)')
            ax_turn.set_xticks([])
            ax_turn.set_yticks([])
                
        # === 强制显示 第12个格子：转向专用的图例控制台 ===
        ax_leg = plt.subplot(4, 3, 12)
        ax_leg.axis('off')
        if turn_handles:
            by_label = dict(zip(turn_labels, turn_handles))
            ax_leg.legend(by_label.values(), by_label.keys(), loc='center', fontsize=8, ncol=2, title="转向流量图例\n(实线直行 / 虚线左转)", title_fontsize=10)
        else:
            ax_leg.text(0.5, 0.5, '无图例数据', ha='center', va='center', fontsize=12, color='gray')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96], w_pad=1.5, h_pad=1.5)
        combined_path = os.path.join(output_dir, f'combined_analysis_{interval}min.png')
        plt.savefig(combined_path, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"生成主线图表时出错: {e}")
        import traceback
        traceback.print_exc()

def generate_vertical_comparison_charts(data_by_interval, intersection_name, output_dir):
    """纵向对比图生成逻辑"""
    try:
        comparison_dir = os.path.join(output_dir, '纵向对比图')
        os.makedirs(comparison_dir, exist_ok=True)
        date_colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'gray']
        metrics = [
            {'column': 'queue_len_max', 'name': '最大排队长度', 'ylabel': '最大排队长度'}, 
            {'column': 'queue_len_avg', 'name': '平均排队长度', 'ylabel': '平均排队长度'},
            {'column': 'pass_flow', 'name': '分均车流量', 'ylabel': '分均车流量 (辆)'},
            {'column': 'stop_time', 'name': '车均灯前停车时间', 'ylabel': '停车时间 (秒)'},
            {'column': 'stop_times', 'name': '车均停车次数', 'ylabel': '停车次数'},
            {'column': 'pass_speed', 'name': '车均通过速度', 'ylabel': '通过速度 (m/s)'},
            {'column': 'no_stop_pass_speed', 'name': '车均不停车通过速度', 'ylabel': '不停车通过速度 (m/s)'},
            {'column': 'delay_index', 'name': '延误指数', 'ylabel': '延误指数'},
            {'column': 'idx_state', 'name': '路口状态', 'ylabel': '路口状态'}
        ]
        
        # 获取日期范围用于主标题
        first_interval = list(data_by_interval.keys())[0]
        dates = sorted(data_by_interval[first_interval]['create_time'].dt.date.unique())
        date_str = format_date_str(dates)
        
        for metric in metrics:
            plt.figure(figsize=(18, 8))
            intervals = [1, 5, 15, 30, 60]
            for i, interval in enumerate(intervals, 1):
                plt.subplot(2, 3, i)
                if interval in data_by_interval and metric['column'] in data_by_interval[interval].columns:
                    df = data_by_interval[interval].copy()
                    df['date'] = df['create_time'].dt.date
                    interval_dates = sorted(df['date'].unique())
                    for j, date in enumerate(interval_dates):
                        date_df = df[df['date'] == date]
                        time_only = date_df['create_time'].dt.time
                        normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
                        plt.plot(normalized_time, date_df[metric['column']], 
                                 marker='o', markersize=2, linewidth=1, 
                                 color=date_colors[j % len(date_colors)],
                                 label=str(date) if i == 1 else "")
                    plt.title(f'{interval}分钟聚合')
                    plt.ylabel(metric['ylabel'])
                    plt.xticks(rotation=45)
                    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    plt.grid(True)
                    if i == 1: plt.legend(loc='upper left', fontsize='small')
            
            plt.suptitle(f'{intersection_name} [{date_str}] - {metric["name"]}不同时间间隔对比', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(comparison_dir, f'{metric["name"]}_时间间隔对比.png'), bbox_inches='tight')
            plt.close()
        
        if 'los' in list(data_by_interval.values())[0].columns:
            plt.figure(figsize=(18, 8))
            los_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'nan': 0}
            for i, interval in enumerate(intervals, 1):
                plt.subplot(2, 3, i)
                if interval in data_by_interval and 'los' in data_by_interval[interval].columns:
                    df = data_by_interval[interval].copy()
                    df['date'] = df['create_time'].dt.date
                    interval_dates = sorted(df['date'].unique())
                    for j, date in enumerate(interval_dates):
                        date_df = df[df['date'] == date]
                        los_values = date_df['los'].map(los_mapping).fillna(0)
                        time_only = date_df['create_time'].dt.time
                        normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
                        plt.plot(normalized_time, los_values, 
                                 marker='o', markersize=2, linewidth=1, 
                                 color=date_colors[j % len(date_colors)],
                                 label=str(date) if i == 1 else "")
                    plt.title(f'{interval}分钟聚合')
                    plt.ylabel('延误指数评级')
                    plt.xticks(rotation=45)
                    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    plt.yticks([0, 1, 2, 3, 4, 5], ['A', 'B', 'C', 'D', 'E', 'F'])
                    plt.grid(True)
                    if i == 1: plt.legend(loc='upper left', fontsize='small')
            plt.suptitle(f'{intersection_name} [{date_str}] - 延误指数评级不同时间间隔对比', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(comparison_dir, '延误指数评级_时间间隔对比.png'), bbox_inches='tight')
            plt.close()

    except Exception as e:
        pass

def get_date_ranges():
    time_periods = [
        (True, '2026-03-10 00:00:00', '2026-03-10 23:59:59'),  
        (False, '2026-03-11 00:00:00', '2026-03-11 23:59:59'),  
        (False, '2026-03-12 00:00:00', '2026-03-12 23:59:59'), 
        (False, '2026-03-22 00:00:00', '2026-03-22 23:59:59'), 
        (False, '2026-03-29 00:00:00', '2026-03-29 23:59:59')  
    ]
    date_ranges = []
    for enable, start_date, end_date in time_periods:
        if enable:
            try: date_ranges.append((pd.Timestamp(start_date), pd.Timestamp(end_date)))
            except: pass
    if not date_ranges: date_ranges.append(None)
    return date_ranges

def analyze_data_completeness(df, date_ranges):
    """分析数据完整性"""
    print("\n开始数据完整性分析...")
    if date_ranges and date_ranges[0] is not None:
        for start_time, end_time in date_ranges:
            complete_minutes = pd.date_range(start=start_time, end=end_time, freq='1min')
            date_mask = (df['create_time'] >= start_time) & (df['create_time'] <= end_time)
            actual_minutes = pd.DatetimeIndex(df[date_mask]['create_time'].dt.floor('1min'))
            missing_minutes = complete_minutes.difference(actual_minutes)
            
            if not missing_minutes.empty:
                print(f"\n日期 {start_time.date()} 数据完整性分析:")
                print(f"缺失率: {len(missing_minutes)/len(complete_minutes):.2%}")
            else:
                print(f"\n日期 {start_time.date()} 数据完整性分析: ✓ 数据完整")
    else:
        if not df.empty:
            dates = df['create_time'].dt.date.unique()
            for date in dates:
                start_time = pd.Timestamp(f'{date} 00:00:00')
                end_time = pd.Timestamp(f'{date} 23:59:59')
                complete_minutes = pd.date_range(start=start_time, end=end_time, freq='1min')
                date_mask = df['create_time'].dt.date == date
                actual_minutes = pd.DatetimeIndex(df[date_mask]['create_time'].dt.floor('1min'))
                missing_minutes = complete_minutes.difference(actual_minutes)
                if not missing_minutes.empty:
                    print(f"\n日期 {date} 数据完整性分析: 缺失率 {len(missing_minutes)/len(complete_minutes):.2%}")
                else:
                    print(f"\n日期 {date} 数据完整性分析: ✓ 数据完整")
    print("\n数据完整性分析完成！")

def analyze_traffic_data(info_csv_path, dir_csv_path, date_ranges, print_raw_data=False):
    """双线分离主控：主表用来算指标，方向表用来画转向图和各方向多指标"""
    
    # ---------------- 1. 加载并处理主线（路口信息） ----------------
    try: df = pd.read_csv(info_csv_path, encoding='GBK')
    except:
        try: df = pd.read_csv(info_csv_path, encoding='utf-8')
        except: df = pd.read_csv(info_csv_path, encoding='latin1')
    
    df['create_time'] = pd.to_datetime(df['create_time'])
    df = df.sort_values('create_time')
    intersection_name = df['inter_name'].iloc[0] if 'inter_name' in df.columns and not df.empty else 'Unknown'

    # ---------------- 2. 加载并处理副线（方向转向信息） ----------------
    df_dir = None
    if dir_csv_path and os.path.exists(dir_csv_path):
        print(f"\n[INFO] 发现方向指标表，启动后台流向计算副线: {os.path.basename(dir_csv_path)}")
        try: df_dir = pd.read_csv(dir_csv_path, encoding='GBK')
        except:
            try: df_dir = pd.read_csv(dir_csv_path, encoding='utf-8')
            except: df_dir = pd.read_csv(dir_csv_path, encoding='latin1')
            
        df_dir['create_time'] = pd.to_datetime(df_dir['create_time'])
        df_dir = df_dir.sort_values('create_time')
        
        if 'lng_lat_seq' in df_dir.columns:
            df_dir = enrich_direction_features(df_dir, offset_degree=0.0)
        else:
            df_dir['main_direction'] = 'Unknown'

    # ---------------- 3. 数据过滤与聚合 ----------------
    if date_ranges and date_ranges[0] is not None:
        intervals = [1, 5, 15, 30, 60]
        data_by_interval = {}
        
        for interval in intervals:
            output_dir = os.path.join(os.path.dirname(info_csv_path), f'analysis_{interval}min')
            os.makedirs(output_dir, exist_ok=True)
            
            # --- 主线计算逻辑 ---
            if interval == 1:
                df_interval = df.groupby(pd.Grouper(key='create_time', freq='1min')).first().reset_index()
                if 'los' in df_interval.columns: df_interval['los'] = df_interval['los'].astype(str)
                if 'idx_state' in df.columns:
                    idx_state_values = df.groupby(pd.Grouper(key='create_time', freq='1min'))['idx_state'].first().reset_index()
                    idx_state_values = idx_state_values.rename(columns={'idx_state': 'idx_state_new'})
                    df_interval = df_interval.merge(idx_state_values, on='create_time', how='left')
                    df_interval = df_interval.drop(columns=['idx_state'])
                    df_interval = df_interval.rename(columns={'idx_state_new': 'idx_state'})
            else:
                numeric_cols = ['queue_len_max', 'queue_len_avg', 'pass_flow', 'stop_time', 'stop_times', 'pass_speed', 'no_stop_pass_speed', 'delay_index', 'confidence']
                numeric_cols = [c for c in numeric_cols if c in df.columns]
                df_interval = df[numeric_cols + ['create_time']].groupby(pd.Grouper(key='create_time', freq=f'{interval}min')).mean().reset_index()
                
                if 'los' in df.columns:
                    def get_mode(series):
                        if len(series.mode()) > 0: return series.mode().iloc[0]
                        else: return series.iloc[0] if len(series) > 0 else None
                    los_values = df.groupby(pd.Grouper(key='create_time', freq=f'{interval}min'))['los'].apply(get_mode).reset_index()
                    df_interval = df_interval.merge(los_values, on='create_time', how='left')
                
                if 'idx_state' in df.columns:
                    idx_state_values = df.groupby(pd.Grouper(key='create_time', freq=f'{interval}min'))['idx_state'].first().reset_index()
                    df_interval = df_interval.merge(idx_state_values, on='create_time', how='left')
            
            mask = False
            for start_time, end_time in date_ranges:
                mask |= (df_interval['create_time'] >= start_time) & (df_interval['create_time'] <= end_time)
            df_interval = df_interval[mask]
            data_by_interval[interval] = df_interval
            
            # --- 副线过滤与制图 ---
            df_dir_filtered_for_chart = None
            if df_dir is not None and 'turn_dir_no' in df_dir.columns and 'main_direction' in df_dir.columns:
                df_dir_filtered_for_chart = df_dir.copy()
                mask_dir = False
                for start_time, end_time in date_ranges:
                    mask_dir |= (df_dir_filtered_for_chart['create_time'] >= start_time) & (df_dir_filtered_for_chart['create_time'] <= end_time)
                df_dir_filtered_for_chart = df_dir_filtered_for_chart[mask_dir]
                
                generate_turning_flow_charts(df_dir_filtered_for_chart, intersection_name, output_dir, interval)
                generate_directional_metrics_charts(df_dir_filtered_for_chart, intersection_name, output_dir, interval)
            
            # 主线总图绘制
            generate_charts(df_interval, intersection_name, output_dir, interval, df_dir_filtered_for_chart)

        generate_vertical_comparison_charts(data_by_interval, intersection_name, os.path.dirname(info_csv_path))
    else:
        # 当未选择特定日期范围时的逻辑处理
        first_date = df['create_time'].min().date()
        start_time = pd.Timestamp(f'{first_date} 00:00:00')
        end_time = pd.Timestamp(f'{first_date} 23:59:59')
        df = df[(df['create_time'] >= start_time) & (df['create_time'] <= end_time)]
        
        intervals = [1, 5, 15, 30, 60]
        data_by_interval = {}
        
        for interval in intervals:
            output_dir = os.path.join(os.path.dirname(info_csv_path), f'analysis_{interval}min')
            os.makedirs(output_dir, exist_ok=True)
            
            if interval == 1:
                df_interval = df.groupby(pd.Grouper(key='create_time', freq='1min')).first().reset_index()
                if 'los' in df_interval.columns: df_interval['los'] = df_interval['los'].astype(str)
                if 'idx_state' in df.columns:
                    idx_state_values = df.groupby(pd.Grouper(key='create_time', freq='1min'))['idx_state'].first().reset_index()
                    idx_state_values = idx_state_values.rename(columns={'idx_state': 'idx_state_new'})
                    df_interval = df_interval.merge(idx_state_values, on='create_time', how='left')
                    df_interval = df_interval.drop(columns=['idx_state'])
                    df_interval = df_interval.rename(columns={'idx_state_new': 'idx_state'})
            else:
                numeric_cols = ['queue_len_max', 'queue_len_avg', 'pass_flow', 'stop_time', 'stop_times', 'pass_speed', 'no_stop_pass_speed', 'delay_index', 'confidence']
                numeric_cols = [c for c in numeric_cols if c in df.columns]
                df_interval = df[numeric_cols + ['create_time']].groupby(pd.Grouper(key='create_time', freq=f'{interval}min')).mean().reset_index()
                
                if 'los' in df.columns:
                    def get_mode(series):
                        if len(series.mode()) > 0: return series.mode().iloc[0]
                        else: return series.iloc[0] if len(series) > 0 else None
                    los_values = df.groupby(pd.Grouper(key='create_time', freq=f'{interval}min'))['los'].apply(get_mode).reset_index()
                    df_interval = df_interval.merge(los_values, on='create_time', how='left')
                
                if 'idx_state' in df.columns:
                    idx_state_values = df.groupby(pd.Grouper(key='create_time', freq=f'{interval}min'))['idx_state'].first().reset_index()
                    df_interval = df_interval.merge(idx_state_values, on='create_time', how='left')
            
            data_by_interval[interval] = df_interval
            
            df_dir_filtered_for_chart = None
            if df_dir is not None and 'turn_dir_no' in df_dir.columns and 'main_direction' in df_dir.columns:
                df_dir_filtered_for_chart = df_dir[(df_dir['create_time'] >= start_time) & (df_dir['create_time'] <= end_time)]
                
                generate_turning_flow_charts(df_dir_filtered_for_chart, intersection_name, output_dir, interval)
                generate_directional_metrics_charts(df_dir_filtered_for_chart, intersection_name, output_dir, interval)
            
            generate_charts(df_interval, intersection_name, output_dir, interval, df_dir_filtered_for_chart)
            
        generate_vertical_comparison_charts(data_by_interval, intersection_name, os.path.dirname(info_csv_path))

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = [f for f in os.listdir(script_dir) if f.endswith('.csv')]
    
    info_csv, dir_csv = None, None
    
    # 智能识别底层文件，分离主表和附表 (通用化匹配)
    for f in csv_files:
        if "info_view" in f: 
            info_csv = os.path.join(script_dir, f)
        elif "index_view" in f: 
            dir_csv = os.path.join(script_dir, f)
            
    if not info_csv:
        if csv_files: 
            info_csv = os.path.join(script_dir, csv_files[0])
        else:
            print(f"错误：在脚本所在目录 {script_dir} 中没有找到任何 CSV 文件！")
            exit(1)
            
    print(f"=================================")
    print(f"[主线] 自动识别的路口主表：{os.path.basename(info_csv)}")
    if dir_csv:
        print(f"[副线] 自动识别的方向附表：{os.path.basename(dir_csv)}")
    else:
        print(f"\n【警告】未在目录中找到包含 'index_view' 的CSV方向数据！\n程序将正常执行主分析逻辑，但无法生成额外的'图1'和'图2'转向流量图。")
    print(f"=================================\n")
    
    date_ranges = get_date_ranges()
    analyze_traffic_data(info_csv, dir_csv, date_ranges, print_raw_data=False)