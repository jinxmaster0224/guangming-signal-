import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import math
import re
import gc

# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  

# ================= 配置时间段与周期显示 =================
# True: 显示; False: 不显示
SHOW_OVERLAY_ON_COMBINED_CHART = True  

# 路口配时方案
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

# 转向流量颜色配置
CARDINAL_HANZI = {"E": "东向", "W": "西向", "S": "南向", "N": "北向"}
DIR_COLORS = {'E': '#1f77b4', 'S': '#2ca02c', 'W': '#d62728', 'N': '#9467bd'}
_WKT_COORD_PATTERN = re.compile(r'(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)')

def add_custom_periods_overlay(ax, periods_config, right_ylabel='周期时间 (秒)', base_date='2026-01-01'):
    """添加时间段红色虚线和信号周期黑色粗虚线"""
    if not periods_config:
        return
        
    times = []
    values = []
    
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
    ax2.legend(loc='upper right')
# =================================================================

# ================= 空间与流向打标模块 =================
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
    print("  -> [进度] 正在计算坐标偏转角与轨迹流向...")
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
    """生成各个方向的转向流量图 (分图与同框)"""
    try:
        df_turn = df[df['turn_dir_no'].isin([1, 2])].dropna(subset=['main_direction'])
        if df_turn.empty: return
            
        grouped = df_turn.groupby([
            pd.Grouper(key='create_time', freq=f'{interval}min'), 
            'main_direction', 'turn_dir_no'
        ])['pass_flow'].mean().reset_index()
        
        grouped['date'] = grouped['create_time'].dt.date
        dates = sorted(grouped['date'].unique())
        directions = ['E', 'S', 'W', 'N']
        
        # --- 1. 2x2 子区图 ---
        fig_sub, axes_sub = plt.subplots(2, 2, figsize=(16, 12))
        fig_sub.suptitle(f'{intersection_name} - 各进口道转向平均车流量 ({interval}分钟聚合)', fontsize=18, y=0.98)
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
            ax.set_ylabel('平均车流量 (辆)', fontsize=12)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, linestyle='--', alpha=0.7)
            if not dir_data.empty: ax.legend()
            
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(output_dir, f'1_转向流量_分图_{interval}min.png'), dpi=150)
        plt.close(fig_sub)

        # --- 2. 全景同框图 ---
        fig_all, ax_all = plt.subplots(figsize=(14, 8))
        fig_all.suptitle(f'{intersection_name} - 全进口道转向平均流量对比 ({interval}分钟聚合)', fontsize=18)
        
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
        ax_all.set_ylabel('平均车流量 (辆)', fontsize=12)
        ax_all.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax_all.tick_params(axis='x', rotation=45)
        ax_all.grid(True, linestyle='--', alpha=0.7)
        ax_all.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
        
        plt.tight_layout(rect=[0, 0, 0.85, 0.96]) 
        plt.savefig(os.path.join(output_dir, f'2_转向流量_全同框_{interval}min.png'), dpi=150)
        plt.close(fig_all)
        print(f"  -> {interval}分钟转向流量图表生成完成")
    except Exception as e: 
        print(f"转向单图时序图出错: {e}")
# =================================================================

def generate_charts(df, intersection_name, output_dir, interval):
    """
    生成指定时间间隔的折线图
    """
    try:
        print(f"开始生成{interval}分钟聚合的图表...")
        
        # 按日期分组
        df['date'] = df['create_time'].dt.date
        dates = sorted(df['date'].unique())
        
        # 定义颜色映射
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'gray']
        
        # ================= 1. 生成最大排队长度折线图 =================
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            ax.plot(normalized_time, date_df['queue_len_max'], marker='o', markersize=3, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        ax.set_title(f'{intersection_name} - 最大排队长度变化趋势 ({interval}分钟)')
        ax.set_xlabel('时间')
        ax.set_ylabel('最大排队长度')
        ax.grid(True)
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        add_custom_periods_overlay(ax, CUSTOM_PERIODS, right_ylabel='周期时间 (秒)')
        ax.legend(loc='upper left')
        
        plt.tight_layout()
        queue_max_path = os.path.join(output_dir, f'queue_len_max_trend_{interval}min.png')
        plt.savefig(queue_max_path)
        plt.close()
        
        # ================= 2. 生成平均排队长度折线图 =================
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            ax.plot(normalized_time, date_df['queue_len_avg'], marker='o', markersize=3, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        ax.set_title(f'{intersection_name} - 平均排队长度变化趋势 ({interval}分钟)')
        ax.set_xlabel('时间')
        ax.set_ylabel('平均排队长度')
        ax.grid(True)
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        add_custom_periods_overlay(ax, CUSTOM_PERIODS, right_ylabel='周期时间 (秒)')
        ax.legend(loc='upper left')
        
        plt.tight_layout()
        queue_avg_path = os.path.join(output_dir, f'queue_len_avg_trend_{interval}min.png')
        plt.savefig(queue_avg_path)
        plt.close()
        
        # ================= 3. 生成车流量折线图 =================
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            ax.plot(normalized_time, date_df['pass_flow'], marker='o', markersize=3, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        ax.set_title(f'{intersection_name} - 车流量变化趋势 ({interval}分钟)')
        ax.set_xlabel('时间')
        ax.set_ylabel('车流量')
        ax.grid(True)
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        add_custom_periods_overlay(ax, CUSTOM_PERIODS, right_ylabel='周期时间 (秒)')
        ax.legend(loc='upper left')
        
        plt.tight_layout()
        flow_path = os.path.join(output_dir, f'pass_flow_trend_{interval}min.png')
        plt.savefig(flow_path)
        plt.close()
        
        # ================= 4. 生成车均灯前停车时间折线图 =================
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            ax.plot(normalized_time, date_df['stop_time'], marker='o', markersize=3, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        ax.set_title(f'{intersection_name} - 车均灯前停车时间变化趋势 ({interval}分钟)')
        ax.set_xlabel('时间')
        ax.set_ylabel('停车时间 (秒)')
        ax.grid(True)
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        add_custom_periods_overlay(ax, CUSTOM_PERIODS, right_ylabel='周期时间 (秒)')
        ax.legend(loc='upper left')
        
        plt.tight_layout()
        stop_time_path = os.path.join(output_dir, f'stop_time_trend_{interval}min.png')
        plt.savefig(stop_time_path)
        plt.close()

        # ================= 5. 生成车均停车次数折线图 =================
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            ax.plot(normalized_time, date_df['stop_times'], marker='o', markersize=3, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        ax.set_title(f'{intersection_name} - 车均停车次数变化趋势 ({interval}分钟)')
        ax.set_xlabel('时间')
        ax.set_ylabel('停车次数')
        ax.grid(True)
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        add_custom_periods_overlay(ax, CUSTOM_PERIODS, right_ylabel='周期时间 (秒)')
        ax.legend(loc='upper left')
        
        plt.tight_layout()
        stop_times_path = os.path.join(output_dir, f'stop_times_trend_{interval}min.png')
        plt.savefig(stop_times_path)
        plt.close()
        
        # ================= 6. 生成车均通过速度折线图 =================
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            ax.plot(normalized_time, date_df['pass_speed'], marker='o', markersize=3, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        ax.set_title(f'{intersection_name} - 车均通过速度变化趋势 ({interval}分钟)')
        ax.set_xlabel('时间')
        ax.set_ylabel('通过速度 (m/s)')
        ax.grid(True)
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        add_custom_periods_overlay(ax, CUSTOM_PERIODS, right_ylabel='周期时间 (秒)')
        ax.legend(loc='upper left')
        
        plt.tight_layout()
        pass_speed_path = os.path.join(output_dir, f'pass_speed_trend_{interval}min.png')
        plt.savefig(pass_speed_path)
        plt.close()
        
        # ================= 7. 生成车均不停车通过速度折线图 =================
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            ax.plot(normalized_time, date_df['no_stop_pass_speed'], marker='o', markersize=3, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        ax.set_title(f'{intersection_name} - 车均不停车通过速度变化趋势 ({interval}分钟)')
        ax.set_xlabel('时间')
        ax.set_ylabel('不停车通过速度 (m/s)')
        ax.grid(True)
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        add_custom_periods_overlay(ax, CUSTOM_PERIODS, right_ylabel='周期时间 (秒)')
        ax.legend(loc='upper left')
        
        plt.tight_layout()
        no_stop_speed_path = os.path.join(output_dir, f'no_stop_pass_speed_trend_{interval}min.png')
        plt.savefig(no_stop_speed_path)
        plt.close()
        
        # ================= 8. 生成延误指数折线图 =================
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            ax.plot(normalized_time, date_df['delay_index'], marker='o', markersize=3, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        ax.set_title(f'{intersection_name} - 延误指数变化趋势 ({interval}分钟)')
        ax.set_xlabel('时间')
        ax.set_ylabel('延误指数')
        ax.grid(True)
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        add_custom_periods_overlay(ax, CUSTOM_PERIODS, right_ylabel='周期时间 (秒)')
        ax.legend(loc='upper left')
        
        plt.tight_layout()
        delay_path = os.path.join(output_dir, f'delay_index_trend_{interval}min.png')
        plt.savefig(delay_path)
        plt.close()
        
        # ================= 9. 生成路口状态折线图 =================
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            # 创建标准化的时间轴
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            ax.plot(normalized_time, date_df['idx_state'], marker='o', markersize=3, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        ax.set_title(f'{intersection_name} - 路口状态变化趋势 ({interval}分钟)')
        ax.set_xlabel('时间')
        ax.set_ylabel('路口状态')
        ax.grid(True)
        plt.setp(ax.get_xticklabels(), rotation=45)
        # 设置时间格式为小时:分钟
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        add_custom_periods_overlay(ax, CUSTOM_PERIODS, right_ylabel='周期时间 (秒)')
        ax.legend(loc='upper left')
        
        plt.tight_layout()
        idx_state_path = os.path.join(output_dir, f'idx_state_trend_{interval}min.png')
        plt.savefig(idx_state_path)
        plt.close()
        
        # ================= 10. 生成延误指数评级折线图 =================
        if 'los' in df.columns:
            plt.figure(figsize=(12, 6))
            ax = plt.gca()
            # 为分类值创建映射，延误程度越轻值越小，这样在图表中越轻的延误显示在下方
            los_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'nan': 0}
            
            for i, date in enumerate(dates):
                date_df = df[df['date'] == date]
                los_values = date_df['los'].map(los_mapping).fillna(0)
                # 创建标准化的时间轴
                time_only = date_df['create_time'].dt.time
                normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
                # 绘制折线图
                ax.plot(normalized_time, los_values, marker='o', markersize=3, linewidth=1, 
                         color=colors[i % len(colors)], label=str(date))
            
            ax.set_title(f'{intersection_name} - 延误指数评级变化趋势 ({interval}分钟)')
            ax.set_xlabel('时间')
            ax.set_ylabel('延误指数评级')
            ax.grid(True)
            plt.setp(ax.get_xticklabels(), rotation=45)
            # 设置时间格式为小时:分钟
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            # 设置纵坐标显示字母标签
            ax.set_yticks([0, 1, 2, 3, 4, 5])
            ax.set_yticklabels(['A', 'B', 'C', 'D', 'E', 'F'])
            
            add_custom_periods_overlay(ax, CUSTOM_PERIODS, right_ylabel='周期时间 (秒)')
            ax.legend(loc='upper left')
            
            plt.tight_layout()
            los_path = os.path.join(output_dir, f'los_trend_{interval}min.png')
            plt.savefig(los_path)
            plt.close()
        
        # ================= 生成多指标对比图（4x3布局） =================
        plt.figure(figsize=(16, 12))
        
        # 按日期分组
        df['date'] = df['create_time'].dt.date
        dates = sorted(df['date'].unique())
        
        # 定义颜色映射
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'gray']
        
        # 最大排队长度
        plt.subplot(4, 3, 1)
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            plt.plot(normalized_time, date_df['queue_len_max'], marker='o', markersize=2, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        plt.title('最大排队长度趋势图')
        plt.ylabel('最大排队长度（米）')
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.legend(loc='upper left')
        plt.grid(True)
        if SHOW_OVERLAY_ON_COMBINED_CHART:
            add_custom_periods_overlay(plt.gca(), CUSTOM_PERIODS, right_ylabel='周期时间 (秒)')
        
        # 平均排队长度
        plt.subplot(4, 3, 2)
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            plt.plot(normalized_time, date_df['queue_len_avg'], marker='o', markersize=2, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        plt.title('平均排队长度趋势图')
        plt.ylabel('平均排队长度（米）')
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.legend(loc='upper left')
        plt.grid(True)
        if SHOW_OVERLAY_ON_COMBINED_CHART:
            add_custom_periods_overlay(plt.gca(), CUSTOM_PERIODS, right_ylabel='周期时间 (秒)')
        
        # 车流量
        plt.subplot(4, 3, 3)
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            plt.plot(normalized_time, date_df['pass_flow'], marker='o', markersize=2, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        plt.title('车流量趋势图')
        plt.ylabel('车流量（辆）')
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.legend(loc='upper left')
        plt.grid(True)
        if SHOW_OVERLAY_ON_COMBINED_CHART:
            add_custom_periods_overlay(plt.gca(), CUSTOM_PERIODS, right_ylabel='周期时间 (秒)')
        
        # 车均灯前停车时间
        plt.subplot(4, 3, 4)
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            plt.plot(normalized_time, date_df['stop_time'], marker='o', markersize=2, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        plt.title('停车时间趋势图')
        plt.ylabel('停车时间 (秒)')
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.legend(loc='upper left')
        plt.grid(True)
        if SHOW_OVERLAY_ON_COMBINED_CHART:
            add_custom_periods_overlay(plt.gca(), CUSTOM_PERIODS, right_ylabel='周期时间 (秒)')
        
        # 车均停车次数
        plt.subplot(4, 3, 5)
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            plt.plot(normalized_time, date_df['stop_times'], marker='o', markersize=2, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        plt.title('停车次数趋势图')
        plt.ylabel('停车次数（次）')
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.legend(loc='upper left')
        plt.grid(True)
        if SHOW_OVERLAY_ON_COMBINED_CHART:
            add_custom_periods_overlay(plt.gca(), CUSTOM_PERIODS, right_ylabel='周期时间 (秒)')
        
        # 车均通过速度
        plt.subplot(4, 3, 6)
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            plt.plot(normalized_time, date_df['pass_speed'], marker='o', markersize=2, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        plt.title('通过速度趋势图')
        plt.ylabel('通过速度 (m/s)')
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.legend(loc='upper left')
        plt.grid(True)
        if SHOW_OVERLAY_ON_COMBINED_CHART:
            add_custom_periods_overlay(plt.gca(), CUSTOM_PERIODS, right_ylabel='周期时间 (秒)')
        
        # 车均不停车通过速度
        plt.subplot(4, 3, 7)
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            plt.plot(normalized_time, date_df['no_stop_pass_speed'], marker='o', markersize=2, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        plt.title('不停车通过速度趋势图')
        plt.ylabel('不停车通过速度 (m/s)')
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.legend(loc='upper left')
        plt.grid(True)
        if SHOW_OVERLAY_ON_COMBINED_CHART:
            add_custom_periods_overlay(plt.gca(), CUSTOM_PERIODS, right_ylabel='周期时间 (秒)')
        
        # 延误指数
        plt.subplot(4, 3, 8)
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            plt.plot(normalized_time, date_df['delay_index'], marker='o', markersize=2, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        plt.title('延误指数趋势图')
        plt.ylabel('延误指数')
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.legend(loc='upper left')
        plt.grid(True)
        if SHOW_OVERLAY_ON_COMBINED_CHART:
            add_custom_periods_overlay(plt.gca(), CUSTOM_PERIODS, right_ylabel='周期时间 (秒)')
        
        # 路口状态
        plt.subplot(4, 3, 9)
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            plt.plot(normalized_time, date_df['idx_state'], marker='o', markersize=2, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        plt.title('路口状态趋势图')
        plt.ylabel('路口状态')
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.legend(loc='upper left')
        plt.grid(True)
        if SHOW_OVERLAY_ON_COMBINED_CHART:
            add_custom_periods_overlay(plt.gca(), CUSTOM_PERIODS, right_ylabel='周期时间 (秒)')
        
        # 延误指数评级
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
            plt.legend(loc='upper left')
            plt.grid(True)
            if SHOW_OVERLAY_ON_COMBINED_CHART:
                add_custom_periods_overlay(plt.gca(), CUSTOM_PERIODS, right_ylabel='周期时间 (秒)')
        
        plt.tight_layout()
        combined_path = os.path.join(output_dir, f'combined_analysis_{interval}min.png')
        plt.savefig(combined_path)
        plt.close()
        
    except Exception as e:
        print(f"生成图表时出错: {e}")
        import traceback
        traceback.print_exc()

def generate_vertical_comparison_charts(data_by_interval, intersection_name, output_dir):
    """
    生成纵向对比图，显示每个指标在不同时间间隔下的对比
    """
    try:
        print("开始生成纵向对比图...")
        
        comparison_dir = os.path.join(output_dir, '纵向对比图')
        os.makedirs(comparison_dir, exist_ok=True)
        
        date_colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'gray']
        
        metrics = [
            {'column': 'queue_len_max', 'name': '最大排队长度', 'ylabel': '最大排队长度'}, 
            {'column': 'queue_len_avg', 'name': '平均排队长度', 'ylabel': '平均排队长度'},
            {'column': 'pass_flow', 'name': '车流量', 'ylabel': '车流量'},
            {'column': 'stop_time', 'name': '车均灯前停车时间', 'ylabel': '停车时间 (秒)'},
            {'column': 'stop_times', 'name': '车均停车次数', 'ylabel': '停车次数'},
            {'column': 'pass_speed', 'name': '车均通过速度', 'ylabel': '通过速度 (m/s)'},
            {'column': 'no_stop_pass_speed', 'name': '车均不停车通过速度', 'ylabel': '不停车通过速度 (m/s)'},
            {'column': 'delay_index', 'name': '延误指数', 'ylabel': '延误指数'},
            {'column': 'idx_state', 'name': '路口状态', 'ylabel': '路口状态'}
        ]
        
        for metric in metrics:
            plt.figure(figsize=(18, 8))
            intervals = [1, 5, 15, 30, 60]
            
            for i, interval in enumerate(intervals, 1):
                plt.subplot(2, 3, i)
                
                if interval in data_by_interval and metric['column'] in data_by_interval[interval].columns:
                    df = data_by_interval[interval].copy()
                    
                    df['date'] = df['create_time'].dt.date
                    dates = sorted(df['date'].unique())
                    
                    for j, date in enumerate(dates):
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
                    if i == 1:
                        plt.legend(loc='upper left', fontsize='small')
            
            plt.suptitle(f'{intersection_name} - {metric["name"]}不同时间间隔对比', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            chart_path = os.path.join(comparison_dir, f'{metric["name"]}_时间间隔对比.png')
            plt.savefig(chart_path)
            plt.close()
        
        if 'los' in list(data_by_interval.values())[0].columns:
            plt.figure(figsize=(18, 8))
            intervals = [1, 5, 15, 30, 60]
            los_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'nan': 0}
            
            for i, interval in enumerate(intervals, 1):
                plt.subplot(2, 3, i)
                
                if interval in data_by_interval and 'los' in data_by_interval[interval].columns:
                    df = data_by_interval[interval].copy()
                    
                    df['date'] = df['create_time'].dt.date
                    dates = sorted(df['date'].unique())
                    
                    for j, date in enumerate(dates):
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
                    if i == 1:
                        plt.legend(loc='upper left', fontsize='small')
            
            plt.suptitle(f'{intersection_name} - 延误指数评级不同时间间隔对比', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            chart_path = os.path.join(comparison_dir, '延误指数评级_时间间隔对比.png')
            plt.savefig(chart_path)
            plt.close()

    except Exception as e:
        print(f"生成纵向对比图时出错: {e}")
        import traceback
        traceback.print_exc()

def get_date_ranges():
    time_periods = [
        (True, '2026-03-01 00:00:00', '2026-03-01 23:59:59'),  # 2月2日全天
        (True, '2026-03-08 00:00:00', '2026-03-08 23:59:59'),  # 时间段2
        (True, '2026-03-15 00:00:00', '2026-03-15 23:59:59'), # 时间段3
        (True, '2026-03-22 00:00:00', '2026-03-22 23:59:59'), # 时间段4
        (False, '2026-03-29 00:00:00', '2026-03-29 23:59:59')  # 时间段5
    ]
    
    date_ranges = []
    
    for i, (enable, start_date, end_date) in enumerate(time_periods, 1):
        if enable:
            try:
                start_time = pd.Timestamp(start_date)
                end_time = pd.Timestamp(end_date)
                date_ranges.append((start_time, end_time))
            except Exception as e:
                pass
    
    if not date_ranges:
        date_ranges.append(None)
    
    return date_ranges

def analyze_data_completeness(df, date_ranges):
    """分析数据完整性，代码保持不变"""
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

def analyze_traffic_data(csv_path, date_ranges, print_raw_data=False):
    # 读取CSV文件，尝试不同编码
    try:
        df = pd.read_csv(csv_path, encoding='GBK')
    except:
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except:
            df = pd.read_csv(csv_path, encoding='latin1')
    
    # 转换时间格式并排序
    df['create_time'] = pd.to_datetime(df['create_time'])
    df = df.sort_values('create_time')
    
    intersection_name = df['inter_name'].iloc[0] if not df.empty else 'Unknown'

    # === 新增：空间坐标打标与流向对齐 ===
    if 'lng_lat_seq' in df.columns:
        df = enrich_direction_features(df, offset_degree=0.0)
    else:
        df['main_direction'] = 'Unknown'
    # ==================================

    if date_ranges and date_ranges[0] is not None:
        analyze_data_completeness(df, date_ranges)
        intervals = [1, 5, 15, 30, 60]
        data_by_interval = {}
        
        for interval in intervals:
            output_dir = os.path.join(os.path.dirname(csv_path), f'analysis_{interval}min')
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
                # 仅保留在 df 中的数值列，防止报错
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
            
            # 过滤聚合后的数据
            if date_ranges and date_ranges[0] is not None:
                mask = False
                for start_time, end_time in date_ranges:
                    mask |= (df_interval['create_time'] >= start_time) & (df_interval['create_time'] <= end_time)
                df_interval = df_interval[mask]
            
            data_by_interval[interval] = df_interval
            generate_charts(df_interval, intersection_name, output_dir, interval)

            # === 新增：生成单独的 转向流量图 ===
            if 'turn_dir_no' in df.columns and 'main_direction' in df.columns:
                df_filtered_for_turn = df.copy()
                mask = False
                for start_time, end_time in date_ranges:
                    mask |= (df_filtered_for_turn['create_time'] >= start_time) & (df_filtered_for_turn['create_time'] <= end_time)
                df_filtered_for_turn = df_filtered_for_turn[mask]
                generate_turning_flow_charts(df_filtered_for_turn, intersection_name, output_dir, interval)
            # ==================================
            
        generate_vertical_comparison_charts(data_by_interval, intersection_name, os.path.dirname(csv_path))
    else:
        first_date = df['create_time'].min().date()
        start_time = pd.Timestamp(f'{first_date} 00:00:00')
        end_time = pd.Timestamp(f'{first_date} 23:59:59')
        df = df[(df['create_time'] >= start_time) & (df['create_time'] <= end_time)]
        
        analyze_data_completeness(df, date_ranges)
        intervals = [1, 5, 15, 30, 60]
        data_by_interval = {}
        
        for interval in intervals:
            output_dir = os.path.join(os.path.dirname(csv_path), f'analysis_{interval}min')
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
            generate_charts(df_interval, intersection_name, output_dir, interval)

            # === 新增：生成单独的 转向流量图 ===
            if 'turn_dir_no' in df.columns and 'main_direction' in df.columns:
                generate_turning_flow_charts(df, intersection_name, output_dir, interval)
            # ==================================
            
        generate_vertical_comparison_charts(data_by_interval, intersection_name, os.path.dirname(csv_path))

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = [f for f in os.listdir(script_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"错误：在脚本所在目录 {script_dir} 中没有找到CSV文件！")
        exit(1)
    elif len(csv_files) > 1:
        print("警告：脚本所在目录中有多个CSV文件，将使用第一个文件：")
        for i, csv_file in enumerate(csv_files):
            print(f"{i+1}. {csv_file}")
        csv_file = csv_files[0]
    else:
        csv_file = csv_files[0]
    
    csv_file_path = os.path.join(script_dir, csv_file)
    print(f"使用CSV文件：{csv_file_path}")
    
    print_raw_data = False
    date_ranges = get_date_ranges()
    analyze_traffic_data(csv_file_path, date_ranges, print_raw_data=print_raw_data)