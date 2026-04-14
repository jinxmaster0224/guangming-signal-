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
SHOW_OVERLAY_ON_COMBINED_CHART = True  

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

CARDINAL_HANZI = {"E": "东向", "W": "西向", "S": "南向", "N": "北向"}
DIR_COLORS = {'E': '#1f77b4', 'S': '#2ca02c', 'W': '#d62728', 'N': '#9467bd'}
_WKT_COORD_PATTERN = re.compile(r'(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)')

def add_custom_periods_overlay(ax, periods_config, right_ylabel='周期时间 (秒)', base_date='2026-01-01'):
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
    ax2.legend(loc='upper right')

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
            del valid_dirs, counts, best_dirs
        else: df['main_direction'] = df['_raw_direction']
    else: df['main_direction'] = df['_raw_direction']
    
    if '_raw_direction' in df.columns: df.drop(columns=['_raw_direction'], inplace=True)
    gc.collect()
    return df

# ================= 自定义非标准探针 =================
def run_custom_window_analysis(df_info, custom_windows, output_dir):
    if not custom_windows: return
    print("\n" + "═"*60)
    print("🚀 启动 [非标准区间] 自定义诊断探针 (基于路口主表)...")
    results = []
    
    for cw in custom_windows:
        start_time = pd.to_datetime(cw['start'])
        end_time = pd.to_datetime(cw['end'])
        name = cw.get('name', '自定义区间')
        duration_mins = (end_time - start_time).total_seconds() / 60.0
        if duration_mins <= 0: continue
        
        mask = (df_info['create_time'] >= start_time) & (df_info['create_time'] <= end_time)
        df_slice = df_info[mask]
        if df_slice.empty: continue
            
        flow = (df_slice['pass_flow'].sum() / duration_mins) if 'pass_flow' in df_slice.columns else 0
        speed = df_slice['pass_speed'].mean() if 'pass_speed' in df_slice.columns else 0
        delay = df_slice['delay_index'].mean() if 'delay_index' in df_slice.columns else 0
        q_max = df_slice['queue_len_max'].mean() if 'queue_len_max' in df_slice.columns else 0
        
        results.append({
            '诊断主题': name,
            '开始时间': start_time.strftime('%H:%M:%S'),
            '结束时间': end_time.strftime('%H:%M:%S'),
            '时长(分)': round(duration_mins, 1),
            '分均流量(辆)': round(flow, 1),
            '平均速度(m/s)': round(speed, 2),
            '延误指数': round(delay, 2),
            '最大排队(m)': round(q_max, 1),
        })
        
    if results:
        res_df = pd.DataFrame(results)
        print("\n📊 探针诊断结果:")
        print(res_df.to_string(index=False))
        save_path = os.path.join(output_dir, '0_自定义非标准区间诊断报告.csv')
        res_df.to_csv(save_path, index=False, encoding='GBK')
    print("═"*60 + "\n")

# ================= 转向流量专属制图 (仅流量) =================
def generate_turning_flow_charts(df_dir_agg, intersection_name, output_dir, interval):
    try:
        dates = sorted(df_dir_agg['date'].unique())
        directions = ['E', 'S', 'W', 'N']
        
        # 1. 2x2 子区图
        fig_sub, axes_sub = plt.subplots(2, 2, figsize=(16, 12))
        fig_sub.suptitle(f'{intersection_name} - 各进口道转向分均车流量 ({interval}分钟)', fontsize=18, y=0.98)
        axes_sub = axes_sub.flatten()
        
        for i, direction in enumerate(directions):
            ax = axes_sub[i]
            dir_data = df_dir_agg[df_dir_agg['main_direction'] == direction]
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
            ax.set_ylabel('分均车流量 (辆)')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, linestyle='--', alpha=0.7)
            if not dir_data.empty: ax.legend()
            
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(output_dir, f'1_转向流量_分图_{interval}min.png'), dpi=150)
        plt.close(fig_sub)

        # 2. 全景同框图
        fig_all, ax_all = plt.subplots(figsize=(14, 8))
        fig_all.suptitle(f'{intersection_name} - 全进口道转向分均流量同框对比 ({interval}分钟)', fontsize=18)
        
        for direction in directions:
            dir_data = df_dir_agg[df_dir_agg['main_direction'] == direction]
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

        ax_all.set_ylabel('分均车流量 (辆)')
        ax_all.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax_all.tick_params(axis='x', rotation=45)
        ax_all.grid(True, linestyle='--', alpha=0.7)
        ax_all.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
        
        if SHOW_OVERLAY_ON_COMBINED_CHART:
            add_custom_periods_overlay(ax_all, CUSTOM_PERIODS, right_ylabel='')
        
        plt.tight_layout(rect=[0, 0, 0.85, 0.96]) 
        plt.savefig(os.path.join(output_dir, f'2_转向流量_全同框_{interval}min.png'), dpi=150)
        plt.close(fig_all)
    except Exception as e: 
        pass

# ================= [🔥重构] 全微观指标独立 2x2 阵列图 =================
def generate_turning_all_metrics_charts(df_dir_agg, intersection_name, output_dir, interval):
    """
    为每一个微观指标（排队、速度、延误等）单独生成一张 2x2（东南西北）的子图阵列，
    彻底解决指标线条混在一起的痛点。
    """
    try:
        dates = sorted(df_dir_agg['date'].unique())
        metrics = [
            ('queue_len_max', '最大排队长度(m)', '最大排队长度'),
            ('queue_len_avg', '平均排队长度(m)', '平均排队长度'),
            ('pass_speed', '通过速度(m/s)', '平均通过速度'),
            ('stop_time', '停车时间(s)', '平均灯前停车时间'),
            ('stop_times', '停车次数(次)', '平均停车次数'),
            ('no_stop_pass_speed', '不停车通过速度(m/s)', '不停车通过速度'),
            ('delay_index', '延误指数', '延误指数'),
            ('pass_flow', '分均车流量(辆)', '分均车流量')
        ]
        
        directions = ['E', 'S', 'W', 'N']
        turn_names = {1: '左转', 2: '直行'}
        line_styles = {1: '--', 2: '-'}
        markers = {1: 'o', 2: 's'}
        
        for col, ylabel, metric_cn in metrics:
            # 防御：如果这个指标在数据中全为空，则跳过不画图
            if col not in df_dir_agg.columns or df_dir_agg[col].isna().all():
                continue
                
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{intersection_name} - 各进口道【{metric_cn}】微观监控 ({interval}分钟)', fontsize=18, y=0.98)
            axes = axes.flatten()
            
            has_any_data = False
            
            for idx, d in enumerate(directions):
                ax = axes[idx]
                dir_color = DIR_COLORS.get(d, 'black')
                
                for t in [1, 2]:
                    subset = df_dir_agg[(df_dir_agg['main_direction'] == d) & (df_dir_agg['turn_dir_no'] == t)]
                    if subset.empty: continue
                    
                    for date in dates:
                        date_subset = subset[subset['date'] == date]
                        if date_subset.empty or date_subset[col].isna().all(): continue
                        
                        norm_time = pd.to_datetime('2026-01-01 ' + date_subset['create_time'].dt.time.astype(str))
                        date_prefix = f"{date} " if len(dates) > 1 else ""
                        
                        # 因为子图已经是东向/西向了，这里图例只标明“左转/直行”即可，非常清爽
                        ax.plot(norm_time, date_subset[col],
                                color=dir_color, linestyle=line_styles[t],
                                marker=markers[t], markersize=3, alpha=0.8,
                                label=f"{date_prefix}{turn_names[t]}")
                        has_any_data = True
                        
                ax.set_title(f'{CARDINAL_HANZI.get(d, d)}进口道', fontsize=14, color=dir_color, fontweight='bold')
                ax.set_ylabel(ylabel, fontsize=12)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # 如果有数据，生成图例
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(fontsize=10, loc='best')
                    
                if SHOW_OVERLAY_ON_COMBINED_CHART:
                    add_custom_periods_overlay(ax, CUSTOM_PERIODS, right_ylabel='')
                    
            if has_any_data:
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                # 输出单独的一张图表，例如：3_微观拆解_最大排队长度_15min.png
                plt.savefig(os.path.join(output_dir, f'3_微观拆解_{metric_cn}_{interval}min.png'), dpi=150)
            plt.close(fig)

    except Exception as e:
        print(f"微观转向指标图谱生成出错: {e}")

# ================= 主线大图绘制 (基于 info_view) =================
def generate_intersection_macro_charts(df_info_agg, intersection_name, output_dir, interval, df_dir_agg=None):
    """路口级宏观大盘：完美恢复 LOS 和 idx_state 曲线波动"""
    try:
        df_info_agg['date'] = df_info_agg['create_time'].dt.date
        dates = sorted(df_info_agg['date'].unique())
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'gray']
        
        plt.figure(figsize=(16, 12))
        fig = plt.gcf()
        fig.suptitle(f'{intersection_name} - 路口级整体运行大盘趋势 ({interval}分钟)', fontsize=18, y=0.98)
        
        metrics_plot = [
            ('queue_len_max', '最大排队长度（米）', '宏观 - 最大排队长度'),
            ('queue_len_avg', '平均排队长度（米）', '宏观 - 平均排队长度'),
            ('pass_flow', '分均总车流量（辆）', '宏观 - 路口总车流量'),
            ('stop_time', '停车时间 (秒)', '宏观 - 平均停车时间'),
            ('stop_times', '停车次数（次）', '宏观 - 平均停车次数'),
            ('pass_speed', '通过速度 (m/s)', '宏观 - 平均通过速度'),
            ('no_stop_pass_speed', '不停车通过速度 (m/s)', '宏观 - 平均不停车速度'),
            ('delay_index', '延误指数', '宏观 - 整体延误指数'),
            ('idx_state', '路口拥堵状态', '宏观 - 路口状态评估')
        ]
        
        for idx, (col, ylabel, title) in enumerate(metrics_plot, 1):
            plt.subplot(4, 3, idx)
            for i, date in enumerate(dates):
                date_df = df_info_agg[df_info_agg['date'] == date]
                if date_df.empty or col not in date_df.columns: continue
                
                time_only = date_df['create_time'].dt.time
                normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
                plt.plot(normalized_time, date_df[col], marker='o', markersize=2, linewidth=1, 
                         color=colors[i % len(colors)], label=str(date))
            plt.title(title)
            plt.ylabel(ylabel)
            plt.xticks(rotation=45)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.legend(loc='upper left', fontsize=8)
            plt.grid(True, linestyle='--', alpha=0.6)
            if SHOW_OVERLAY_ON_COMBINED_CHART:
                add_custom_periods_overlay(plt.gca(), CUSTOM_PERIODS, right_ylabel='')
        
        # 第10格子：完美的延误指数评级 (LOS)
        if 'los' in df_info_agg.columns:
            plt.subplot(4, 3, 10)
            los_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'nan': 0}
            for i, date in enumerate(dates):
                date_df = df_info_agg[df_info_agg['date'] == date]
                los_values = date_df['los'].map(los_mapping).fillna(0)
                time_only = date_df['create_time'].dt.time
                normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
                plt.plot(normalized_time, los_values, marker='o', markersize=2, linewidth=1, 
                         color=colors[i % len(colors)], label=str(date))
            plt.title('宏观 - 延误指数评级 (LOS)')
            plt.ylabel('评级阶梯')
            plt.xticks(rotation=45)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.yticks([0, 1, 2, 3, 4, 5], ['A', 'B', 'C', 'D', 'E', 'F'])
            plt.legend(loc='upper left', fontsize=8)
            plt.grid(True, linestyle='--', alpha=0.6)
            if SHOW_OVERLAY_ON_COMBINED_CHART:
                add_custom_periods_overlay(plt.gca(), CUSTOM_PERIODS, right_ylabel='')

        # 第11、12格子：留给车流量转向作为补充参照
        ax_turn = plt.subplot(4, 3, 11)
        turn_handles, turn_labels = [], []
        if df_dir_agg is not None and not df_dir_agg.empty:
            dates_turn = sorted(df_dir_agg['date'].unique())
            directions = ['E', 'S', 'W', 'N']
            for d in directions:
                for t in [1, 2]:
                    subset = df_dir_agg[(df_dir_agg['main_direction'] == d) & (df_dir_agg['turn_dir_no'] == t)]
                    if subset.empty: continue
                    for date in dates_turn:
                        ds = subset[subset['date'] == date]
                        if ds.empty: continue
                        norm_time = pd.to_datetime('2026-01-01 ' + ds['create_time'].dt.time.astype(str))
                        dp = f"{date} " if len(dates_turn) > 1 else ""
                        ls = '--' if t == 1 else '-'
                        line, = ax_turn.plot(norm_time, ds['pass_flow'], 
                                     color=DIR_COLORS.get(d, 'black'), linestyle=ls, marker='o', markersize=2, alpha=0.8)
                        turn_handles.append(line)
                        turn_labels.append(f"{dp}{CARDINAL_HANZI.get(d)}{'左转' if t==1 else '直行'}")
            ax_turn.set_title('微观参照 - 转向分均流量')
            ax_turn.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.setp(ax_turn.get_xticklabels(), rotation=45)
            ax_turn.grid(True, linestyle='--', alpha=0.6)
        
        ax_leg = plt.subplot(4, 3, 12)
        ax_leg.axis('off')
        if turn_handles:
            by_label = dict(zip(turn_labels, turn_handles))
            ax_leg.legend(by_label.values(), by_label.keys(), loc='center', fontsize=8, ncol=2)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(output_dir, f'4_路口整体大盘趋势_{interval}min.png'), dpi=150)
        plt.close()
    except Exception as e:
        print(f"生成主线图表时出错: {e}")

def generate_vertical_comparison_charts(data_by_interval, intersection_name, output_dir):
    try:
        comparison_dir = os.path.join(output_dir, '5_宏观纵向对比')
        os.makedirs(comparison_dir, exist_ok=True)
        date_colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'gray']
        metrics = [
            {'column': 'queue_len_max', 'name': '最大排队长度', 'ylabel': '最大排队长度'}, 
            {'column': 'queue_len_avg', 'name': '平均排队长度', 'ylabel': '平均排队长度'},
            {'column': 'pass_flow', 'name': '分均车流量', 'ylabel': '分均车流量 (辆)'},
            {'column': 'stop_time', 'name': '车均灯前停车时间', 'ylabel': '停车时间 (秒)'},
            {'column': 'pass_speed', 'name': '车均通过速度', 'ylabel': '通过速度 (m/s)'},
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
                    if i == 1: plt.legend(loc='upper left', fontsize='small')
            
            plt.suptitle(f'{intersection_name} - {metric["name"]}宏观时间跨度对比', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(comparison_dir, f'{metric["name"]}_时间对比.png'))
            plt.close()
    except: pass

def get_date_ranges():
    time_periods = [
        (True, '2026-03-10 00:00:00', '2026-03-10 23:59:59'),  
        (False, '2026-03-11 00:00:00', '2026-03-11 23:59:59'),  
        (False, '2026-03-12 00:00:00', '2026-03-12 23:59:59')
    ]
    date_ranges = []
    for enable, start_date, end_date in time_periods:
        if enable:
            try: date_ranges.append((pd.Timestamp(start_date), pd.Timestamp(end_date)))
            except: pass
    if not date_ranges: date_ranges.append(None)
    return date_ranges


def analyze_traffic_data(info_csv_path, dir_csv_path, date_ranges, offset_degree=0.0, custom_windows=None):
    if not info_csv_path or not dir_csv_path:
        print("❌ 缺少必要的原始表，系统需同时载入 info_view 和 index_view。")
        return

    # ==== 加载 INFO 表 (负责宏观大盘) ====
    try: df_info = pd.read_csv(info_csv_path, encoding='GBK')
    except: df_info = pd.read_csv(info_csv_path, encoding='utf-8')
    df_info['create_time'] = pd.to_datetime(df_info['create_time'])
    df_info = df_info.sort_values('create_time')
    intersection_name = df_info['inter_name'].iloc[0] if 'inter_name' in df_info.columns and not df_info.empty else 'Unknown'

    # ==== 加载 DIR 表 (负责微观转向) ====
    try: df_dir = pd.read_csv(dir_csv_path, encoding='GBK')
    except: df_dir = pd.read_csv(dir_csv_path, encoding='utf-8')
    df_dir['create_time'] = pd.to_datetime(df_dir['create_time'])
    df_dir = df_dir.sort_values('create_time')
    
    # 清洗 DIR 表
    if 'lng_lat_seq' in df_dir.columns:
        print(f"  -> [配置生效] 当前路口偏转补偿角度已设定为: {offset_degree}°")
        df_dir = enrich_direction_features(df_dir, offset_degree=offset_degree)
    else: df_dir['main_direction'] = 'Unknown'
    
    # 微观表严格限定转向数据
    if 'turn_dir_no' in df_dir.columns:
        df_dir = df_dir[df_dir['turn_dir_no'].isin([1, 2])].copy()

    # ==== 自定义探针分析 ====
    if custom_windows:
        run_custom_window_analysis(df_info, custom_windows, os.path.dirname(info_csv_path))

    # ==== 过滤时间区间 ====
    if date_ranges and date_ranges[0] is not None:
        mask_info = pd.Series(False, index=df_info.index)
        mask_dir = pd.Series(False, index=df_dir.index)
        for start_time, end_time in date_ranges:
            mask_info |= (df_info['create_time'] >= start_time) & (df_info['create_time'] <= end_time)
            mask_dir |= (df_dir['create_time'] >= start_time) & (df_dir['create_time'] <= end_time)
        df_info = df_info[mask_info]
        df_dir = df_dir[mask_dir]
        
        data_by_interval = {}
        intervals = [1, 5, 15, 30, 60]
        
        for interval in intervals:
            output_dir = os.path.join(os.path.dirname(info_csv_path), f'analysis_{interval}min')
            os.makedirs(output_dir, exist_ok=True)
            
            # --- 1. INFO 宏观聚合 (防二次平滑) ---
            df_info_src = df_info[df_info['create_time'].dt.minute % 5 == 0].copy() if interval >= 5 else df_info.copy()
            state_cols = ['queue_len_max', 'queue_len_avg', 'stop_time', 'stop_times', 'pass_speed', 'no_stop_pass_speed', 'delay_index']
            state_cols = [c for c in state_cols if c in df_info_src.columns]
            df_info_state = df_info_src[state_cols + ['create_time']].groupby(pd.Grouper(key='create_time', freq=f'{interval}min')).mean().reset_index()
            
            if 'pass_flow' in df_info_src.columns:
                df_info_flow = df_info_src[['pass_flow', 'create_time']].groupby(pd.Grouper(key='create_time', freq=f'{interval}min')).sum().reset_index()
                df_info_flow['pass_flow'] = df_info_flow['pass_flow'] / (max(1, interval // 5) if interval >= 5 else interval)
                df_info_agg = pd.merge(df_info_state, df_info_flow, on='create_time', how='left')
            else: df_info_agg = df_info_state
            
            # 恢复 LOS 和 idx_state
            if 'los' in df_info_src.columns:
                def get_mode(series):
                    if len(series.mode()) > 0: return series.mode().iloc[0]
                    else: return series.iloc[0] if len(series) > 0 else None
                los_values = df_info_src.groupby(pd.Grouper(key='create_time', freq=f'{interval}min'))['los'].apply(get_mode).reset_index()
                df_info_agg = df_info_agg.merge(los_values, on='create_time', how='left')
            if 'idx_state' in df_info_src.columns:
                idx_state_values = df_info_src.groupby(pd.Grouper(key='create_time', freq=f'{interval}min'))['idx_state'].first().reset_index()
                df_info_agg = df_info_agg.merge(idx_state_values, on='create_time', how='left')
            
            data_by_interval[interval] = df_info_agg
            
            # --- 2. DIR 微观转向聚合 (防二次平滑) ---
            df_dir_src = df_dir[df_dir['create_time'].dt.minute % 5 == 0].copy() if interval >= 5 else df_dir.copy()
            grouper_dir = [pd.Grouper(key='create_time', freq=f'{interval}min'), 'main_direction', 'turn_dir_no']
            
            dir_state_cols = [c for c in state_cols if c in df_dir_src.columns]
            df_dir_state = df_dir_src[dir_state_cols + ['create_time', 'main_direction', 'turn_dir_no']].groupby(grouper_dir).mean().reset_index()
            
            if 'pass_flow' in df_dir_src.columns:
                df_dir_flow = df_dir_src[['pass_flow', 'create_time', 'main_direction', 'turn_dir_no']].groupby(grouper_dir).sum().reset_index()
                df_dir_flow['pass_flow'] = df_dir_flow['pass_flow'] / (max(1, interval // 5) if interval >= 5 else interval)
                df_dir_agg = pd.merge(df_dir_state, df_dir_flow, on=['create_time', 'main_direction', 'turn_dir_no'], how='left')
            else: df_dir_agg = df_dir_state
            df_dir_agg['date'] = df_dir_agg['create_time'].dt.date
            
            # --- 3. 图表输出 ---
            print(f"  [{interval}分钟] 正在输出图表...")
            generate_turning_flow_charts(df_dir_agg, intersection_name, output_dir, interval)
            generate_turning_all_metrics_charts(df_dir_agg, intersection_name, output_dir, interval) # 🔥 重构后的一指标一图生成器
            generate_intersection_macro_charts(df_info_agg, intersection_name, output_dir, interval, df_dir_agg) 

        generate_vertical_comparison_charts(data_by_interval, intersection_name, os.path.dirname(info_csv_path))


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = [f for f in os.listdir(script_dir) if f.endswith('.csv')]
    
    info_csv, dir_csv = None, None
    for f in csv_files:
        if "info_view" in f: info_csv = os.path.join(script_dir, f)
        elif "index_view" in f: dir_csv = os.path.join(script_dir, f)
            
    print(f"=================================")
    print(f"路口级宏观主表: {os.path.basename(info_csv) if info_csv else '缺失！'}")
    print(f"转向级微观副表: {os.path.basename(dir_csv) if dir_csv else '缺失！'}")
    print(f"=================================\n")
    
    # ================= 配置区 =================
    #False,True
    OFFSET_DEGREE = -50.0  
    ENABLE_CUSTOM_PROBE = False
    
    CUSTOM_ANALYSIS_WINDOWS = [
        {"start": "2026-03-10 08:07:00", "end": "2026-03-10 08:22:00", "name": "早高峰突发拥堵诊断 (15分钟)"}
    ]
    # ==========================================
    
    date_ranges = get_date_ranges()
    probe_windows = CUSTOM_ANALYSIS_WINDOWS if ENABLE_CUSTOM_PROBE else None
    
    analyze_traffic_data(
        info_csv, dir_csv, 
        date_ranges, 
        offset_degree=OFFSET_DEGREE,
        custom_windows=probe_windows
    )