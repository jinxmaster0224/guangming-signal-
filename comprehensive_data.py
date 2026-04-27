import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import matplotlib.dates as mdates
import math
import re
import gc
import numpy as np

# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']  
plt.rcParams['axes.unicode_minus'] = False  

# ================= 核心业务与拓扑配置区 =================
TARGET_DIRECTION = 'S' 

INTER_ID_NAME_MAP = {
    '6caaa6f15b735a': '光明大道与光辉大道',
    '6caa3ec15b566e': '光明大道与光明大街',
    '6caa06b15b50eb': '光明大道与光安路',
    '6ca93d215b3ee3': '光明大道与华夏路',
    '6ca887115b2e22': '光明大道与华裕路'
}

CORRIDOR_LINKS = [
    {"name": "光明大道与光辉大道", "seq": 1, "dist_to_next": 857},
    {"name": "光明大道与光明大街", "seq": 2, "dist_to_next": 150},
    {"name": "光明大道与光安路",   "seq": 3, "dist_to_next": 600},
    {"name": "光明大道与华夏路",   "seq": 4, "dist_to_next": 555}, 
    {"name": "光明大道与华裕路",   "seq": 5, "dist_to_next": 0}
]

INTER_OFFSET_MAP = {
    '光明大道与光辉大道': 0.0,
    '光明大道与光明大街': -32.57,
    '光明大道与光安路': -33.69,
    '光明大道与华夏路': -63.5,
    '光明大道与华裕路': -63.0,
}

INTER_CONFIG = {}
for i_id, i_name in INTER_ID_NAME_MAP.items():
    link_info = next((item for item in CORRIDOR_LINKS if item["name"] == i_name), None)
    INTER_CONFIG[i_id] = {
        "name": i_name,
        "offset": INTER_OFFSET_MAP.get(i_name, 0.0),  
        "seq": link_info["seq"] if link_info else 99,
        "dist_to_next": link_info["dist_to_next"] if link_info else 0
    }

SHOW_OVERLAY_ON_COMBINED_CHART = True  

PEAK_PERIODS = {
    '早高峰': ('07:30', '09:00'),
    '晚高峰': ('17:30', '19:30')
}

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
    return ax2

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

# ================= 【极简版：宏观诊断评估生成】 =================
def run_bottleneck_diagnosis_on_slice(df_dir, inter_name):
    df_indexed = df_dir.set_index('create_time')
    results = []

    for period_name, (start, end) in PEAK_PERIODS.items():
        df_peak = df_indexed.between_time(start, end)
        
        for (main_dir, turn_dir), group in df_peak.groupby(['进口道方向', '转向']):
            if group.empty: continue
            
            val_90th_delay = group['延误指数'].quantile(0.90) if '延误指数' in group.columns else None
            avg_delay = group['延误指数'].mean() if '延误指数' in group.columns else None
            max_15min_delay = group['延误指数'].rolling(window=3, min_periods=3).mean().max() if '延误指数' in group.columns else None
            
            avg_stop_times = group['stop_times'].mean() if 'stop_times' in group.columns else None
            avg_no_stop_speed = group['no_stop_pass_speed'].mean() if 'no_stop_pass_speed' in group.columns else None
            mode_los = group['los'].mode()[0] if ('los' in group.columns and not group['los'].mode().empty) else None
            
            avg_queue = None
            max_queue = None
            
            if 'queue_len_avg' in group.columns:
                avg_queue = pd.to_numeric(group['queue_len_avg'], errors='coerce').mean()
            if 'queue_len_max' in group.columns:
                max_queue = pd.to_numeric(group['queue_len_max'], errors='coerce').max()

            results.append({
                '路口名称': inter_name,
                '时段': period_name,
                '进口道方向': main_dir,
                '转向': turn_dir,
                '延误指数': round(avg_delay, 2) if avg_delay else None,
                '90%分位延误': round(val_90th_delay, 2) if val_90th_delay else None,
                '15min平均延误': round(max_15min_delay, 2) if pd.notna(max_15min_delay) else None,
                '停车次数': round(avg_stop_times, 1) if pd.notna(avg_stop_times) else None,
                '不停车速度通过速度': round(avg_no_stop_speed, 1) if pd.notna(avg_no_stop_speed) else None,
                '延误指数评级': mode_los,
                
                '平均排队长度': round(avg_queue, 1) if pd.notna(avg_queue) else None,
                '最大排队长度': round(max_queue, 1) if pd.notna(max_queue) else None
            })
            
    return results

# ================= 【极简版：路口全流向深度诊断图谱】 =================
def generate_all_turning_status_charts(df, inter_name, plot_date, base_output_dir):
    df_date = df[df['create_time'].dt.date == pd.to_datetime(plot_date).date()].copy()
    if df_date.empty: return

    directions = ['E', 'S', 'W', 'N']
    turns = {1: '左转', 2: '直行'}
    
    output_dir = os.path.join(base_output_dir, f'全流向诊断图_{plot_date}')
    os.makedirs(output_dir, exist_ok=True)
    charts_generated = 0

    for d in directions:
        for t_no, t_name in turns.items():
            df_target = df_date[(df_date['main_direction'] == d) & (df_date['turn_dir_no'] == t_no)].copy()
            if df_target.empty: continue
            
            df_target_sampled = df_target.sort_values('create_time')
            
            fig, ax1 = plt.subplots(figsize=(10, 4))
            dir_hanzi = CARDINAL_HANZI.get(d, d)
            
            color_idx = '#1f77b4'
            ax1.plot(df_target_sampled['create_time'], df_target_sampled['延误指数'], 
                     color=color_idx, linewidth=2, marker='o', markersize=4, label=f'{dir_hanzi}{t_name}延误 (5min切片)')
            ax1.set_ylabel('延误指数', color=color_idx, fontsize=12, fontweight='bold')
            ax1.tick_params(axis='y', labelcolor=color_idx)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax1.grid(True, linestyle='--', alpha=0.4)
            ax1.legend(loc='upper left', fontsize=10)
            
            plt.title(f'{inter_name} [{dir_hanzi}{t_name}] 延误指数折线图 ({plot_date})', fontsize=14, pad=15)
            plt.tight_layout()
            
            file_name = f'{dir_hanzi}_{t_name}_延误指数折线图.png'
            plt.savefig(os.path.join(output_dir, file_name), dpi=120)
            plt.close(fig)
            charts_generated += 1
            
    print(f"  -> [{plot_date}] 共成功生成 {charts_generated} 张全流向诊断图。")

def generate_summary_line_charts(df_report, output_dir, inter_config, target_main_dir='南向'):
    df = df_report.copy()
    df = df[df['进口道方向'] == target_main_dir].copy()
    if df.empty:
        print(f"⚠️ 汇总图生成跳过：汇总表中没有方向为【{target_main_dir}】的数据。")
        return

    name_to_seq = {v['name']: v['seq'] for k, v in inter_config.items()}
    df['seq'] = df['路口名称'].map(name_to_seq).fillna(99)
    periods = df['时段'].dropna().unique()
    turns = df['转向'].dropna().unique()

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'干线走廊 ({target_main_dir}) 瓶颈诊断空间剖面图', fontsize=20, fontweight='bold', y=0.96)

    metrics = [
        ('90%分位延误', '90%分位延误指数', axes[0]),
        ('停车次数', '平均停车次数 (次)', axes[1]),
        ('不停车速度通过速度', '不停车速度 (m/s)', axes[2])
    ]

    styles = {
        '早高峰-直行': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-'},   
        '早高峰-左转': {'color': '#1f77b4', 'marker': 's', 'linestyle': '--'},  
        '晚高峰-直行': {'color': '#ff7f0e', 'marker': 'o', 'linestyle': '-'},   
        '晚高峰-左转': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--'}   
    }

    x_labels = sorted(df['路口名称'].unique(), key=lambda x: name_to_seq.get(x, 99))

    for metric_col, ylabel, ax in metrics:
        if metric_col not in df.columns:
            continue

        for period in periods:
            for turn in turns:
                group = df[(df['时段'] == period) & (df['转向'] == turn)]
                if group.empty:
                    continue

                group_dict = dict(zip(group['路口名称'], group[metric_col]))
                y_values = [group_dict.get(name, np.nan) for name in x_labels]

                style_key = f"{period}-{turn}"
                style = styles.get(style_key, {'color': 'gray', 'marker': 'x', 'linestyle': '-'})

                ax.plot(x_labels, y_values, label=style_key,
                        color=style['color'], marker=style['marker'], linestyle=style['linestyle'],
                        linewidth=2.5, markersize=8, alpha=0.8)

        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        if metric_col == '90%分位延误':
            ax.axhline(y=2.0, color='red', linestyle=':', linewidth=1.5, label='高压警戒线 (DI=2.0)')

        ax.legend(loc='upper right', framealpha=0.9)

    plt.xticks(range(len(x_labels)), x_labels, rotation=15, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(output_dir, f'干线瓶颈诊断空间剖面图_{target_main_dir}.png')
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"✅ 走廊剖面折线图已生成: {save_path}")

def generate_turning_all_metrics_charts(df_dir_agg, intersection_name, output_dir, interval):
    try:
        dates = sorted(df_dir_agg['date'].unique())
        metrics = [
            ('stop_time', '停车时间(s)', '平均灯前停车时间'),
            ('stop_times', '停车次数(次)', '平均停车次数'),
            ('delay_index', '延误指数', '延误指数'),
            ('los', '延误指数评级', '延误指数评级(LOS)')
        ]
        
        los_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'nan': 0}
        directions = ['E', 'S', 'W', 'N']
        turn_names = {1: '左转', 2: '直行'}
        line_styles = {1: '--', 2: '-'}
        markers = {1: 'o', 2: 's'}
        
        for col, ylabel, metric_cn in metrics:
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
                        
                        y_data = date_subset[col].map(los_mapping).fillna(0) if col == 'los' else date_subset[col]
                        
                        ax.plot(norm_time, y_data,
                                color=dir_color, linestyle=line_styles[t],
                                marker=markers[t], markersize=3, alpha=0.8,
                                label=f"{date_prefix}{turn_names[t]}")
                        has_any_data = True
                        
                ax.set_title(f'{CARDINAL_HANZI.get(d, d)}进口道', fontsize=14, color=dir_color, fontweight='bold')
                ax.set_ylabel(ylabel, fontsize=12)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, linestyle='--', alpha=0.7)
                
                if col == 'los':
                    ax.set_yticks([0, 1, 2, 3, 4, 5])
                    ax.set_yticklabels(['A', 'B', 'C', 'D', 'E', 'F'])
                
                ax2 = None
                if SHOW_OVERLAY_ON_COMBINED_CHART:
                    ax2 = add_custom_periods_overlay(ax, CUSTOM_PERIODS, right_ylabel='')
                
                handles, labels = ax.get_legend_handles_labels()
                if ax2:
                    h2, l2 = ax2.get_legend_handles_labels()
                    handles.extend(h2)
                    labels.extend(l2)
                    
                if handles:
                    ax.legend(handles, labels, fontsize=10, loc='best')
                    
            if has_any_data:
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                save_path = os.path.join(output_dir, f'{metric_cn}_{interval}min.png')
                plt.savefig(save_path, dpi=150)
            plt.close(fig)

    except Exception as e:
        print(f"微观转向指标图谱生成出错: {e}")

def get_date_ranges():
    time_periods = [
        (True, '2026-03-25 00:00:00', '2026-03-25 23:59:59'),  
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

# ================= 核心分析模块 (强制 5 分钟切片驱动版) =================
def analyze_multiple_files(file_paths, date_ranges, inter_config):
    if not file_paths:
        print("❌ 当前文件夹下未找到任何包含 'index_view' 的文件。")
        return

    global_bottleneck_report = []
    global_granular_data = [] 

    for file_path in file_paths:
        print(f"\n📂 正在读取文件: {os.path.basename(file_path)}")
        try: df_dir = pd.read_csv(file_path, encoding='GBK')
        except: df_dir = pd.read_csv(file_path, encoding='utf-8')
        
        if df_dir.empty or 'inter_id' not in df_dir.columns:
            continue
            
        current_inter_id = str(df_dir['inter_id'].iloc[0]).strip().lower()
        if current_inter_id not in inter_config:
            continue
            
        config = inter_config[current_inter_id]
        inter_name = config.get('name', f"未知路口_{current_inter_id}")
        offset = config.get('offset', 0.0)
        seq = config.get('seq', 99)
        
        df_dir['create_time'] = pd.to_datetime(df_dir['create_time'])

        if date_ranges and date_ranges[0] is not None:
            mask_dir = pd.Series(False, index=df_dir.index)
            for start_time, end_time in date_ranges:
                mask_dir |= (df_dir['create_time'] >= start_time) & (df_dir['create_time'] <= end_time)
            df_dir = df_dir[mask_dir].copy()

        df_dir = df_dir[df_dir['create_time'].dt.minute % 5 == 0].copy()
        df_dir = df_dir.sort_values('create_time')

        if df_dir.empty:
            print(f"⚠️ 路口 {inter_name} 在指定日期范围内无有效抽样数据，跳过。")
            continue

        print("  -> 正在清洗流向并提取基础指标...")
        df_dir = enrich_direction_features(df_dir, offset_degree=offset)
        
        if 'turn_dir_no' in df_dir.columns:
            df_dir = df_dir[df_dir['turn_dir_no'].isin([1, 2])].copy()

        df_dir['路口名称'] = inter_name
        df_dir['进口道方向'] = df_dir['main_direction'].map(CARDINAL_HANZI)
        df_dir['转向'] = df_dir['turn_dir_no'].map({1: '左转', 2: '直行'})
        df_dir['延误指数'] = df_dir.get('delay_index', 0.0)

        pass_flow_cols = [c for c in df_dir.columns if 'pass_flow' in c.lower() or '分均流量' in c]
        if pass_flow_cols:
            p_col = pass_flow_cols[0]
            df_dir['pass_flow'] = df_dir[p_col].astype(str).str.replace(',', '', regex=False)
            df_dir['pass_flow'] = df_dir['pass_flow'].str.extract(r'(\d+\.?\d*)')[0]
            df_dir['pass_flow'] = pd.to_numeric(df_dir['pass_flow'], errors='coerce').fillna(0)
        else:
            df_dir['pass_flow'] = 0

        print("  -> 正在基于 5 分钟代表值计算 [历史最大流量] 与 [瞬时流量占比]...")
        hist_max_df = df_dir.groupby(['进口道方向', '转向'])['pass_flow'].max().reset_index()
        hist_max_df.rename(columns={'pass_flow': '历史最大分均流量'}, inplace=True)
        df_dir = pd.merge(df_dir, hist_max_df, on=['进口道方向', '转向'], how='left')

        total_flow_5min = df_dir.groupby('create_time')['pass_flow'].transform('sum')
        df_dir['流量占比(%)'] = np.where(total_flow_5min > 0, (df_dir['pass_flow'] / total_flow_5min) * 100, 0.0)

        # 👇 [修复核心]：将 queue_len_avg 和 queue_len_max 纳入提取范围 👇
        df_export = df_dir.dropna(subset=['进口道方向', '转向']).copy()
        export_cols = [
            '路口名称', '进口道方向', '转向', 'create_time', '延误指数', 
            'pass_flow', '历史最大分均流量', '流量占比(%)',
            'queue_len_avg', 'queue_len_max'
        ]
        
        for c in export_cols:
            if c not in df_export.columns: df_export[c] = None
        
        # 👇 仅在这里导出明细表时，统一将列名更改为中文 👇
        df_final_export = df_export[export_cols].rename(columns={
            'pass_flow': '5分钟车流量',
            '历史最大分均流量': '历史最大5分钟流量',
            'queue_len_avg': '平均排队长度',
            'queue_len_max': '最大排队长度'
        })
        global_granular_data.append(df_final_export)

        print("  -> 正在计算高峰期流向级瓶颈聚合指标...")
        report_slice = run_bottleneck_diagnosis_on_slice(df_dir, inter_name)
        global_bottleneck_report.extend(report_slice)

        folder_name = f'Output_{seq:02d}_{inter_name}'
        output_dir = os.path.join(os.path.dirname(file_path), folder_name)
        
        if date_ranges and date_ranges[0] is not None:
            target_date_str = date_ranges[0][0].strftime('%Y-%m-%d')
            generate_all_turning_status_charts(df_dir, inter_name, target_date_str, output_dir)
            
            intervals = [30] 
            for interval in intervals:
                analysis_output_dir = os.path.join(output_dir, f'analysis_{interval}min')
                os.makedirs(analysis_output_dir, exist_ok=True)
                
                df_dir_src = df_dir.copy() 
                grouper_dir = [pd.Grouper(key='create_time', freq=f'{interval}min'), 'main_direction', 'turn_dir_no']
                
                target_cols = ['stop_time', 'stop_times', 'delay_index']
                available_cols = [c for c in target_cols if c in df_dir_src.columns]
                
                if available_cols:
                    grouped_sampled = df_dir_src[available_cols + ['create_time', 'main_direction', 'turn_dir_no']].groupby(grouper_dir)
                    df_dir_agg = grouped_sampled.mean().reset_index()
                    
                    if 'los' in df_dir_src.columns:
                        def get_mode(series):
                            mode_series = series.mode()
                            return mode_series.iloc[0] if len(mode_series) > 0 else (series.iloc[0] if len(series) > 0 else None)
                        los_values = df_dir_src.groupby(grouper_dir)['los'].apply(get_mode).reset_index()
                        df_dir_agg = pd.merge(df_dir_agg, los_values, on=['create_time', 'main_direction', 'turn_dir_no'], how='left')
                    
                    df_dir_agg['date'] = df_dir_agg['create_time'].dt.date
                    print(f"  -> [{interval}分钟] 微观指标阵列图已生成。")
                    generate_turning_all_metrics_charts(df_dir_agg, inter_name, analysis_output_dir, interval)

    base_output_dir = os.path.dirname(file_paths[0]) if file_paths else ""

    if global_bottleneck_report:
        df_final_report = pd.DataFrame(global_bottleneck_report)
        print("\n================ 🚦 干线流向级瓶颈诊断汇总表 ================")
        print(df_final_report.to_string(index=False, justify='center'))
        
        report_save_path = os.path.join(base_output_dir, "干线流向级瓶颈诊断汇总表.csv")
        try:
            df_final_report.to_csv(report_save_path, index=False, encoding='GBK')
        except Exception: pass
        
     
        # target_hanzi = CARDINAL_HANZI.get(TARGET_DIRECTION, TARGET_DIRECTION)
        # generate_summary_line_charts(df_report=df_final_report, output_dir=base_output_dir, inter_config=inter_config, target_main_dir=target_hanzi)

    if global_granular_data:
        df_all_granular = pd.concat(global_granular_data, ignore_index=True)
        granular_save_path = os.path.join(base_output_dir, "干线全流向_延误与流量占比明细表.csv")
        try:
            df_all_granular.to_csv(granular_save_path, index=False, encoding='GBK')
            print(f"\n✅ [供推演引擎使用] 带精确5分钟切片与极大值的明细表已保存至: {granular_save_path}")
        except PermissionError:
            print(f"\n⚠️ 警告: 无法保存明细表CSV！请先关闭正在 Excel 中打开的文件。")

    print("\n✅ 所有配置路口的分析任务均已执行完毕！")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    csv_files = [os.path.join(script_dir, f) for f in os.listdir(script_dir) if "index_view" in f and f.endswith('.csv')]
    
    print(f"=================================")
    print(f"共发现 {len(csv_files)} 个 index_view 数据文件")
    print(f"=================================\n")
    
    date_ranges = get_date_ranges()
    
    analyze_multiple_files(
        file_paths=csv_files, 
        date_ranges=date_ranges, 
        inter_config=INTER_CONFIG
    )