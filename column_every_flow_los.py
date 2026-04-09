import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates
import numpy as np
import re
import math
import gc

# ================= 1. 日期选择 =================
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  

CARDINAL_HANZI = {"E": "东向", "W": "西向", "S": "南向", "N": "北向"}
_WKT_COORD_PATTERN = re.compile(r'(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)')

# 🎯 ID与名称映射字典
INTER_ID_NAME_MAP = {
    '6ca93d215b3ee3': '光明大道与华夏路',
    '6caa06b15b50eb': '光明大道与光安路',
    '6caa3ec15b566e': '光明大道与光明大街',
    '6caaa6f15b735a': '光明大道与光辉大道',
    '6ca887115b2e22': '光明大道与华裕路'
}

# 🚥 自定义周期时间配置
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

# 📊 从 index_view 提取的指标配置
METRICS_CONFIG = {
    'no_stop_pass_speed': ('不停车通过速度', '不停车速度 (m/s)'),
    'pass_speed': ('通过速度', '通过速度 (m/s)'),
    'queue_len_max': ('最大排队长度', '最大排队 (米)'),
    'queue_len_avg': ('平均排队长度', '平均排队 (米)'),
    'stop_time': ('停车时间', '停车时间 (秒)'),
    'stop_times': ('停车次数', '停车次数 (次)')
}

def get_date_ranges():
    time_periods = [
        (True,  '2026-03-10 00:00:00', '2026-03-10 23:59:59'),  
        (False, '2026-04-01 00:00:00', '2026-04-01 23:59:59'),  
        (False, '2026-04-01 00:00:00', '2026-04-01 23:59:59'),
        (False, '2026-04-01 00:00:00', '2026-04-01 23:59:59'),
        (False, '2026-04-01 00:00:00', '2026-04-01 23:59:59'),
    ]
    date_ranges = []
    for enable, start_date, end_date in time_periods:
        if enable:
            try:
                date_ranges.append((pd.Timestamp(start_date), pd.Timestamp(end_date)))
                print(f"分析日期: {start_date[:10]} 至 {end_date[:10]}")
            except Exception:
                pass
                
    if not date_ranges:
        print("未选择日期")
        date_ranges.append(None)
    return date_ranges

# ================= 2. 空间与流向计算 =================
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

def enrich_direction_features(df: pd.DataFrame) -> pd.DataFrame:
    mask = df['lng_lat_seq'].notna() & (df['lng_lat_seq'] != '')
    df.loc[mask, '_raw_direction'] = df.loc[mask, 'lng_lat_seq'].apply(lambda x: extract_direction_from_coords(x))
    
    if "frid" in df.columns:
        valid_dirs = df[['frid', '_raw_direction']].dropna()
        if not valid_dirs.empty:
            counts = valid_dirs.groupby(['frid', '_raw_direction']).size().reset_index(name='count')
            best_dirs = counts.sort_values(by=['frid', 'count'], ascending=[True, False]).drop_duplicates('frid').set_index('frid')['_raw_direction']
            df['main_direction'] = df['frid'].map(best_dirs)
        else: df['main_direction'] = df['_raw_direction']
    else: df['main_direction'] = df['_raw_direction']
    
    if '_raw_direction' in df.columns: df.drop(columns=['_raw_direction'], inplace=True)
    gc.collect()
    return df

# ================= 3. 绘图核心逻辑 =================
def add_custom_periods_overlay(ax, periods_config, base_date='2026-01-01'):
    """添加时段划分红色虚线与信号周期黑色虚线"""
    if not periods_config: return
    times, values = [], []
    for period in periods_config:
        start_time = pd.to_datetime(f"{base_date} {period['start']}")
        times.append(start_time)
        values.append(period['value'])
        ax.axvline(x=start_time, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
        
    last_end_time = pd.to_datetime(f"{base_date} {periods_config[-1]['end']}")
    times.append(last_end_time)
    values.append(periods_config[-1]['value']) 
    ax.axvline(x=last_end_time, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
    
    ax3 = ax.twinx()
    ax3.spines['right'].set_position(('outward', 60))  
    ax3.step(times, values, where='post', color='black', linestyle='--', linewidth=2.5, label='周期时间')
    ax3.set_ylabel('周期时间 (秒)', color='black', fontweight='bold')
    ax3.tick_params(axis='y', labelcolor='black')
    ax3.legend(loc='upper right')

def plot_flow_3bar(dir_df, date, dir_cn, intersection_name, output_dir):
    """分均转向流量三柱图"""
    fig, ax1 = plt.subplots(figsize=(15, 7))
    fig.subplots_adjust(right=0.85) 
    bar_width = pd.Timedelta(minutes=13)
    
    ax1.bar(dir_df['norm_time'] - pd.Timedelta(minutes=15), dir_df['flow_straight'], 
            width=bar_width, color='#4c72b0', alpha=0.85, label='直行流量 (辆/分)')
    ax1.bar(dir_df['norm_time'], dir_df['flow_left'], 
            width=bar_width, color='#55a868', alpha=0.85, label='左转流量 (辆/分)')
    
    ax1.set_ylabel('分均车流量 (辆/分)', color='black', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xlabel('时间 ', fontsize=12)
    
    ax2 = ax1.twinx()
    if 'delay_index' in dir_df.columns:
        ax2.bar(dir_df['norm_time'] + pd.Timedelta(minutes=15), dir_df['delay_index'], 
                width=bar_width, color='#dd8452', alpha=0.85, label='交叉口延误指数')
        ax2.set_ylabel('延误指数', color='#dd8452', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='#dd8452')
    
    add_custom_periods_overlay(ax1, CUSTOM_PERIODS)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=11)

    ax1.set_xticks(pd.date_range('2026-01-01 00:00', '2026-01-01 23:59', freq='1H'))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax1.get_xticklabels(), rotation=45)
    ax1.grid(True, axis='y', linestyle=':', alpha=0.5)

    plt.title(f'{intersection_name} - {dir_cn}进口道 分均流量 ({date})', fontsize=16, fontweight='bold', pad=15)
    output_path = os.path.join(output_dir, f'01_分均流量_{dir_cn}向_{date}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_metric_2bar(dir_df, date, dir_cn, intersection_name, metric_col, metric_name, ylabel_str, output_dir, file_prefix):
    """其他指标双柱图"""
    if metric_col not in dir_df.columns or dir_df[metric_col].isna().all():
        return
        
    fig, ax1 = plt.subplots(figsize=(14, 7))
    fig.subplots_adjust(right=0.85) 
    bar_width = pd.Timedelta(minutes=18)
    
    ax1.bar(dir_df['norm_time'] - pd.Timedelta(minutes=9), dir_df[metric_col], 
            width=bar_width, color='#4c72b0', alpha=0.85, label=metric_name)
    ax1.set_ylabel(ylabel_str, color='#4c72b0', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#4c72b0')
    ax1.set_xlabel('时间', fontsize=12)
    
    ax2 = ax1.twinx()
    if 'delay_index' in dir_df.columns:
        ax2.bar(dir_df['norm_time'] + pd.Timedelta(minutes=9), dir_df['delay_index'], 
                width=bar_width, color='#dd8452', alpha=0.85, label='交叉口延误指数')
        ax2.set_ylabel('延误指数', color='#dd8452', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='#dd8452')
    
    add_custom_periods_overlay(ax1, CUSTOM_PERIODS)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=11)

    ax1.set_xticks(pd.date_range('2026-01-01 00:00', '2026-01-01 23:59', freq='1H'))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax1.get_xticklabels(), rotation=45)
    ax1.grid(True, axis='y', linestyle=':', alpha=0.5)

    plt.title(f'{intersection_name} - {dir_cn}进口道 {metric_name} ({date})', fontsize=16, fontweight='bold', pad=15)
    output_path = os.path.join(output_dir, f'{file_prefix}_{metric_name}_{dir_cn}向_{date}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

# ================= 4. 数据处理 =================
def load_csv_safe(csv_path):
    for enc in ['GBK', 'utf-8', 'latin1']:
        try: return pd.read_csv(csv_path, encoding=enc)
        except: pass
    return pd.DataFrame()

def analyze_traffic_data(dir_csv_path, date_ranges):
    print(f"正在加载表数据 (index_view): {os.path.basename(dir_csv_path)}")
    df_dir = load_csv_safe(dir_csv_path)
    df_dir['create_time'] = pd.to_datetime(df_dir['create_time'])

    # 将 inter_id 映射为中文名
    if 'inter_id' in df_dir.columns:
        df_dir['inter_id_clean'] = df_dir['inter_id'].astype(str).str.strip().str.lower()
        df_dir['inter_name'] = df_dir['inter_id_clean'].map(INTER_ID_NAME_MAP)
    
    intersection_name = df_dir['inter_name'].dropna().iloc[0] if 'inter_name' in df_dir.columns and not df_dir['inter_name'].dropna().empty else '未知路口'
    
    print("  -> 进行空间方向判断...")
    df_dir = enrich_direction_features(df_dir)
    
    # 仅保留直行和左转
    if 'turn_dir_no' in df_dir.columns:
        df_dir['turn_dir_no'] = pd.to_numeric(df_dir['turn_dir_no'], errors='coerce')
        df_dir = df_dir[df_dir['turn_dir_no'].isin([1, 2])].dropna(subset=['main_direction'])

    # 日期过滤
    if date_ranges and date_ranges[0] is not None:
        mask = False
        for start_time, end_time in date_ranges:
            mask |= (df_dir['create_time'] >= start_time) & (df_dir['create_time'] <= end_time)
        df_dir = df_dir[mask]
        
    if df_dir.empty:
        print("\n⚠️ 警告：过滤后没有可用数据！")
        return

    # 💡 核心逻辑分离点
    print("  -> [分均车流量(pass_flow)] 执行 5 分钟滑动平均")
    df_dir = df_dir.set_index('create_time').sort_index()
    if 'pass_flow' in df_dir.columns:
        df_dir['pass_flow'] = df_dir.groupby(['main_direction', 'turn_dir_no'])['pass_flow'].transform(lambda x: x.rolling('5min', min_periods=1).mean())
    df_dir = df_dir.reset_index()

    print("  -> 拆解车流量结构 (直行与左转)...")
    df_flow = df_dir.pivot_table(
        index=['create_time', 'main_direction'],
        columns='turn_dir_no',
        values='pass_flow',
        aggfunc='mean'
    ).reset_index()
    df_flow = df_flow.rename(columns={1: 'flow_left', 2: 'flow_straight'})
    for col in ['flow_left', 'flow_straight']:
        if col not in df_flow.columns: df_flow[col] = 0
    df_flow[['flow_left', 'flow_straight']] = df_flow[['flow_left', 'flow_straight']].fillna(0)

    print("  -> 正在提取其他指标")
    target_num_cols = ['delay_index'] + list(METRICS_CONFIG.keys())
    valid_num_cols = [c for c in target_num_cols if c in df_dir.columns]
    
    df_other = df_dir.groupby(['create_time', 'main_direction'])[valid_num_cols].mean().reset_index()
    
    # 将流量和其他指标合并
    df_min = pd.merge(df_flow, df_other, on=['create_time', 'main_direction'], how='outer')

    print("  -> 执行60 分钟聚合")
    df_60 = df_min.groupby([pd.Grouper(key='create_time', freq='60min'), 'main_direction']).mean().reset_index()
    df_60['date'] = df_60['create_time'].dt.date

    # ================= 5. 批量制图 =================
    print("准备分发图片")
    output_dir = os.path.join(os.path.dirname(dir_csv_path), '分析结果_单表全量指标图')
    os.makedirs(output_dir, exist_ok=True)
    
    dates = sorted(df_60['date'].unique())
    directions = ['E', 'S', 'W', 'N']

    for date in dates:
        date_df = df_60[df_60['date'] == date].copy()
        if date_df.empty: continue
            
        for direction in directions:
            dir_df = date_df[date_df['main_direction'] == direction].copy()
            if dir_df.empty: continue
                
            dir_cn = CARDINAL_HANZI.get(direction, direction)
            dir_df['norm_time'] = pd.to_datetime('2026-01-01 ' + dir_df['create_time'].dt.time.astype(str))
            
            # [1] 画分均车流量的 3 柱图
            plot_flow_3bar(dir_df, date, dir_cn, intersection_name, output_dir)
            
            # [2] 画剩余提取出的其他指标的 2 柱图
            prefix_idx = 2
            for metric_col, (metric_name, ylabel_str) in METRICS_CONFIG.items():
                if metric_col in dir_df.columns:
                    plot_metric_2bar(dir_df, date, dir_cn, intersection_name, metric_col, metric_name, ylabel_str, output_dir, f"{prefix_idx:02d}")
                    prefix_idx += 1
                    
        print(f"    ✓ 日期 [{date}] 四个方向的全部可用指标图已生成！")

    print(f"运行完成！所有图表已生成完毕！")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = [f for f in os.listdir(script_dir) if f.endswith('.csv')]
    
    # 只读取 index_view (方向表) 文件
    dir_csv = next((os.path.join(script_dir, f) for f in csv_files if "index_view" in f), None)
        
    if dir_csv:
        date_ranges = get_date_ranges()
        analyze_traffic_data(dir_csv, date_ranges)
    else:
        print(f"❌ 错误：请确保脚本目录下包含 index_view (方向附表) CSV 文件！")