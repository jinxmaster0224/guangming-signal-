import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates
import numpy as np
import re
import math
import gc

# ================= 1. 核心配置与日期选择区 =================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False  

CARDINAL_HANZI = {"E": "东向", "W": "西向", "S": "南向", "N": "北向"}
_WKT_COORD_PATTERN = re.compile(r'(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)')

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

def get_date_ranges():
    time_periods = [
        (True,  '2026-03-10 00:00:00', '2026-03-10 23:59:59'),  # 示例查询日期
        (False, '2026-04-01 00:00:00', '2026-04-01 23:59:59'),  
        (False, '2026-04-02 00:00:00', '2026-04-05 23:59:59'),  
    ]
    
    date_ranges = []
    for enable, start_date, end_date in time_periods:
        if enable:
            try:
                date_ranges.append((pd.Timestamp(start_date), pd.Timestamp(end_date)))
                print(f"[配置启用] 分析日期: {start_date[:10]} 至 {end_date[:10]}")
            except Exception as e:
                print(f"日期格式错误: {e}")
                
    if not date_ranges:
        print("[配置启用] 未选择日期，将自动扫描分析全表数据")
        date_ranges.append(None)
        
    return date_ranges

# ================= 2. 空间与流向计算模块 =================
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
    """添加红色时间分割线与外扩的第三Y轴周期阶梯图"""
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

def generate_directional_bar_charts(df_60, intersection_name, output_dir):
    """按东南西北四个方向，生成1小时聚合的三柱图（直行、左转、延误并排）"""
    print("\n[进度] 正在生成各方向的 流量-延误 柱状图...")
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
            
            fig, ax1 = plt.subplots(figsize=(15, 7))
            fig.subplots_adjust(right=0.85) 
            
            # 三根柱子，每根宽度 13 分钟，加上间隙刚好填满 1 小时 (60分钟)
            bar_width = pd.Timedelta(minutes=13)
            
            # --- 💡主轴（左）：分均转向车流量 (并排双柱) ---
            # 柱1：直行流量 (向左偏移 15 分钟)
            bars_straight = ax1.bar(dir_df['norm_time'] - pd.Timedelta(minutes=15), dir_df['flow_straight'], 
                            width=bar_width, color='#4c72b0', alpha=0.85, label='直行流量 (辆/分)')
            
            # 柱2：左转流量 (居中)
            bars_left = ax1.bar(dir_df['norm_time'], dir_df['flow_left'], 
                            width=bar_width, color='#55a868', alpha=0.85, label='左转流量 (辆/分)')
            
            ax1.set_ylabel('分均车流量 (辆/分)', color='black', fontsize=12, fontweight='bold')
            ax1.tick_params(axis='y', labelcolor='black')
            ax1.set_xlabel('时间', fontsize=12)
            
            # --- 次轴（右）：延误指数 ---
            ax2 = ax1.twinx()
            # 柱3：延误指数 (向右偏移 15 分钟)
            bars_delay = ax2.bar(dir_df['norm_time'] + pd.Timedelta(minutes=15), dir_df['delay_index'], 
                            width=bar_width, color='#dd8452', alpha=0.85, label='交叉口延误指数')
            ax2.set_ylabel('延误指数', color='#dd8452', fontsize=12, fontweight='bold')
            ax2.tick_params(axis='y', labelcolor='#dd8452')
            
            # --- 周期外侧黑轴 ---
            add_custom_periods_overlay(ax1, CUSTOM_PERIODS)
            
            # 图例与美化合并显示
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=11)

            ax1.set_xticks(pd.date_range('2026-01-01 00:00', '2026-01-01 23:59', freq='1H'))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.setp(ax1.get_xticklabels(), rotation=45)
            ax1.grid(True, axis='y', linestyle=':', alpha=0.5)

            plt.title(f'{intersection_name} - {dir_cn}进口道 转向分均流量与延误分析 ({date})', fontsize=16, fontweight='bold', pad=15)
            
            output_path = os.path.join(output_dir, f'{date}_{dir_cn}向_三柱图分析.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    ✓ 已生成: {os.path.basename(output_path)}")

# ================= 4. 数据摄入与预处理管线 =================
def load_csv_safe(csv_path):
    try: return pd.read_csv(csv_path, encoding='GBK')
    except:
        try: return pd.read_csv(csv_path, encoding='utf-8')
        except: return pd.read_csv(csv_path, encoding='latin1')

def analyze_traffic_data(info_csv_path, dir_csv_path, date_ranges):
    print(f"\n[1] 正在加载路口主表 (用于读取 延误指数): {os.path.basename(info_csv_path)}")
    df_info = load_csv_safe(info_csv_path)
    df_info['create_time'] = pd.to_datetime(df_info['create_time'])
    intersection_name = df_info['inter_name'].iloc[0] if 'inter_name' in df_info.columns and not df_info.empty else '未知路口'
    
    # 提取 info 表的延误指数 (确保干净)
    df_delay = df_info[['create_time', 'delay_index']].groupby('create_time').mean().reset_index()

    print(f"[2] 正在加载方向附表 (用于读取 转向流量): {os.path.basename(dir_csv_path)}")
    df_dir = load_csv_safe(dir_csv_path)
    df_dir['create_time'] = pd.to_datetime(df_dir['create_time'])
    
    # 【彻底修复 KeyError 的核心】：丢弃附表自带的 delay_index，强行使用主表的
    if 'delay_index' in df_dir.columns:
        df_dir = df_dir.drop(columns=['delay_index'])

    print("  -> 正在进行空间打标，分离左转(1)与直行(2)...")
    df_dir = enrich_direction_features(df_dir)
    if 'turn_dir_no' in df_dir.columns:
        df_dir = df_dir[df_dir['turn_dir_no'].isin([1, 2])].dropna(subset=['main_direction'])
    
    print("  -> 正在融合主表延误指数与附表转向流量...")
    df_merged = pd.merge(df_dir, df_delay, on='create_time', how='left')

    # 日期过滤
    if date_ranges and date_ranges[0] is not None:
        mask = False
        for start_time, end_time in date_ranges:
            mask |= (df_merged['create_time'] >= start_time) & (df_merged['create_time'] <= end_time)
        df_merged = df_merged[mask]
        if df_merged.empty:
            print("\n⚠️ 警告：配置的日期范围内没有找到任何数据！")
            return

    print("  -> 正在应用 5 分钟滑动平均算法处理各转向流量与延误...")
    df_merged = df_merged.set_index('create_time').sort_index()
    # 滑动平均
    df_merged['pass_flow'] = df_merged.groupby(['main_direction', 'turn_dir_no'])['pass_flow'].transform(lambda x: x.rolling('5min', min_periods=1).mean())
    df_merged['delay_index'] = df_merged.groupby(['main_direction', 'turn_dir_no'])['delay_index'].transform(lambda x: x.rolling('5min', min_periods=1).mean())
    df_merged = df_merged.reset_index()

    # 第一级重构：将左转与直行拆解为两列独立字段
    print("  -> 正在重构流量结构，区分左转与直行...")
    df_flow = df_merged.pivot_table(
        index=['create_time', 'main_direction'],
        columns='turn_dir_no',
        values='pass_flow',
        aggfunc='mean'
    ).reset_index()
    
    # 1为左转，2为直行
    df_flow = df_flow.rename(columns={1: 'flow_left', 2: 'flow_straight'})
    if 'flow_left' not in df_flow.columns: df_flow['flow_left'] = 0
    if 'flow_straight' not in df_flow.columns: df_flow['flow_straight'] = 0
    df_flow[['flow_left', 'flow_straight']] = df_flow[['flow_left', 'flow_straight']].fillna(0)

    # 合并延误指数
    df_delay_min = df_merged.groupby(['create_time', 'main_direction'])['delay_index'].mean().reset_index()
    df_min = pd.merge(df_flow, df_delay_min, on=['create_time', 'main_direction'])

    # 第二级聚合：按1小时计算小时内的平均值
    print("  -> 正在执行最终的 60 分钟降维聚合...")
    df_60 = df_min.groupby([pd.Grouper(key='create_time', freq='60min'), 'main_direction']).mean().reset_index()
    df_60['date'] = df_60['create_time'].dt.date
    
    output_dir = os.path.join(os.path.dirname(info_csv_path), '分析结果_各向三柱图')
    generate_directional_bar_charts(df_60, intersection_name, output_dir)
    print("\n🎉 分析管线执行完毕！")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = [f for f in os.listdir(script_dir) if f.endswith('.csv')]
    
    info_csv = next((os.path.join(script_dir, f) for f in csv_files if "info_view" in f), None)
    dir_csv = next((os.path.join(script_dir, f) for f in csv_files if "index_view" in f), None)
        
    if info_csv and dir_csv:
        date_ranges = get_date_ranges()
        analyze_traffic_data(info_csv, dir_csv, date_ranges)
    else:
        print(f"❌ 错误：请确保脚本目录下同时包含 info_view (主表) 和 index_view (附表) CSV 文件！")