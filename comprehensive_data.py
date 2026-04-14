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
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  

# ================= 核心业务与拓扑配置区 =================
# 🎯 干线分析方向
TARGET_DIRECTION = 'S' 

# 🎯 ID与名称映射字典 (全部转为小写标准格式)
INTER_ID_NAME_MAP = {
    '6caaa6f15b735a': '光明大道与光辉大道',
    '6caa3ec15b566e': '光明大道与光明大街',
    '6caa06b15b50eb': '光明大道与光安路',
    '6ca93d215b3ee3': '光明大道与华夏路',
    '6ca887115b2e22': '光明大道与华裕路'
}

# 🎯 干线物理拓扑
CORRIDOR_LINKS = [
    {"name": "光明大道与光辉大道", "seq": 1, "dist_to_next": 857},
    {"name": "光明大道与光明大街", "seq": 2, "dist_to_next": 150},
    {"name": "光明大道与光安路",   "seq": 3, "dist_to_next": 600},
    {"name": "光明大道与华夏路",   "seq": 4, "dist_to_next": 555}, 
    {"name": "光明大道与华裕路",   "seq": 5, "dist_to_next": 0}
]

# 🎯 各路口偏转角自定义接口 (单位: 度)
# 在这里直观地为您想要修正的路口填入角度，未填写的默认按 0.0 度处理
INTER_OFFSET_MAP = {
    '光明大道与光辉大道': 0.0,
    '光明大道与光明大街': -32.57,
    '光明大道与光安路': -33.69,
    '光明大道与华夏路': -63.5,
    '光明大道与华裕路': -63.0,
    
    
    
    
}

# 【底层自动整合】将以上配置自动打包，无需在此修改
INTER_CONFIG = {}
for i_id, i_name in INTER_ID_NAME_MAP.items():
    link_info = next((item for item in CORRIDOR_LINKS if item["name"] == i_name), None)
    INTER_CONFIG[i_id] = {
        "name": i_name,
        "offset": INTER_OFFSET_MAP.get(i_name, 0.0),  # 自动从上面的角度字典读取对应偏转角
        "seq": link_info["seq"] if link_info else 99,
        "dist_to_next": link_info["dist_to_next"] if link_info else 0
    }

# ================= 周期时间等其他配置 =================
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
    return ax2

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

# ================= 微观指标独立 2x2 阵列图 =================
def generate_turning_all_metrics_charts(df_dir_agg, intersection_name, output_dir, interval):
    try:
        dates = sorted(df_dir_agg['date'].unique())
        metrics = [
            ('stop_time', '停车时间(s)', '平均灯前停车时间'),
            ('stop_times', '停车次数(次)', '平均停车次数'),
            ('delay_index', '延误指数', '延误指数'),
            ('los', '评级阶梯', '延误指数评级(LOS)')
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
                
                # 严格区分左转(1)与直行(2)
                for t in [1, 2]:
                    subset = df_dir_agg[(df_dir_agg['main_direction'] == d) & (df_dir_agg['turn_dir_no'] == t)]
                    if subset.empty: continue
                    
                    for date in dates:
                        date_subset = subset[subset['date'] == date]
                        if date_subset.empty or date_subset[col].isna().all(): continue
                        
                        norm_time = pd.to_datetime('2026-01-01 ' + date_subset['create_time'].dt.time.astype(str))
                        date_prefix = f"{date} " if len(dates) > 1 else ""
                        
                        if col == 'los':
                            y_data = date_subset[col].map(los_mapping).fillna(0)
                        else:
                            y_data = date_subset[col]
                        
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
                
                # 🎯 替换后 (图例合并逻辑)
                ax2 = None
                if SHOW_OVERLAY_ON_COMBINED_CHART:
                    ax2 = add_custom_periods_overlay(ax, CUSTOM_PERIODS, right_ylabel='')
                
                # 分别提取主轴和副轴的线条标签，打包在一起
                handles, labels = ax.get_legend_handles_labels()
                if ax2:
                    h2, l2 = ax2.get_legend_handles_labels()
                    handles.extend(h2)
                    labels.extend(l2)
                    
                # 只生成唯一的一个图例盒子
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

# ================= 核心分析模块 (多文件独立读取 & 抽样防平滑) =================
def analyze_multiple_files(file_paths, date_ranges, inter_config):
    if not file_paths:
        print("❌ 当前文件夹下未找到任何包含 'index_view' 的文件。")
        return

    for file_path in file_paths:
        print(f"\n📂 正在读取文件: {os.path.basename(file_path)}")
        try: df_dir = pd.read_csv(file_path, encoding='GBK')
        except: df_dir = pd.read_csv(file_path, encoding='utf-8')
        
        if df_dir.empty or 'inter_id' not in df_dir.columns:
            print("⚠️ 数据为空或缺失 'inter_id' 字段，跳过。")
            continue
            
        current_inter_id = str(df_dir['inter_id'].iloc[0]).strip().lower()
        
        if current_inter_id not in inter_config:
            print(f"⚠️ 跳过: 配置文件中未登记该路口 ID ({current_inter_id})。")
            continue
            
        config = inter_config[current_inter_id]
        inter_name = config.get('name', f"未知路口_{current_inter_id}")
        offset = config.get('offset', 0.0)
        seq = config.get('seq', 99)
        dist = config.get('dist_to_next', 0)
        
        print(f"🚀 开始处理: [序列 {seq}] {inter_name} | ID: {current_inter_id} | 距离下一路口: {dist}m")
        
        df_dir['create_time'] = pd.to_datetime(df_dir['create_time'])
        df_dir = df_dir.sort_values('create_time')

        # 执行流向清洗
        print("  -> 正在计算坐标偏转角与轨迹流向...")
        df_dir = enrich_direction_features(df_dir, offset_degree=offset)
        
        if 'turn_dir_no' in df_dir.columns:
            df_dir = df_dir[df_dir['turn_dir_no'].isin([1, 2])].copy()

        # 时间区间过滤与聚合出图
        if date_ranges and date_ranges[0] is not None:
            mask_dir = pd.Series(False, index=df_dir.index)
            for start_time, end_time in date_ranges:
                mask_dir |= (df_dir['create_time'] >= start_time) & (df_dir['create_time'] <= end_time)
            df_dir = df_dir[mask_dir]
            
            # 🎯 当前测试：仅运行 30 分钟聚合
            intervals = [30] 
            
            for interval in intervals:
                folder_name = f'Output_{seq:02d}_{inter_name}'
                output_dir = os.path.join(os.path.dirname(file_path), folder_name, f'analysis_{interval}min')
                os.makedirs(output_dir, exist_ok=True)
                
                df_dir_src = df_dir.copy() 
                grouper_dir = [pd.Grouper(key='create_time', freq=f'{interval}min'), 'main_direction', 'turn_dir_no']
                
                target_cols = ['stop_time', 'stop_times', 'delay_index']
                available_cols = [c for c in target_cols if c in df_dir_src.columns]
                
                if available_cols:
                    if interval <= 5:
                        # 5分钟以内：直接取最后时刻代表该区间
                        grouped = df_dir_src[available_cols + ['create_time', 'main_direction', 'turn_dir_no']].groupby(grouper_dir)
                        df_dir_agg = grouped.last().reset_index()
                        
                        if 'los' in df_dir_src.columns:
                            los_values = df_dir_src.groupby(grouper_dir)['los'].last().reset_index()
                            df_dir_agg = pd.merge(df_dir_agg, los_values, on=['create_time', 'main_direction', 'turn_dir_no'], how='left')
                    else:
                        # 🎯 超过5分钟(如30min): 消除二次平滑！
                        # 严格只抽取独立不重叠的 5 分钟快照 (例如 00, 05, 10, 15 分钟时刻)
                        df_sampled = df_dir_src[df_dir_src['create_time'].dt.minute % 5 == 0].copy()
                        
                        # 基于抽取后的离散数据进行均值计算
                        grouped_sampled = df_sampled[available_cols + ['create_time', 'main_direction', 'turn_dir_no']].groupby(grouper_dir)
                        df_dir_agg = grouped_sampled.mean().reset_index()
                        
                        # LOS 评级求众数
                        if 'los' in df_sampled.columns:
                            def get_mode(series):
                                mode_series = series.mode()
                                return mode_series.iloc[0] if len(mode_series) > 0 else (series.iloc[0] if len(series) > 0 else None)
                            los_values = df_sampled.groupby(grouper_dir)['los'].apply(get_mode).reset_index()
                            df_dir_agg = pd.merge(df_dir_agg, los_values, on=['create_time', 'main_direction', 'turn_dir_no'], how='left')
                    
                    df_dir_agg['date'] = df_dir_agg['create_time'].dt.date
                    print(f"  -> [{interval}分钟] 图表已生成。")
                    generate_turning_all_metrics_charts(df_dir_agg, inter_name, output_dir, interval)

    print("\n✅ 所有配置路口的分析任务均已执行完毕！")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 扫描文件夹下所有包含 "index_view" 的 csv 文件
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