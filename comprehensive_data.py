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
INTER_OFFSET_MAP = {
    '光明大道与光辉大道': 0.0,
    '光明大道与光明大街': -32.57,
    '光明大道与光安路': -33.69,
    '光明大道与华夏路': -63.5,
    '光明大道与华裕路': -63.0,
}

# 【底层自动整合】将以上配置自动打包
INTER_CONFIG = {}
for i_id, i_name in INTER_ID_NAME_MAP.items():
    link_info = next((item for item in CORRIDOR_LINKS if item["name"] == i_name), None)
    INTER_CONFIG[i_id] = {
        "name": i_name,
        "offset": INTER_OFFSET_MAP.get(i_name, 0.0),  
        "seq": link_info["seq"] if link_info else 99,
        "dist_to_next": link_info["dist_to_next"] if link_info else 0
    }

# ================= 周期时间等其他配置 =================
SHOW_OVERLAY_ON_COMBINED_CHART = True  

# 【新功能】：自定义高峰期诊断时段 (在此处修改早晚高峰时间)
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

# ================= 【新功能：流向级多维诊断】 =================
# ================= 【新功能：流向级多维诊断 (解耦抽样版)】 =================
def run_bottleneck_diagnosis_on_slice(df_dir, inter_name):
    """提取数据的 5 分钟抽样算延误，使用全量数据抓取溢出警报"""
    
    # 清洗列名，防止 CSV 表头带有隐形空格
    df_dir.columns = df_dir.columns.str.strip()
    
    # 【修复核心 1】：建立全量时间索引数据，专门用来算溢出率，绝不漏掉任何一分钟的警报！
    df_full = df_dir.sort_values('create_time').copy()
    df_full_indexed = df_full.set_index('create_time')
    
    # 【保留逻辑 2】：严格 5 分钟刻度抽样，专门用来算延误，防平滑
    df_sampled = df_dir[df_dir['create_time'].dt.minute % 5 == 0].copy()
    df_sampled = df_sampled.sort_values('create_time')
    df_sampled_indexed = df_sampled.set_index('create_time')
    
    results = []
    
    for period_name, (start, end) in PEAK_PERIODS.items():
        # 分别切片出“抽样高峰期”和“全量高峰期”
        df_peak_sampled = df_sampled_indexed.between_time(start, end)
        df_peak_full = df_full_indexed.between_time(start, end)
        
        for (main_dir, turn_dir), group_sampled in df_peak_sampled.groupby(['main_direction', 'turn_dir_no']):
            if group_sampled.empty: continue
            
            # 【修复核心 3】：从全量高峰数据中，提取出对应流向的 full_group 用来算溢出率
            group_full = df_peak_full[(df_peak_full['main_direction'] == main_dir) & 
                                      (df_peak_full['turn_dir_no'] == turn_dir)]
            
            # --- 以下使用【抽样数据】计算延误和体验类指标 ---
            val_90th_delay = group_sampled['delay_index'].quantile(0.90) if 'delay_index' in group_sampled.columns else None
            max_15min_delay = group_sampled['delay_index'].rolling(window=3, min_periods=3).mean().max() if 'delay_index' in group_sampled.columns else None
            
            avg_stop_times = group_sampled['stop_times'].mean() if 'stop_times' in group_sampled.columns else None
            avg_no_stop_speed = group_sampled['no_stop_pass_speed'].mean() if 'no_stop_pass_speed' in group_sampled.columns else None
            mode_los = group_sampled['los'].mode()[0] if ('los' in group_sampled.columns and not group_sampled['los'].mode().empty) else None
            
            # --- 以下使用【全量无损数据】计算溢出率 ---
            if 'idx_state' in group_full.columns:
                # 把状态列转为数字，计算大于等于 1（包含溢出 1 和严重 2）的比例
                numeric_states = pd.to_numeric(group_full['idx_state'], errors='coerce')
                spill_rate = (numeric_states >= 1).mean()
            else:
                spill_rate = 0.0
            
            results.append({
                '路口名称': inter_name,
                '时段': period_name,
                '进口道方向': CARDINAL_HANZI.get(main_dir, main_dir),
                '转向': '左转' if turn_dir == 1 else '直行',
                '90%分位延误': round(val_90th_delay, 2) if val_90th_delay else None,
                '最大15min平均延误': round(max_15min_delay, 2) if pd.notna(max_15min_delay) else None,
                '溢出率': f"{spill_rate:.1%}",
                '停车次数': round(avg_stop_times, 1) if pd.notna(avg_stop_times) else None,
                '不停车通过速度': round(avg_no_stop_speed, 1) if pd.notna(avg_no_stop_speed) else None,
                '延误指数评级': mode_los
            })
            
    return results

# ================= 【终极版：路口全流向深度诊断图谱】 =================
def generate_all_turning_status_charts(df, inter_name, plot_date, base_output_dir):
    """
    自动为路口的 8 个流向（4进口x2转向）生成延误与全局状态对比图
    逻辑：蓝线展示【该流向本身】的真实延误体验，红阶梯展示【该时刻整个路口】的最恶劣溢出状态。
    """
    # 1. 预处理日期与全局状态
    df['create_time'] = pd.to_datetime(df['create_time'])
    df_date = df[df['create_time'].dt.date == pd.to_datetime(plot_date).date()].copy()
    if df_date.empty: return

    # 计算该路口在该时刻的“全局最高溢出等级”
    if 'idx_state' in df_date.columns:
        df_date['idx_state'] = pd.to_numeric(df_date['idx_state'], errors='coerce').fillna(0)
        df_global_state = df_date.groupby('create_time')['idx_state'].max().reset_index()
        df_global_state.rename(columns={'idx_state': 'global_max_state'}, inplace=True)
    else:
        df_global_state = pd.DataFrame({'create_time': df_date['create_time'].unique(), 'global_max_state': 0})

    # 2. 遍历 4 个方向和 2 种转向
    directions = ['E', 'S', 'W', 'N']
    turns = {1: '左转', 2: '直行'}
    
    # 创建专门的存放子文件夹
    output_dir = os.path.join(base_output_dir, f'全流向诊断图_{plot_date}')
    os.makedirs(output_dir, exist_ok=True)
    
    charts_generated = 0

    for d in directions:
        for t_no, t_name in turns.items():
            # 过滤局部流向数据
            df_target = df_date[(df_date['main_direction'] == d) & (df_date['turn_dir_no'] == t_no)].copy()
            if df_target.empty: continue
            
            df_target = df_target.sort_values('create_time')
            # 融合局部延误与全局状态
            df_plot = pd.merge(df_target[['create_time', 'delay_index']], df_global_state, on='create_time', how='left')
            
            # 开始绘图
            fig, ax1 = plt.subplots(figsize=(12, 5))
            dir_hanzi = CARDINAL_HANZI.get(d, d)
            
            # 左轴：延误指数
            color_idx = '#1f77b4'
            ax1.plot(df_plot['create_time'], df_plot['delay_index'], color=color_idx, linewidth=2, label=f'{dir_hanzi}{t_name}延误')
            ax1.set_ylabel('延误指数', color=color_idx, fontsize=12, fontweight='bold')
            ax1.tick_params(axis='y', labelcolor=color_idx)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax1.grid(True, linestyle='--', alpha=0.4)
            
            # 右轴：路口全局最高溢出状态
            ax2 = ax1.twinx()
            color_st = '#d62728'
            ax2.step(df_plot['create_time'], df_plot['global_max_state'], color=color_st, where='post', linewidth=2, alpha=0.6, label='路口全局最高状态')
            ax2.set_ylabel('路口状态', color=color_st, fontsize=12, fontweight='bold')
            ax2.tick_params(axis='y', labelcolor=color_st)
            ax2.set_ylim(-0.2, 2.5) 
            ax2.set_yticks([0, 1, 2])
            ax2.set_yticklabels(['0-正常', '1-溢出', '2-失衡']) 
            
            # 合并图例
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=10)
            
            plt.title(f'{inter_name} [{dir_hanzi}{t_name}] 延误指数-路口状态对比图 ({plot_date})', fontsize=14, pad=15)
            plt.tight_layout()
            
            # 保存
            file_name = f'{dir_hanzi}_{t_name}_延误指数与路口状态对比图.png'
            plt.savefig(os.path.join(output_dir, file_name), dpi=120)
            plt.close(fig)
            charts_generated += 1
            
    print(f"  -> [{plot_date}] 共成功生成 {charts_generated} 张全流向诊断图。")

# ================= 【新功能：干线瓶颈诊断汇总折线图 (剖面图)】 =================
def generate_summary_line_charts(df_report, output_dir, inter_config, target_main_dir='南向'):
    """
    基于汇总表数据，沿干线方向绘制空间剖面折线图
    """
    df = df_report.copy()
    
    # 仅过滤目标主干线方向的数据（如：南向）
    df = df[df['进口道方向'] == target_main_dir].copy()
    if df.empty:
        print(f"⚠️ 汇总图生成跳过：汇总表中没有方向为【{target_main_dir}】的数据。")
        return

    # 1. 数据预处理
    if '溢出率' in df.columns:
        df['溢出率数值'] = df['溢出率'].astype(str).str.replace('%', '', regex=False)
        df['溢出率数值'] = pd.to_numeric(df['溢出率数值'], errors='coerce')

    # 映射拓扑序列号，确保 X 轴按真实地理顺序排列
    name_to_seq = {v['name']: v['seq'] for k, v in inter_config.items()}
    df['seq'] = df['路口名称'].map(name_to_seq).fillna(99)
    
    periods = df['时段'].dropna().unique()
    turns = df['转向'].dropna().unique()

    # 2. 创建 4x1 的子图布局
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(f'干线走廊 ({target_main_dir}) 瓶颈诊断空间剖面图', fontsize=20, fontweight='bold', y=0.96)

    # 定义要画的 4 个关键指标
    metrics = [
        ('90%分位延误', '90%分位延误指数', axes[0]),
        ('溢出率数值', '溢出触发率 (%)', axes[1]),
        ('停车次数', '平均停车次数 (次)', axes[2]),
        ('不停车速度通过速度', '不停车速度 (m/s)', axes[3])
    ]

    # 定义线条样式
    styles = {
        '早高峰-直行': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-'},   
        '早高峰-左转': {'color': '#1f77b4', 'marker': 's', 'linestyle': '--'},  
        '晚高峰-直行': {'color': '#ff7f0e', 'marker': 'o', 'linestyle': '-'},   
        '晚高峰-左转': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--'}   
    }

    # 获取按地理序列排序的路口名称列表作为统一的 X 轴
    x_labels = sorted(df['路口名称'].unique(), key=lambda x: name_to_seq.get(x, 99))

    # 3. 逐个指标绘图
    for metric_col, ylabel, ax in metrics:
        if metric_col not in df.columns:
            continue

        for period in periods:
            for turn in turns:
                group = df[(df['时段'] == period) & (df['转向'] == turn)]
                if group.empty:
                    continue

                # 提取 Y 值，确保缺失路口用 NaN 占位
                group_dict = dict(zip(group['路口名称'], group[metric_col]))
                y_values = [group_dict.get(name, np.nan) for name in x_labels]

                style_key = f"{period}-{turn}"
                style = styles.get(style_key, {'color': 'gray', 'marker': 'x', 'linestyle': '-'})

                ax.plot(x_labels, y_values, label=style_key,
                        color=style['color'], marker=style['marker'], linestyle=style['linestyle'],
                        linewidth=2.5, markersize=8, alpha=0.8)

        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 优化 Y 轴显示：如果是延误，加一根 2.0 的拥堵警戒红线
        if metric_col == '90%分位延误':
            ax.axhline(y=2.0, color='red', linestyle=':', linewidth=1.5, label='高压警戒线 (DI=2.0)')
        elif metric_col == '溢出率数值':
            ax.axhline(y=15.0, color='red', linestyle=':', linewidth=1.5, label='溢出警戒线 (15%)')

        ax.legend(loc='upper right', framealpha=0.9)

    # 4. 优化 X 轴显示并保存
    plt.xticks(range(len(x_labels)), x_labels, rotation=15, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(output_dir, f'干线瓶颈诊断空间剖面图_{target_main_dir}.png')
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"✅ 走廊剖面折线图已生成: {save_path}")


# ================= 微观指标独立 2x2 阵列图 =================
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

    global_bottleneck_report = []

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

        # 1. 执行流向清洗
        print("  -> 正在计算坐标偏转角与轨迹流向...")
        df_dir = enrich_direction_features(df_dir, offset_degree=offset)
        
        if 'turn_dir_no' in df_dir.columns:
            df_dir = df_dir[df_dir['turn_dir_no'].isin([1, 2])].copy()

        # 2. 拦截并执行单文件的高峰期瓶颈诊断，加入总表
        print("  -> 正在计算高峰期流向级瓶颈指标...")
        report_slice = run_bottleneck_diagnosis_on_slice(df_dir, inter_name)
        global_bottleneck_report.extend(report_slice)

        # 3. 数据可视化处理流程
        folder_name = f'Output_{seq:02d}_{inter_name}'
        output_dir = os.path.join(os.path.dirname(file_path), folder_name)
        
        if date_ranges and date_ranges[0] is not None:
            # 【全新调用：生成全流向局部体验vs全局环境深度诊断图】
            target_date_str = date_ranges[0][0].strftime('%Y-%m-%d')
            generate_all_turning_status_charts(df_dir, inter_name, target_date_str, output_dir)
            
            # 时间区间过滤与聚合出图
            mask_dir = pd.Series(False, index=df_dir.index)
            for start_time, end_time in date_ranges:
                mask_dir |= (df_dir['create_time'] >= start_time) & (df_dir['create_time'] <= end_time)
            df_dir = df_dir[mask_dir]
            
            intervals = [30] 
            
            for interval in intervals:
                analysis_output_dir = os.path.join(output_dir, f'analysis_{interval}min')
                os.makedirs(analysis_output_dir, exist_ok=True)
                
                df_dir_src = df_dir.copy() 
                grouper_dir = [pd.Grouper(key='create_time', freq=f'{interval}min'), 'main_direction', 'turn_dir_no']
                
                target_cols = ['stop_time', 'stop_times', 'delay_index']
                available_cols = [c for c in target_cols if c in df_dir_src.columns]
                
                if available_cols:
                    df_sampled = df_dir_src[df_dir_src['create_time'].dt.minute % 5 == 0].copy()
                    grouped_sampled = df_sampled[available_cols + ['create_time', 'main_direction', 'turn_dir_no']].groupby(grouper_dir)
                    df_dir_agg = grouped_sampled.mean().reset_index()
                    
                    if 'los' in df_sampled.columns:
                        def get_mode(series):
                            mode_series = series.mode()
                            return mode_series.iloc[0] if len(mode_series) > 0 else (series.iloc[0] if len(series) > 0 else None)
                        los_values = df_sampled.groupby(grouper_dir)['los'].apply(get_mode).reset_index()
                        df_dir_agg = pd.merge(df_dir_agg, los_values, on=['create_time', 'main_direction', 'turn_dir_no'], how='left')
                    
                    df_dir_agg['date'] = df_dir_agg['create_time'].dt.date
                    print(f"  -> [{interval}分钟] 微观指标阵列图已生成。")
                    generate_turning_all_metrics_charts(df_dir_agg, inter_name, analysis_output_dir, interval)

    # 4. 循环结束后，统一输出全局干线体检报告
    if global_bottleneck_report:
        df_final_report = pd.DataFrame(global_bottleneck_report)
        print("\n================ 🚦 干线流向级瓶颈诊断汇总表 ================")
        print(df_final_report.to_string(index=False, justify='center'))
        
        base_output_dir = os.path.dirname(file_paths[0])
        report_save_path = os.path.join(base_output_dir, "干线流向级瓶颈诊断汇总表.csv")
        
        try:
            df_final_report.to_csv(report_save_path, index=False, encoding='GBK')
            print(f"\n✅ 汇总表已保存至: {report_save_path}")
        except PermissionError:
            print(f"\n⚠️ 警告: 无法保存 CSV！请先关闭正在 Excel 中打开的 {report_save_path} 文件。")
            print("⚠️ 虽然 CSV 没保存成功，但程序将继续为您生成折线图...")
        
        # 自动生成汇总折线图
        target_hanzi = CARDINAL_HANZI.get(TARGET_DIRECTION, TARGET_DIRECTION)
        generate_summary_line_charts(
            df_report=df_final_report, 
            output_dir=base_output_dir, 
            inter_config=inter_config, 
            target_main_dir=target_hanzi
        )

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