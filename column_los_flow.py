import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates
import numpy as np

# ================= 核心配置区 =================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False  

# 自定义周期时间配置
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
        (True,  '2026-03-10 00:00:00', '2026-03-10 23:59:59'),  # 示例日期 1
        (True,  '2026-03-11 00:00:00', '2026-03-16 23:59:59'),  # 示例日期 2
        (False, '2026-04-03 00:00:00', '2026-04-05 23:59:59'),  # 示例连续时间段
    ]
    
    date_ranges = []
    for enable, start_date, end_date in time_periods:
        if enable:
            try:
                date_ranges.append((pd.Timestamp(start_date), pd.Timestamp(end_date)))
                print(f"[启用日期]: {start_date[:10]} 至 {end_date[:10]}")
            except Exception as e:
                print(f"日期格式错误: {e}")
                
    # 如果全部是 False，则默认返回 None，触发扫描全表
    if not date_ranges:
        print("[启用日期]: 自动扫描全表")
        date_ranges.append(None)
        
    return date_ranges

# ================= 绘图辅助与核心逻辑 =================
def add_custom_periods_overlay(ax, periods_config, base_date='2026-01-01'):
    """添加时段划分红色虚线与信号周期黑色虚线图"""
    if not periods_config: return
    times, values = [], []
    
    # 绘制红色垂直虚线
    for period in periods_config:
        start_time = pd.to_datetime(f"{base_date} {period['start']}")
        times.append(start_time)
        values.append(period['value'])
        ax.axvline(x=start_time, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
        
    last_end_time = pd.to_datetime(f"{base_date} {periods_config[-1]['end']}")
    times.append(last_end_time)
    values.append(periods_config[-1]['value']) 
    ax.axvline(x=last_end_time, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
    
    # 创建第三个Y轴（向外偏移60像素，避免与延误指数轴重叠）
    ax3 = ax.twinx()
    ax3.spines['right'].set_position(('outward', 60))  
    ax3.step(times, values, where='post', color='black', linestyle='--', linewidth=2.5, label='周期时间')
    
    ax3.set_ylabel('周期时间 (秒)', color='black', fontweight='bold')
    ax3.tick_params(axis='y', labelcolor='black')
    ax3.legend(loc='upper right')

def generate_hourly_bar_charts(df, intersection_name, output_dir):
    """生成1小时聚合的双柱图+周期虚线"""
    print("\n[进度] 正在生成 分均流量-延误指数柱状图...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 强制按 60 分钟聚合求均值
    df_60 = df.groupby(pd.Grouper(key='create_time', freq='60min'))[['pass_flow', 'delay_index']].mean().reset_index()
    df_60['date'] = df_60['create_time'].dt.date
    dates = sorted(df_60['date'].unique())

    for date in dates:
        date_df = df_60[df_60['date'] == date].copy()
        if date_df.empty: continue
            
        # 时间标准化对齐到图表系
        date_df['norm_time'] = pd.to_datetime('2026-01-01 ' + date_df['create_time'].dt.time.astype(str))
        
        # 创建画布，留出右侧空间给第三Y轴
        fig, ax1 = plt.subplots(figsize=(14, 7))
        fig.subplots_adjust(right=0.85) 

        # 柱子宽度设置：每根占18分钟，两根拼在一起约半小时
        bar_width = pd.Timedelta(minutes=18)
        
        # --- 绘制主轴（左）：分均车流量 ---
        # 柱子向左偏移 9 分钟，紧贴中心点
        bars1 = ax1.bar(date_df['norm_time'] - pd.Timedelta(minutes=9), date_df['pass_flow'], 
                        width=bar_width, color='#4c72b0', alpha=0.85, label='分均车流量')
        ax1.set_ylabel('分均车流量 (辆/分)', color='#4c72b0', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='#4c72b0')
        ax1.set_xlabel('时间', fontsize=12)
        
        # --- 绘制次轴（右）：延误指数 ---
        ax2 = ax1.twinx()
        # 柱子向右偏移 9 分钟，与流量柱紧紧依附
        bars2 = ax2.bar(date_df['norm_time'] + pd.Timedelta(minutes=9), date_df['delay_index'], 
                        width=bar_width, color='#dd8452', alpha=0.85, label='延误指数')
        ax2.set_ylabel('延误指数', color='#dd8452', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='#dd8452')
        
        # --- 绘制周期刻度虚线（扩展右侧黑轴） ---
        add_custom_periods_overlay(ax1, CUSTOM_PERIODS)
        
        # 图例合并显示在左上角
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=11)

        # X轴刻度美化（整点显示）
        ax1.set_xticks(pd.date_range('2026-01-01 00:00', '2026-01-01 23:59', freq='1H'))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax1.get_xticklabels(), rotation=45)
        ax1.grid(True, axis='y', linestyle=':', alpha=0.5)

        plt.title(f'{intersection_name} - 分均流量与延误指数分析 ({date})', fontsize=16, fontweight='bold', pad=15)
        
        output_path = os.path.join(output_dir, f'分均流量-延误指数分析_{date}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    ✓ 成功生成: {os.path.basename(output_path)}")

# ================= 数据入口管线 =================
def analyze_traffic_data(info_csv_path, date_ranges):
    print(f"\n正在读取路口数据: {os.path.basename(info_csv_path)}")
    try: df = pd.read_csv(info_csv_path, encoding='GBK')
    except:
        try: df = pd.read_csv(info_csv_path, encoding='utf-8')
        except: df = pd.read_csv(info_csv_path, encoding='latin1')
        
    df['create_time'] = pd.to_datetime(df['create_time'])
    df = df.sort_values('create_time')
    intersection_name = df['inter_name'].iloc[0] if 'inter_name' in df.columns and not df.empty else '未知路口'
    
    # 根据代码中配置的日期进行过滤
    if date_ranges and date_ranges[0] is not None:
        mask = False
        for start_time, end_time in date_ranges:
            mask |= (df['create_time'] >= start_time) & (df['create_time'] <= end_time)
        df = df[mask]
        if df.empty:
            print("\n⚠️ 警告：配置的日期范围内没有找到任何数据！请检查日期是否正确。")
            return
    
    output_dir = os.path.join(os.path.dirname(info_csv_path), '分析结果_1小时双柱图')
    generate_hourly_bar_charts(df, intersection_name, output_dir)
    print("图表生成完毕！")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = [f for f in os.listdir(script_dir) if f.endswith('.csv')]
    
    info_csv = next((os.path.join(script_dir, f) for f in csv_files if "info_view" in f), None)
    if not info_csv and csv_files: info_csv = os.path.join(script_dir, csv_files[0])
        
    if info_csv:
        # 1. 直接获取代码里写好的日期范围
        date_ranges = get_date_ranges()
        # 2. 传给分析函数执行
        analyze_traffic_data(info_csv, date_ranges)
    else:
        print(f"错误：在脚本所在目录 {script_dir} 中没有找到任何 CSV 文件！")