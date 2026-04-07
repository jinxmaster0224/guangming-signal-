import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import matplotlib.dates as mdates

# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  

def generate_charts(df, intersection_name, output_dir, interval):
    """
    生成指定时间间隔的折线图
    """
    try:
        print(f"开始生成{interval}分钟聚合的图表...")
        print(f"数据列: {list(df.columns)}")
        
        # 按日期分组
        df['date'] = df['create_time'].dt.date
        dates = sorted(df['date'].unique())
        
        # 定义颜色映射
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'gray']
        
        # 生成最大排队长度折线图
        plt.figure(figsize=(12, 6))
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            # 创建标准化的时间轴，将所有日期映射到同一天
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            plt.plot(normalized_time, date_df['queue_len_max'], marker='o', markersize=3, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        plt.title(f'{intersection_name} - 最大排队长度变化趋势 ({interval}分钟)')
        plt.xlabel('时间')
        plt.ylabel('最大排队长度')
        plt.grid(True)
        plt.xticks(rotation=45)
        # 设置时间格式为小时:分钟
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.legend()
        plt.tight_layout()
        queue_max_path = os.path.join(output_dir, f'queue_len_max_trend_{interval}min.png')
        plt.savefig(queue_max_path)
        plt.close()
        
        # 生成平均排队长度折线图
        plt.figure(figsize=(12, 6))
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            # 创建标准化的时间轴
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            plt.plot(normalized_time, date_df['queue_len_avg'], marker='o', markersize=3, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        plt.title(f'{intersection_name} - 平均排队长度变化趋势 ({interval}分钟)')
        plt.xlabel('时间')
        plt.ylabel('平均排队长度')
        plt.grid(True)
        plt.xticks(rotation=45)
        # 设置时间格式为小时:分钟
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.legend()
        plt.tight_layout()
        queue_avg_path = os.path.join(output_dir, f'queue_len_avg_trend_{interval}min.png')
        plt.savefig(queue_avg_path)
        plt.close()
        
        # 生成车流量折线图
        plt.figure(figsize=(12, 6))
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            # 创建标准化的时间轴
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            plt.plot(normalized_time, date_df['pass_flow'], marker='o', markersize=3, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        plt.title(f'{intersection_name} - 车流量变化趋势 ({interval}分钟)')
        plt.xlabel('时间')
        plt.ylabel('车流量')
        plt.grid(True)
        plt.xticks(rotation=45)
        # 设置时间格式为小时:分钟
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.legend()
        plt.tight_layout()
        flow_path = os.path.join(output_dir, f'pass_flow_trend_{interval}min.png')
        plt.savefig(flow_path)
        plt.close()
        
        # 生成车均灯前停车时间折线图
        plt.figure(figsize=(12, 6))
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            # 创建标准化的时间轴
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            plt.plot(normalized_time, date_df['stop_time'], marker='o', markersize=3, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        plt.title(f'{intersection_name} - 车均灯前停车时间变化趋势 ({interval}分钟)')
        plt.xlabel('时间')
        plt.ylabel('停车时间 (秒)')
        plt.grid(True)
        plt.xticks(rotation=45)
        # 设置时间格式为小时:分钟
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.legend()
        plt.tight_layout()
        stop_time_path = os.path.join(output_dir, f'stop_time_trend_{interval}min.png')
        plt.savefig(stop_time_path)
        plt.close()

        
        # 生成车均停车次数折线图
        plt.figure(figsize=(12, 6))
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            # 创建标准化的时间轴
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            plt.plot(normalized_time, date_df['stop_times'], marker='o', markersize=3, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        plt.title(f'{intersection_name} - 车均停车次数变化趋势 ({interval}分钟)')
        plt.xlabel('时间')
        plt.ylabel('停车次数')
        plt.grid(True)
        plt.xticks(rotation=45)
        # 设置时间格式为小时:分钟
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.legend()
        plt.tight_layout()
        stop_times_path = os.path.join(output_dir, f'stop_times_trend_{interval}min.png')
        plt.savefig(stop_times_path)
        plt.close()
        
        # 生成车均通过速度折线图
        plt.figure(figsize=(12, 6))
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            # 创建标准化的时间轴
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            plt.plot(normalized_time, date_df['pass_speed'], marker='o', markersize=3, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        plt.title(f'{intersection_name} - 车均通过速度变化趋势 ({interval}分钟)')
        plt.xlabel('时间')
        plt.ylabel('通过速度 (m/s)')
        plt.grid(True)
        plt.xticks(rotation=45)
        # 设置时间格式为小时:分钟
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.legend()
        plt.tight_layout()
        pass_speed_path = os.path.join(output_dir, f'pass_speed_trend_{interval}min.png')
        plt.savefig(pass_speed_path)
        plt.close()
        
        # 生成车均不停车通过速度折线图
        plt.figure(figsize=(12, 6))
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            # 创建标准化的时间轴
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            plt.plot(normalized_time, date_df['no_stop_pass_speed'], marker='o', markersize=3, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        plt.title(f'{intersection_name} - 车均不停车通过速度变化趋势 ({interval}分钟)')
        plt.xlabel('时间')
        plt.ylabel('不停车通过速度 (m/s)')
        plt.grid(True)
        plt.xticks(rotation=45)
        # 设置时间格式为小时:分钟
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.legend()
        plt.tight_layout()
        no_stop_speed_path = os.path.join(output_dir, f'no_stop_pass_speed_trend_{interval}min.png')
        plt.savefig(no_stop_speed_path)
        plt.close()
        
        # 生成延误指数折线图
        plt.figure(figsize=(12, 6))
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            # 创建标准化的时间轴
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            plt.plot(normalized_time, date_df['delay_index'], marker='o', markersize=3, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        plt.title(f'{intersection_name} - 延误指数变化趋势 ({interval}分钟)')
        plt.xlabel('时间')
        plt.ylabel('延误指数')
        plt.grid(True)
        plt.xticks(rotation=45)
        # 设置时间格式为小时:分钟
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.legend()
        plt.tight_layout()
        delay_path = os.path.join(output_dir, f'delay_index_trend_{interval}min.png')
        plt.savefig(delay_path)
        plt.close()
        
        # 生成路口状态折线图
        plt.figure(figsize=(12, 6))
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            # 创建标准化的时间轴
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            plt.plot(normalized_time, date_df['idx_state'], marker='o', markersize=3, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        plt.title(f'{intersection_name} - 路口状态变化趋势 ({interval}分钟)')
        plt.xlabel('时间')
        plt.ylabel('路口状态')
        plt.grid(True)
        plt.xticks(rotation=45)
        # 设置时间格式为小时:分钟
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.legend()
        plt.tight_layout()
        idx_state_path = os.path.join(output_dir, f'idx_state_trend_{interval}min.png')
        plt.savefig(idx_state_path)
        plt.close()
        
        # 生成延误指数评级折线图
        if 'los' in df.columns:
            plt.figure(figsize=(12, 6))
            # 为分类值创建映射，延误程度越轻值越小，这样在图表中越轻的延误显示在下方
            los_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'nan': 0}
            
            for i, date in enumerate(dates):
                date_df = df[df['date'] == date]
                los_values = date_df['los'].map(los_mapping).fillna(0)
                # 创建标准化的时间轴
                time_only = date_df['create_time'].dt.time
                normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
                # 绘制折线图
                plt.plot(normalized_time, los_values, marker='o', markersize=3, linewidth=1, 
                         color=colors[i % len(colors)], label=str(date))
            
            plt.title(f'{intersection_name} - 延误指数评级变化趋势 ({interval}分钟)')
            plt.xlabel('时间')
            plt.ylabel('延误指数评级')
            plt.grid(True)
            plt.xticks(rotation=45)
            # 设置时间格式为小时:分钟
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            # 设置纵坐标显示字母标签
            plt.yticks([0, 1, 2, 3, 4, 5], ['A', 'B', 'C', 'D', 'E', 'F'])
            plt.legend()
            plt.tight_layout()
            los_path = os.path.join(output_dir, f'los_trend_{interval}min.png')
            plt.savefig(los_path)
            plt.close()
        
        # 生成多指标对比图（4x3布局）
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
            # 创建标准化的时间轴
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            plt.plot(normalized_time, date_df['queue_len_max'], marker='o', markersize=2, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        plt.title('最大排队长度趋势图')
        plt.ylabel('最大排队长度（米）')
        plt.xticks(rotation=45)
        # 设置时间格式为小时:分钟
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.legend()
        plt.grid(True)
        
        # 平均排队长度
        plt.subplot(4, 3, 2)
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            # 创建标准化的时间轴
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            plt.plot(normalized_time, date_df['queue_len_avg'], marker='o', markersize=2, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        plt.title('平均排队长度趋势图')
        plt.ylabel('平均排队长度（米）')
        plt.xticks(rotation=45)
        # 设置时间格式为小时:分钟
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.legend()
        plt.grid(True)
        
        # 车流量
        plt.subplot(4, 3, 3)
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            # 创建标准化的时间轴
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            plt.plot(normalized_time, date_df['pass_flow'], marker='o', markersize=2, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        plt.title('车流量趋势图')
        plt.ylabel('车流量（辆）')
        plt.xticks(rotation=45)
        # 设置时间格式为小时:分钟
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.legend()
        plt.grid(True)
        
        # 车均灯前停车时间
        plt.subplot(4, 3, 4)
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            # 创建标准化的时间轴
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            plt.plot(normalized_time, date_df['stop_time'], marker='o', markersize=2, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        plt.title('停车时间趋势图')
        plt.ylabel('停车时间 (秒)')
        plt.xticks(rotation=45)
        # 设置时间格式为小时:分钟
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.legend()
        plt.grid(True)
        
        # 车均停车次数
        plt.subplot(4, 3, 5)
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            # 创建标准化的时间轴
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            plt.plot(normalized_time, date_df['stop_times'], marker='o', markersize=2, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        plt.title('停车次数趋势图')
        plt.ylabel('停车次数（次）')
        plt.xticks(rotation=45)
        # 设置时间格式为小时:分钟
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.legend()
        plt.grid(True)
        
        # 车均通过速度
        plt.subplot(4, 3, 6)
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            # 创建标准化的时间轴
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            plt.plot(normalized_time, date_df['pass_speed'], marker='o', markersize=2, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        plt.title('通过速度趋势图')
        plt.ylabel('通过速度 (m/s)')
        plt.xticks(rotation=45)
        # 设置时间格式为小时:分钟
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.legend()
        plt.grid(True)
        
        # 车均不停车通过速度
        plt.subplot(4, 3, 7)
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            # 创建标准化的时间轴
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            plt.plot(normalized_time, date_df['no_stop_pass_speed'], marker='o', markersize=2, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        plt.title('不停车通过速度趋势图')
        plt.ylabel('不停车通过速度 (m/s)')
        plt.xticks(rotation=45)
        # 设置时间格式为小时:分钟
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.legend()
        plt.grid(True)
        
        # 延误指数
        plt.subplot(4, 3, 8)
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            # 创建标准化的时间轴
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            plt.plot(normalized_time, date_df['delay_index'], marker='o', markersize=2, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        plt.title('延误指数趋势图')
        plt.ylabel('延误指数')
        plt.xticks(rotation=45)
        # 设置时间格式为小时:分钟
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.legend()
        plt.grid(True)
        
        # 路口状态
        plt.subplot(4, 3, 9)
        for i, date in enumerate(dates):
            date_df = df[df['date'] == date]
            # 创建标准化的时间轴
            time_only = date_df['create_time'].dt.time
            normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
            plt.plot(normalized_time, date_df['idx_state'], marker='o', markersize=2, linewidth=1, 
                     color=colors[i % len(colors)], label=str(date))
        plt.title('路口状态趋势图')
        plt.ylabel('路口状态')
        plt.xticks(rotation=45)
        # 设置时间格式为小时:分钟
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.legend()
        plt.grid(True)
        
        # 延误指数评级
        if 'los' in df.columns:
            plt.subplot(4, 3, 10)
            # 为分类值创建映射，延误程度越轻值越小，这样在图表中越轻的延误显示在下方
            los_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'nan': 0}
            
            for i, date in enumerate(dates):
                date_df = df[df['date'] == date]
                los_values = date_df['los'].map(los_mapping).fillna(0)
                # 创建标准化的时间轴
                time_only = date_df['create_time'].dt.time
                normalized_time = pd.to_datetime('2026-01-01 ' + time_only.astype(str))
                # 绘制折线图
                plt.plot(normalized_time, los_values, marker='o', markersize=2, linewidth=1, 
                         color=colors[i % len(colors)], label=str(date))
            
            plt.title('延误指数评级趋势图')
            plt.ylabel('延误指数评级')
            plt.xticks(rotation=45)
            # 设置时间格式为小时:分钟
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            # 设置纵坐标显示字母标签
            plt.yticks([0, 1, 2, 3, 4, 5], ['A', 'B', 'C', 'D', 'E', 'F'])
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        combined_path = os.path.join(output_dir, f'combined_analysis_{interval}min.png')
        plt.savefig(combined_path)
        plt.close()
        print(f"生成多指标对比图完成")
        
        print(f"{interval}分钟聚合分析完成！")
        
        if 'los' in df.columns:
            print(f"- 延误指数评级趋势图: {los_path}")
        print(f"- 多指标对比图: {combined_path}")
    except Exception as e:
        print(f"生成图表时出错: {e}")
        import traceback
        traceback.print_exc()

def generate_vertical_comparison_charts(data_by_interval, intersection_name, output_dir):
    """
    生成纵向对比图，显示每个指标在不同时间间隔下的对比
    
    参数:
    data_by_interval: dict - 键为时间间隔（分钟），值为对应的DataFrame
    intersection_name: str - 路口名称
    output_dir: str - 输出目录
    """
    try:
        print("开始生成纵向对比图...")
        
        # 创建输出目录
        comparison_dir = os.path.join(output_dir, '纵向对比图')
        os.makedirs(comparison_dir, exist_ok=True)
        
        # 定义颜色映射
        colors = {
            1: 'blue',
            5: 'green',
            15: 'red',
            30: 'purple',
            60: 'orange'
        }
        
        # 定义指标配置
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
        
        # 为每个指标生成2x3布局的对比图
        for metric in metrics:
            plt.figure(figsize=(18, 8))
            
            # 按照时间间隔顺序排列
            intervals = [1, 5, 15, 30, 60]
            
            for i, interval in enumerate(intervals, 1):
                plt.subplot(2, 3, i)
                
                if interval in data_by_interval and metric['column'] in data_by_interval[interval].columns:
                    df = data_by_interval[interval]
                    plt.plot(df['create_time'], df[metric['column']], 
                             marker='o', markersize=2, linewidth=1, 
                             color=colors.get(interval, 'gray'))
                    plt.title(f'{interval}分钟聚合')
                    plt.ylabel(metric['ylabel'])
                    plt.xticks(rotation=45)
                    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    plt.grid(True)
            
            plt.suptitle(f'{intersection_name} - {metric["name"]}不同时间间隔对比', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # 保存图表
            chart_path = os.path.join(comparison_dir, f'{metric["name"]}_时间间隔对比.png')
            plt.savefig(chart_path)
            plt.close()
        
        # 为延误指数评级生成2x3布局的对比图（如果有）
        if 'los' in list(data_by_interval.values())[0].columns:
            plt.figure(figsize=(18, 8))
            
            # 按照时间间隔顺序排列
            intervals = [1, 5, 15, 30, 60]
            los_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'nan': 0}
            
            for i, interval in enumerate(intervals, 1):
                plt.subplot(2, 3, i)
                
                if interval in data_by_interval and 'los' in data_by_interval[interval].columns:
                    df = data_by_interval[interval]
                    los_values = df['los'].map(los_mapping).fillna(0)
                    plt.plot(df['create_time'], los_values, 
                             marker='o', markersize=2, linewidth=1, 
                             color=colors.get(interval, 'gray'))
                    plt.title(f'{interval}分钟聚合')
                    plt.ylabel('延误指数评级')
                    plt.xticks(rotation=45)
                    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    plt.yticks([0, 1, 2, 3, 4, 5], ['A', 'B', 'C', 'D', 'E', 'F'])
                    plt.grid(True)
            
            plt.suptitle(f'{intersection_name} - 延误指数评级不同时间间隔对比', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # 保存图表
            chart_path = os.path.join(comparison_dir, '延误指数评级_时间间隔对比.png')
            plt.savefig(chart_path)
            plt.close()

        
        print("纵向对比图生成完成！")
    except Exception as e:
        print(f"生成纵向对比图时出错: {e}")
        import traceback
        traceback.print_exc()

def get_date_ranges():
    """
    获取预设的日期范围
    
    返回:
    list - 日期范围列表，每个元素是 (start_time, end_time) 元组
    """
    # 预设时间段配置，True，False。
    # 格式: (启用标志, 开始时间, 结束时间)
    time_periods = [
        (True, '2026-03-10 00:00:00', '2026-03-10 23:59:59'),  # 2月2日全天
        (True, '2026-03-11 00:00:00', '2026-03-11 23:59:59'),  # 时间段2
        (True, '2026-03-12 00:00:00', '2026-03-12 23:59:59'), # 时间段3
        (False, '2026-03-22 00:00:00', '2026-03-22 23:59:59'), # 时间段4
        (False, '2026-03-29 00:00:00', '2026-03-29 23:59:59')  # 时间段5
    ]
    
    date_ranges = []
    
    for i, (enable, start_date, end_date) in enumerate(time_periods, 1):
        if enable:
            try:
                start_time = pd.Timestamp(start_date)
                end_time = pd.Timestamp(end_date)
                date_ranges.append((start_time, end_time))
                print(f"已启用时间段{i}：{start_date} 到 {end_date}")
            except Exception as e:
                print(f"时间段{i}日期格式错误：{e}")
    
    if not date_ranges:
        # 默认时间段
        print("未启用任何时间段，使用数据中的第一天")
        # 这里会在 analyze_traffic_data 函数中处理
        date_ranges.append(None)
    
    return date_ranges

def analyze_data_completeness(df, date_ranges):
    """
    分析数据完整性，检查所选时间段内的分钟级数据是否完整
    
    参数:
    df: DataFrame - 过滤后的交通数据
    date_ranges: list - 日期范围列表，每个元素是 (start_time, end_time) 元组
    """
    print("\n开始数据完整性分析...")
    
    # 按日期分组检查
    if date_ranges and date_ranges[0] is not None:
        for start_time, end_time in date_ranges:
            # 生成完整的分钟级时间序列
            complete_minutes = pd.date_range(start=start_time, end=end_time, freq='1min')
            
            # 提取该日期数据中实际存在的分钟
            date_mask = (df['create_time'] >= start_time) & (df['create_time'] <= end_time)
            actual_minutes = pd.DatetimeIndex(df[date_mask]['create_time'].dt.floor('1min'))
            
            # 找出缺失的分钟
            missing_minutes = complete_minutes.difference(actual_minutes)
            
            # 按日期分组统计缺失情况
            if not missing_minutes.empty:
                print(f"\n日期 {start_time.date()} 数据完整性分析:")
                print(f"总分钟数: {len(complete_minutes)}")
                print(f"实际分钟数: {len(actual_minutes.unique())}")
                print(f"缺失分钟数: {len(missing_minutes)}")
                print(f"缺失率: {len(missing_minutes)/len(complete_minutes):.2%}")
                
                # 打印缺失的分钟
                if len(missing_minutes) <= 10:
                    print("缺失的具体分钟:")
                    for minute in missing_minutes:
                        print(f"  - {minute.strftime('%H:%M')}")
                else:
                    print("缺失的前10个分钟:")
                    for minute in missing_minutes[:10]:
                        print(f"  - {minute.strftime('%H:%M')}")
                    print(f"... 还有 {len(missing_minutes) - 10} 个缺失分钟")
            else:
                print(f"\n日期 {start_time.date()} 数据完整性分析:")
                print("✓ 数据完整，无缺失分钟")
    else:
        # 当没有指定日期范围时，使用数据中的日期
        if not df.empty:
            # 获取数据中的所有日期
            dates = df['create_time'].dt.date.unique()
            for date in dates:
                start_time = pd.Timestamp(f'{date} 00:00:00')
                end_time = pd.Timestamp(f'{date} 23:59:59')
                
                # 生成完整的分钟级时间序列
                complete_minutes = pd.date_range(start=start_time, end=end_time, freq='1min')
                
                # 提取该日期数据中实际存在的分钟
                date_mask = df['create_time'].dt.date == date
                actual_minutes = pd.DatetimeIndex(df[date_mask]['create_time'].dt.floor('1min'))
                
                # 找出缺失的分钟
                missing_minutes = complete_minutes.difference(actual_minutes)
                
                if not missing_minutes.empty:
                    print(f"\n日期 {date} 数据完整性分析:")
                    print(f"总分钟数: {len(complete_minutes)}")
                    print(f"实际分钟数: {len(actual_minutes.unique())}")
                    print(f"缺失分钟数: {len(missing_minutes)}")
                    print(f"缺失率: {len(missing_minutes)/len(complete_minutes):.2%}")
                    
                    # 打印缺失的分钟
                    if len(missing_minutes) <= 10:
                        print("缺失的具体分钟:")
                        for minute in missing_minutes:
                            print(f"  - {minute.strftime('%H:%M')}")
                    else:
                        print("缺失的前10个分钟:")
                        for minute in missing_minutes[:10]:
                            print(f"  - {minute.strftime('%H:%M')}")
                        print(f"... 还有 {len(missing_minutes) - 10} 个缺失分钟")
                else:
                    print(f"\n日期 {date} 数据完整性分析:")
                    print("✓ 数据完整，无缺失分钟")
    print("\n数据完整性分析完成！")

def analyze_traffic_data(csv_path, date_ranges, print_raw_data=False):
    """
    分析交通数据并生成折线图
    
    参数:
    csv_path: str - CSV文件路径
    date_ranges: list - 日期范围列表，每个元素是 (start_time, end_time) 元组
    print_raw_data: bool - 是否打印过滤后的原始数据前100条
    """
    # 读取CSV文件，尝试不同编码
    try:
        df = pd.read_csv(csv_path, encoding='GBK')
    except:
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except:
            df = pd.read_csv(csv_path, encoding='latin1')
    
    # 转换时间格式
    df['create_time'] = pd.to_datetime(df['create_time'])
    
    # 按时间排序
    df = df.sort_values('create_time')
    
    # 提取路口名称
    intersection_name = df['inter_name'].iloc[0] if not df.empty else 'Unknown'
    
    # 过滤数据：根据用户选择的日期范围
    if date_ranges and date_ranges[0] is not None:
        # 打印使用的时间段
        print("数据过滤完成！使用的时间段：")
        for i, (start_time, end_time) in enumerate(date_ranges, 1):
            print(f"时间段{i}：{start_time.strftime('%Y-%m-%d %H:%M:%S')} 到 {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 数据完整性分析
        analyze_data_completeness(df, date_ranges)
        
        # 定义不同的时间间隔
        intervals = [1, 5, 15, 30, 60]
        
        # 收集所有时间间隔的数据
        data_by_interval = {}
        
        for interval in intervals:
            # 创建输出目录
            output_dir = os.path.join(os.path.dirname(csv_path), f'analysis_{interval}min')
            os.makedirs(output_dir, exist_ok=True)
            
            # 按指定时间间隔聚合数据
            if interval == 1:
                # 1分钟间隔，直接去重
                df_interval = df.groupby(pd.Grouper(key='create_time', freq='1min')).first().reset_index()
                
                # 对于los列，确保数据类型为字符串
                if 'los' in df_interval.columns:
                    df_interval['los'] = df_interval['los'].astype(str)
                
                # 对于idx_state列，使用第一个值（路口状态是离散值，不应该计算平均值）
                if 'idx_state' in df.columns:
                    # 获取每个时间间隔的第一个idx_state值
                    idx_state_values = df.groupby(pd.Grouper(key='create_time', freq='1min'))['idx_state'].first().reset_index()
                    # 重命名列以避免冲突
                    idx_state_values = idx_state_values.rename(columns={'idx_state': 'idx_state_new'})
                    # 合并到结果中
                    df_interval = df_interval.merge(idx_state_values, on='create_time', how='left')
                    # 删除原始的idx_state列，重命名新的列
                    df_interval = df_interval.drop(columns=['idx_state'])
                    df_interval = df_interval.rename(columns={'idx_state_new': 'idx_state'})
            else:
                # 其他间隔，只对数值列使用平均值聚合
                # 排除los列和idx_state列，因为它们是离散状态值
                numeric_cols = ['queue_len_max', 'queue_len_avg', 'pass_flow', 'stop_time', 'stop_times', 'pass_speed', 'no_stop_pass_speed', 'delay_index', 'confidence']
                df_interval = df[numeric_cols + ['create_time']].groupby(pd.Grouper(key='create_time', freq=f'{interval}min')).mean().reset_index()
                
                # 对于los列，使用众数（最常见的值）
                if 'los' in df.columns:
                    # 获取每个时间间隔的los众数
                    def get_mode(series):
                        if len(series.mode()) > 0:
                            return series.mode().iloc[0]
                        else:
                            return series.iloc[0] if len(series) > 0 else None
                    
                    los_values = df.groupby(pd.Grouper(key='create_time', freq=f'{interval}min'))['los'].apply(get_mode).reset_index()
                    # 合并到结果中
                    df_interval = df_interval.merge(los_values, on='create_time', how='left')
                
                # 对于idx_state列，使用第一个值（路口状态是离散值，不应该计算平均值）
                if 'idx_state' in df.columns:
                    # 获取每个时间间隔的第一个idx_state值
                    idx_state_values = df.groupby(pd.Grouper(key='create_time', freq=f'{interval}min'))['idx_state'].first().reset_index()
                    # 合并到结果中
                    df_interval = df_interval.merge(idx_state_values, on='create_time', how='left')
            
            # 过滤聚合后的数据，只保留指定时间段内的数据
            if date_ranges and date_ranges[0] is not None:
                mask = False
                for start_time, end_time in date_ranges:
                    mask |= (df_interval['create_time'] >= start_time) & (df_interval['create_time'] <= end_time)
                df_interval = df_interval[mask]
            
            print(f"{interval}分钟聚合完成！共 {len(df_interval)} 条记录")
            
            # 保存数据到字典中
            data_by_interval[interval] = df_interval
            
            # 生成图表
            generate_charts(df_interval, intersection_name, output_dir, interval)
        
        # 生成纵向对比图
        generate_vertical_comparison_charts(data_by_interval, intersection_name, os.path.dirname(csv_path))
    else:
        # 过滤数据：只取一天24小时的数据
        # 取数据中的第一天
        first_date = df['create_time'].min().date()
        start_time = pd.Timestamp(f'{first_date} 00:00:00')
        end_time = pd.Timestamp(f'{first_date} 23:59:59')
        
        # 过滤数据
        df = df[(df['create_time'] >= start_time) & (df['create_time'] <= end_time)]
        
        print(f"数据过滤完成！使用 {first_date} 的数据")
        
        # 数据完整性分析
        analyze_data_completeness(df, date_ranges)
        
        # 如果开启打印原始数据开关，打印前20条数据
        if print_raw_data:
            print("\n过滤后的原始数据前20条：")
            # 设置pandas显示选项，确保完整显示所有列和行，不要省略
            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_rows', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', None)
            print(df.head(20))
        
        # 定义不同的时间间隔
        intervals = [1, 5, 15, 30, 60]
        
        # 收集所有时间间隔的数据
        data_by_interval = {}
        
        for interval in intervals:
            # 创建输出目录
            output_dir = os.path.join(os.path.dirname(csv_path), f'analysis_{interval}min')
            os.makedirs(output_dir, exist_ok=True)
            
            # 按指定时间间隔聚合数据
            if interval == 1:
                # 1分钟间隔，直接去重
                df_interval = df.groupby(pd.Grouper(key='create_time', freq='1min')).first().reset_index()
                
                # 对于los列，确保数据类型为字符串
                if 'los' in df_interval.columns:
                    df_interval['los'] = df_interval['los'].astype(str)
                
                # 对于idx_state列，使用第一个值（路口状态是离散值，不应该计算平均值）
                if 'idx_state' in df.columns:
                    # 获取每个时间间隔的第一个idx_state值
                    idx_state_values = df.groupby(pd.Grouper(key='create_time', freq='1min'))['idx_state'].first().reset_index()
                    # 重命名列以避免冲突
                    idx_state_values = idx_state_values.rename(columns={'idx_state': 'idx_state_new'})
                    # 合并到结果中
                    df_interval = df_interval.merge(idx_state_values, on='create_time', how='left')
                    # 删除原始的idx_state列，重命名新的列
                    df_interval = df_interval.drop(columns=['idx_state'])
                    df_interval = df_interval.rename(columns={'idx_state_new': 'idx_state'})
            else:
                # 其他间隔，只对数值列使用平均值聚合
                # 排除los列和idx_state列，因为它们是离散状态值
                numeric_cols = ['queue_len_max', 'queue_len_avg', 'pass_flow', 'stop_time', 'stop_times', 'pass_speed', 'no_stop_pass_speed', 'delay_index', 'confidence']
                df_interval = df[numeric_cols + ['create_time']].groupby(pd.Grouper(key='create_time', freq=f'{interval}min')).mean().reset_index()
                
                # 对于los列，使用众数（最常见的值）
                if 'los' in df.columns:
                    # 获取每个时间间隔的los众数
                    def get_mode(series):
                        if len(series.mode()) > 0:
                            return series.mode().iloc[0]
                        else:
                            return series.iloc[0] if len(series) > 0 else None
                    
                    los_values = df.groupby(pd.Grouper(key='create_time', freq=f'{interval}min'))['los'].apply(get_mode).reset_index()
                    # 合并到结果中
                    df_interval = df_interval.merge(los_values, on='create_time', how='left')
                
                # 对于idx_state列，使用第一个值（路口状态是离散值，不应该计算平均值）
                if 'idx_state' in df.columns:
                    # 获取每个时间间隔的第一个idx_state值
                    idx_state_values = df.groupby(pd.Grouper(key='create_time', freq=f'{interval}min'))['idx_state'].first().reset_index()
                    # 合并到结果中
                    df_interval = df_interval.merge(idx_state_values, on='create_time', how='left')
            
            print(f"{interval}分钟聚合完成！共 {len(df_interval)} 条记录")
            
            # 保存数据到字典中
            data_by_interval[interval] = df_interval
            
            # 生成图表
            generate_charts(df_interval, intersection_name, output_dir, interval)
        
        # 生成纵向对比图
        generate_vertical_comparison_charts(data_by_interval, intersection_name, os.path.dirname(csv_path))

if __name__ == "__main__":
    # 自动查找脚本所在目录中的CSV文件
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
    
    # 构建完整的CSV文件路径
    csv_file_path = os.path.join(script_dir, csv_file)
    print(f"使用CSV文件：{csv_file_path}")
    
    # 控制是否打印过滤后的原始数据前100条
    # 设置为True时打印，False时不打印
    print_raw_data = False
    # 获取预设的日期范围
    date_ranges = get_date_ranges()
    # 分析交通数据
    analyze_traffic_data(csv_file_path, date_ranges, print_raw_data=print_raw_data)