import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import re

# ==========================================
# 0. 环境、配置与图表样式设置
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font="SimHei")

# 路口名称与偏移角配置
INTER_NAME_MAP = {
    '6ca93d215b3ee3': '光明大道与华夏路',
    '6caa06b15b50eb': '光明大道与光安路',
    '6caa3ec15b566e': '光明大道与光明大街',
    '6caaa6f15b735a': '光明大道与光辉大道',
    '6ca887115b2e22': '光明大道与华裕路'
}

INTER_OFFSET_MAP = {
    '光明大道与光辉大道': 0.0, 
    '光明大道与光明大街': -32.57, 
    '光明大道与光安路': -33.69,
    '光明大道与华夏路': -63.5, 
    '光明大道与华裕路': -63.0,
}

CARDINAL_HANZI = {"E": "东向", "W": "西向", "S": "南向", "N": "北向"}
TURN_HANZI = {1: "左转", 2: "直行"}
_WKT_COORD_PATTERN = re.compile(r'(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)')

# ==========================================
# 1. 核心换算模块
# ==========================================
def extract_direction_from_coords(lng_lat_seq: str, offset_degree: float = 0.0) -> str:
    """根据经纬度坐标序列计算实际进口道方向(E, W, S, N)"""
    if not isinstance(lng_lat_seq, str) or not str(lng_lat_seq).strip(): 
        return None
    matches = _WKT_COORD_PATTERN.findall(str(lng_lat_seq))
    if len(matches) < 2: 
        return None
    
    try: 
        points = [(float(x), float(y)) for x, y in matches]
    except ValueError: 
        return None
        
    dx = points[-1][0] - points[0][0]
    dy = points[-1][1] - points[0][1]
    
    if abs(dx) < 1e-6 and abs(dy) < 1e-6: 
        return None
        
    angle_deg = math.degrees(math.atan2(dy, dx)) - offset_degree
    if angle_deg > 180: angle_deg -= 360
    elif angle_deg <= -180: angle_deg += 360
    
    if -45.0 <= angle_deg < 45.0: return "E"
    if 45.0 <= angle_deg < 135.0: return "N"
    if -135.0 <= angle_deg < -45.0: return "S"
    return "W"

# ==========================================
# 2. 数据处理与绘图模块
# ==========================================
def prepare_data(df, inter_id):
    """
    数据预处理：通过聚合行算方向 -> 共享给同进口道 -> 提取 8 个转向 + 1 个汇总
    """
    df_inter = df[df['inter_id'] == inter_id].copy()
    
    # 剔除缺失核心指标的空行
    df_inter = df_inter.dropna(subset=['pass_flow', 'queue_len_max', 'stop_time'])
    
    inter_name = INTER_NAME_MAP.get(inter_id, inter_id)
    offset_deg = INTER_OFFSET_MAP.get(inter_name, 0.0)
    
    # 1. 提取 frid -> 物理方向的映射 (仅依赖 turn_dir_no == 0 的聚合行)
    frid_to_dir = {}
    agg_rows = df_inter[df_inter['turn_dir_no'] == 0]
    for _, row in agg_rows.iterrows():
        if pd.notna(row['lng_lat_seq']):
            dir_code = extract_direction_from_coords(row['lng_lat_seq'], offset_deg)
            if dir_code:
                frid_to_dir[row['frid']] = dir_code

    # 2. 广播：将物理方向映射回该进口道的所有数据行
    df_inter['dir_code'] = df_inter['frid'].map(frid_to_dir)
    
    plot_data_dict = {}
    
    # 3. 精准提取 8 个转向的数据
    for dir_code in ['E', 'S', 'W', 'N']:
        for turn_no in [1, 2]:
            mask = (df_inter['dir_code'] == dir_code) & (df_inter['turn_dir_no'] == turn_no)
            subset = df_inter[mask].copy()
            if not subset.empty:
                dir_zh = CARDINAL_HANZI.get(dir_code, dir_code)
                turn_zh = TURN_HANZI.get(turn_no, str(turn_no))
                key_name = f"{dir_zh} {turn_zh}"
                plot_data_dict[key_name] = subset

    # 4. 计算路口整体汇总数据 (按 create_time 聚合 turn_dir_no == 0 的行)
    df_agg = df_inter[df_inter['turn_dir_no'] == 0].copy()
    
    if not df_agg.empty:
        total_data = df_agg.groupby('create_time').apply(
            lambda x: pd.Series({
                'pass_flow': x['pass_flow'].sum(),
                'queue_len_max': x['queue_len_max'].max(),
                'stop_time': np.average(x['stop_time'], weights=x['pass_flow'] + 1e-5),
                'delay_index': np.average(x['delay_index'], weights=x['pass_flow'] + 1e-5)
            })
        ).reset_index()
        plot_data_dict['全路口整体汇总'] = total_data
        
    return plot_data_dict

def plot_9_grid(data_dict, x_col, y_col, title, x_label, y_label, save_path, color_col=None):
    """
    绘制 3x3 九宫格散点图并保存到本地
    """
    if not data_dict:
        print(f"❌ 没有提取到有效数据，无法生成图表: {title}")
        return

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.95)
    
    axes = axes.flatten()
    
    # 固定排版顺序
    ordered_keys = [
        "东向 左转", "东向 直行", 
        "南向 左转", "南向 直行", 
        "西向 左转", "西向 直行", 
        "北向 左转", "北向 直行", 
        "全路口整体汇总"
    ]
    
    items = [(k, data_dict[k]) for k in ordered_keys if k in data_dict]
    
    for i, ax in enumerate(axes):
        if i < len(items):
            name, subset = items[i]
            
            if color_col and color_col in subset.columns:
                sc = ax.scatter(
                    subset[x_col], subset[y_col], 
                    c=subset[color_col], cmap='coolwarm', 
                    alpha=0.6, edgecolors='none', s=20
                )
                if i == len(items) - 1:
                    plt.colorbar(sc, ax=ax, label='延误指数')
            else:
                ax.scatter(subset[x_col], subset[y_col], alpha=0.5, color='#1f77b4', s=20)
            
            if not subset.empty and len(subset) > 10:
                c_cap = subset[y_col].quantile(0.95)
                ax.axhline(y=c_cap, color='red', linestyle='--', alpha=0.7, label=f'95% 流量: {c_cap:.0f}')
                ax.legend(loc='lower right', fontsize=9)
            
            ax.set_title(name, fontsize=14)
            ax.set_xlabel(x_label, fontsize=11)
            ax.set_ylabel(y_label, fontsize=11)
            ax.grid(True, linestyle=':', alpha=0.6)
        else:
            ax.set_visible(False)
            
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ 九宫格图表已保存至: {save_path}")

def plot_single(df, x_col, y_col, title, x_label, y_label, save_path, color_col=None):
    """
    单独绘制一张大图并保存到本地
    """
    if df is None or df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    
    if color_col and color_col in df.columns:
        sc = ax.scatter(
            df[x_col], df[y_col], 
            c=df[color_col], cmap='coolwarm', 
            alpha=0.7, edgecolors='none', s=40
        )
        plt.colorbar(sc, ax=ax, label='延误指数')
    else:
        ax.scatter(df[x_col], df[y_col], alpha=0.6, color='#1f77b4', s=40)
    
    if len(df) > 10:
        c_cap = df[y_col].quantile(0.95)
        ax.axhline(y=c_cap, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'95% 流量平台: {c_cap:.0f}')
        ax.legend(loc='lower right', fontsize=12)
    
    ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ 独立总图已保存至: {save_path}")

# ==========================================
# 主执行逻辑
# ==========================================
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_pattern = os.path.join(current_dir, "dwd_gaode_inter_intersection_direction_index_view*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"❌ 未找到文件！请确保 CSV 文件与代码保存在同一文件夹下：\n{current_dir}")
    else:
        csv_files.sort(reverse=True)
        target_csv = csv_files[0] 
        
        print(f"✅ 在当前文件夹共找到 {len(csv_files)} 个匹配的 CSV 文件。")
        print(f"📂 自动为您选择读取最新的文件: {os.path.basename(target_csv)}")
        
        try:
            df = pd.read_csv(target_csv)
            print(f"📊 数据加载成功，共 {len(df)} 行数据。")
            
            # ==========================================
            # 【新增配置】自定义研究时段过滤
            # ==========================================
            start_time = "2026-04-15 00:00:00"  # 请在此输入起始时间
            end_time = "2026-04-15 23:59:59"    # 请在此输入结束时间
            
            # 转换时间格式并过滤数据
            if 'create_time' in df.columns:
                df['create_time'] = pd.to_datetime(df['create_time'])
                df = df[(df['create_time'] >= pd.to_datetime(start_time)) & (df['create_time'] <= pd.to_datetime(end_time))]
                print(f"⏳ 已根据自定义时段 ({start_time} 至 {end_time}) 过滤数据，剩余 {len(df)} 行。")
            else:
                print("⚠️ 警告：未找到 'create_time' 列，时间过滤未生效。")
            # ==========================================
            
            # 【重要配置】目标交叉口 ID
            target_inter_id = '6caa3ec15b566e' 
            
            inter_name = INTER_NAME_MAP.get(target_inter_id, target_inter_id)
            
            if target_inter_id not in df['inter_id'].unique():
                print(f"⚠️ 警告：在数据集中未找到路口 {target_inter_id} ({inter_name})。")
            else:
                print(f"🔍 正在提取路口 {inter_name} 的各转向数据...")
                
                output_folder = os.path.join(current_dir, f"路口状态图-{inter_name}")
                os.makedirs(output_folder, exist_ok=True)
                
                plot_data = prepare_data(df, inter_id=target_inter_id)
                
                # 第一部分：生成九宫格图表
                print("\n📈 [1/2] 正在生成【九宫格演变图】...")
                plot_9_grid(
                    data_dict=plot_data, 
                    x_col='queue_len_max', 
                    y_col='pass_flow', 
                    title=f'{inter_name} 流率-排队空间演变图 (各转向)',
                    x_label='最大排队长度 (queue_len_max) [米]',
                    y_label='5min车流量 (pass_flow) [辆]',
                    save_path=os.path.join(output_folder, f"{inter_name}_九宫格_流率_排队.png")
                )

                plot_9_grid(
                    data_dict=plot_data, 
                    x_col='stop_time', 
                    y_col='pass_flow', 
                    title=f'{inter_name} 动态阻抗图 (停车时间-流量) (各转向)',
                    x_label='车均停车时间 (stop_time) [秒]',
                    y_label='5min车流量 (pass_flow) [辆]',
                    save_path=os.path.join(output_folder, f"{inter_name}_九宫格_动态阻抗.png"),
                    color_col='delay_index' 
                )
                
                # 第二部分：生成独立的全路口总图
                total_df = plot_data.get("全路口整体汇总")
                if total_df is not None and not total_df.empty:
                    print("\n📈 [2/2] 正在单独生成【全路口汇总大图】...")
                    plot_single(
                        df=total_df, 
                        x_col='queue_len_max', 
                        y_col='pass_flow', 
                        title=f'{inter_name} 全路口整体：流率-排队空间演变图',
                        x_label='全路口最大排队长度 (queue_len_max) [米]',
                        y_label='全路口5min车流量 (pass_flow) [辆]',
                        save_path=os.path.join(output_folder, f"{inter_name}_独立总图_流率_排队.png")
                    )

                    plot_single(
                        df=total_df, 
                        x_col='stop_time', 
                        y_col='pass_flow', 
                        title=f'{inter_name} 全路口整体：动态阻抗图 (停车时间-流量)',
                        x_label='全路口车均停车时间 (stop_time) [秒]',
                        y_label='全路口5min车流量 (pass_flow) [辆]',
                        save_path=os.path.join(output_folder, f"{inter_name}_独立总图_动态阻抗.png"),
                        color_col='delay_index'
                    )
                
                print(f"\n✨ 全部运行完毕！请在 '{output_folder}' 文件夹中查看结果。")
                
        except Exception as e:
            print(f"❌ 读取数据或处理时发生错误: {e}")