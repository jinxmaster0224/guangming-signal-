#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ==========================================
# 🎨 全局图表设置
# ==========================================
# 设置中文字体，防止图表出现方块 (Windows/Mac通用设置)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 基础常量映射
CARDINAL_HANZI = {"E": "东向", "W": "西向", "S": "南向", "N": "北向"}
TURN_LABEL_CN = {0: "聚合", 1: "左转", 2: "直行", 3: "右转"}
_WKT_COORD_PATTERN = re.compile(r'(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)')

# ==========================================
# ⚙️ 配置区 1：时间段自定义接口
# ==========================================
def get_date_ranges():
    """
    获取预设的日期范围。
    """
    time_periods = [ 
         (True, '2026-03-11 00:00:00', '2026-03-11 23:59:59'),  # 时间段1 (目前已启用)
         (False, '2026-03-09 00:00:00', '2026-03-09 23:59:59'), # 时间段2 
         (False, '2026-03-16 00:00:00', '2026-03-16 23:59:59'), # 时间段3 
         (False, '2026-03-23 00:00:00', '2026-03-23 23:59:59'), # 时间段4 
         (False, '2026-03-29 00:00:00', '2026-03-29 23:59:59')  # 时间段5 
     ]
    
    date_ranges = []
    for i, (enable, start_date, end_date) in enumerate(time_periods, 1):
        if enable:
            try:
                start_time = pd.Timestamp(start_date)
                end_time = pd.Timestamp(end_date)
                date_ranges.append((start_time, end_time))
                print(f"[INFO] 已启用时间段{i}：{start_date} 到 {end_date}")
            except Exception as e:
                print(f"[错误] 时间段{i}日期格式错误：{e}")
    
    if not date_ranges:
        print("[INFO] 未启用任何时间段，将使用全量数据")
        date_ranges.append(None)
    
    return date_ranges

# ==========================================
# 🧠 核心算法模块
# ==========================================
def extract_direction_from_coords(lng_lat_seq: str, offset_degree: float = 0.0) -> str:
    """根据首尾坐标点计算宏观行驶方向"""
    if not isinstance(lng_lat_seq, str) or not str(lng_lat_seq).strip():
        return None

    matches = _WKT_COORD_PATTERN.findall(str(lng_lat_seq))
    if len(matches) < 2:
        return None

    try:
        points = [(float(x), float(y)) for x, y in matches]
    except ValueError:
        return None

    x0, y0 = points[0]
    x1, y1 = points[-1]
    dx = x1 - x0
    dy = y1 - y0
    
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None

    # 计算原始航向角
    angle_deg = math.degrees(math.atan2(dy, dx))
    
    # 减去路口偏转角，在数学上把斜交路口“掰正”
    angle_deg = angle_deg - offset_degree
    
    # 修正角度溢出
    if angle_deg > 180:
        angle_deg -= 360
    elif angle_deg <= -180:
        angle_deg += 360

    # 标准正交切分判断
    if -45.0 <= angle_deg < 45.0:
        return "E"
    if 45.0 <= angle_deg < 135.0:
        return "N"
    if -135.0 <= angle_deg < -45.0:
        return "S"
    return "W"

def _mode_valid(series: pd.Series) -> str:
    """返回frid内出现频率最高的方向（众数），防止单车轨迹漂移"""
    s = series.dropna()
    if s.empty:
        return None
    return s.mode().iloc[0]

def enrich_direction_features(df: pd.DataFrame, offset_degree: float = 0.0) -> pd.DataFrame:
    """为数据打上统一的进口道方向标签"""
    f = df.copy()
    f["_raw_direction"] = f["lng_lat_seq"].apply(lambda x: extract_direction_from_coords(x, offset_degree))
    
    if "frid" in f.columns:
        f["main_direction"] = f.groupby("frid")["_raw_direction"].transform(_mode_valid)
    else:
        f["main_direction"] = f["_raw_direction"]
    
    f.drop(columns=["_raw_direction"], inplace=True)
    return f

# ==========================================
# 📊 统计与可视化模块
# ==========================================
def analyze_and_plot(df: pd.DataFrame, out_img: str, offset_degree: float):
    """统计流量并画图"""
    if df.empty:
        print("[错误] 数据为空，无法分析。")
        return

    # 剔除缺失值
    df_clean = df.dropna(subset=["main_direction", "turn_dir_no"]).copy()
    df_clean["turn_dir_no"] = pd.to_numeric(df_clean["turn_dir_no"], errors='coerce').fillna(0).astype(int)
    df_clean["pass_flow"] = pd.to_numeric(df_clean["pass_flow"], errors='coerce').fillna(0)
    
    # 【核心过滤】：只保留明确有转向行为（1,2,3）的车辆，丢弃聚合(0)
    df_clean = df_clean[df_clean["turn_dir_no"] > 0]

    # 统计流量 (按方向和转向分组累加 pass_flow)
    flow_stats = df_clean.groupby(["main_direction", "turn_dir_no"])["pass_flow"].sum().reset_index(name="volume")
    
    if flow_stats.empty:
        print("[警告] 过滤聚合数据后，没有找到任何明确的转向流量数据！")
        return

    # 生成中文标签
    flow_stats["direction_cn"] = flow_stats["main_direction"].map(CARDINAL_HANZI)
    flow_stats["turn_cn"] = flow_stats["turn_dir_no"].map(TURN_LABEL_CN)
    flow_stats["label"] = flow_stats["direction_cn"] + flow_stats["turn_cn"]
    
    # 排序
    dir_order = {"E": 1, "S": 2, "W": 3, "N": 4}
    flow_stats["_order"] = flow_stats["main_direction"].map(dir_order) * 10 + flow_stats["turn_dir_no"]
    flow_stats = flow_stats.sort_values("_order").drop(columns=["_order"])

    # 控制台打印核对
    print("\n=== 路口各转向真实流量统计 ===")
    print(flow_stats[["main_direction", "turn_dir_no", "label", "volume"]].to_string(index=False))

    # 画图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(flow_stats["label"], flow_stats["volume"], color='#4C72B0', edgecolor='black', alpha=0.8)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + yval*0.01, f"{int(yval):,}", ha='center', va='bottom', fontsize=11)

    # 图表标题会动态显示您设定的偏转角度，方便排错
    plt.title(f"路口转向流量统计 (坐标系偏转: {offset_degree}°)", fontsize=16, pad=15)
    plt.xlabel("转向情况", fontsize=12)
    plt.ylabel("流量 (辆)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(out_img, dpi=300)
    print(f"\n[INFO] 流量柱状图已生成并保存至: {out_img}")
    plt.show() # 运行完毕直接在编辑器/系统里弹窗显示图片


if __name__ == "__main__":
    # ========================================================
    # ⚙️ 配置区 2：路口偏转角自定义接口
    # ========================================================
    # 【修改指南】：
    # - 正东南西北路口：填 0.0
    # - 东北/西南走向的路口（顺时针偏）：填正数，如 45.0， 30.5
    # - 西北/东南走向的路口（逆时针偏）：填负数，如 -45.0, -15.0
    # ========================================================
    OFFSET_DEGREE = 0.0  
    # ========================================================

    current_dir = Path(__file__).resolve().parent
    print(f"\n[INFO] 启动流向分析，设定的路口偏转角为: {OFFSET_DEGREE}°")
    print(f"[INFO] 正在扫描目录: {current_dir}")

    csv_files = list(current_dir.glob("*.csv"))
    
    if not csv_files:
        print("[错误] 在当前文件夹下没有找到任何 CSV 数据文件！")
    else:
        # 优先寻找包含 "ods_gaode" 的原始文件，找不到就用第一个 CSV
        input_csv = next((f for f in csv_files if "ods_gaode" in f.name), csv_files[0])
        output_img = current_dir / "flow_chart.png"
        
        print(f"[INFO] 找到目标数据: {input_csv.name}")
        
        try:
            date_ranges = get_date_ranges()
            
            print("[INFO] 正在读取数据...")
            usecols = ["frid", "lng_lat_seq", "turn_dir_no", "pass_flow", "create_time"]
            df_raw = pd.read_csv(input_csv, sep=None, engine="python", usecols=usecols)
            print(f"[INFO] 数据读取完成，共 {len(df_raw)} 条记录")
            
            # 日期过滤
            df_raw['create_time'] = pd.to_datetime(df_raw['create_time'])
            if date_ranges and date_ranges[0] is not None:
                mask = False
                for start_time, end_time in date_ranges:
                    mask |= (df_raw['create_time'] >= start_time) & (df_raw['create_time'] <= end_time)
                df_raw = df_raw[mask]
                print(f"[INFO] 日期过滤完成，剩余 {len(df_raw)} 条记录")
            
            print("[INFO] 正在提取进口道方向并进行容错处理...")
            # 传入您在上面设定的偏转角
            df_processed = enrich_direction_features(df_raw, offset_degree=OFFSET_DEGREE)
            
            print("[INFO] 正在进行流向汇总与可视化...")
            analyze_and_plot(df_processed, str(output_img), OFFSET_DEGREE)
            print("[INFO] 全部流程执行完毕！")
            
        except Exception as e:
            print(f"[错误] 程序运行出错: {e}")
            import traceback
            traceback.print_exc()