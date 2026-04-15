import pandas as pd
import numpy as np
import os

# ================= 配置区 =================
# 定义早晚高峰的时间段（用于从明细表中提取关键时刻）
PEAK_PERIODS = {
    '早高峰': ('07:30', '09:00'),
    '晚高峰': ('17:30', '19:30')
}

# 物理相位映射关系 (假设标准四相位结构)
PHASE_MAPPING = {
    '1. 南北直行': [('南向', '直行'), ('北向', '直行')],
    '2. 南北左转': [('南向', '左转'), ('北向', '左转')],
    '3. 东西直行': [('东向', '直行'), ('西向', '直行')],
    '4. 东西左转': [('东向', '左转'), ('西向', '左转')]
}

# ================= 核心推演算法 =================
def load_and_preprocess_data(csv_path):
    """读取明细表，并清洗数据"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到文件 {csv_path}，请先运行综合数据处理脚本。")
    
    df = pd.read_csv(csv_path, encoding='GBK')
    
    # 将时间字符串转换为 datetime 以便进行区间过滤
    # 因为原表里只有 'HH:MM'，我们随便拼一个日期方便计算时间差
    df['_temp_time'] = pd.to_datetime('2026-01-01 ' + df['时间'])
    
    # 确保数值列的数据类型
    df['延误指数'] = pd.to_numeric(df['延误指数'], errors='coerce').fillna(0)
    df['流量占比(%)'] = pd.to_numeric(df['流量占比(%)'], errors='coerce').fillna(0)
    
    return df

def determine_flow_tag(delay_90th, avg_ratio):
    """
    【全新规则】：基于 HCM 绝对延误(秒)与流量占比进行打标
    拥堵红线：90%分位延误 > 55秒 (进入 E 级甚至 F 级)
    主力红线：平均流量占比 >= 15%
    """
    is_congested = delay_90th >= 55.0  
    is_high_flow = avg_ratio >= 15.0  
    
    if is_high_flow and is_congested: return '🔥核心保卫'
    if not is_high_flow and is_congested: return '☠️隐形杀手'
    if is_high_flow and not is_congested: return '🌊虚假主流'
    return '✅常规流向'

def estimate_subzone_ccl(df):
    """提取高峰期切片，推演关键交叉口与公共周期"""
    print("\n" + "="*55)
    print(" 启动子区公共周期 (CCL) 智能推演")
    print("="*55)

    for period_name, (start_time, end_time) in PEAK_PERIODS.items():
        # 1. 过滤当前高峰期的数据
        start_dt = pd.to_datetime(f'2026-01-01 {start_time}')
        end_dt = pd.to_datetime(f'2026-01-01 {end_time}')
        df_period = df[(df['_temp_time'] >= start_dt) & (df['_temp_time'] <= end_dt)]
        
        if df_period.empty:
            continue
            
        inter_evals = []
        
        # 2. 以路口为单位进行病情评估
        for inter, group_inter in df_period.groupby('路口名称'):
            # 计算该路口 8 个流向在高峰期的聚合特征
            flow_stats = group_inter.groupby(['进口道方向', '转向']).agg(
                延误_90分位=('延误指数', lambda x: x.quantile(0.90)),
                平均流量占比=('流量占比(%)', 'mean')
            ).reset_index()
            
            # 打标签
            flow_stats['标签'] = flow_stats.apply(
                lambda row: determine_flow_tag(row['延误_90分位'], row['平均流量占比']), axis=1
            )
            
            # 计算路口“病情得分” (核心保卫=3分, 隐形杀手=2分)
            core_cnt = (flow_stats['标签'] == '🔥核心保卫').sum()
            killer_cnt = (flow_stats['标签'] == '☠️隐形杀手').sum()
            stress_score = (core_cnt * 3) + (killer_cnt * 2)
            
            # 3. 模拟相位推演，自下而上计算维持该路口生存的最小周期
            cycle_est = 16 # 起步固定损失时间 (4个相位 * 每个4秒)
            phase_details = {}
            
            for phase_name, directions in PHASE_MAPPING.items():
                p_tags = []
                for d, t in directions:
                    mov = flow_stats[(flow_stats['进口道方向'] == d) & (flow_stats['转向'] == t)]
                    if not mov.empty:
                        p_tags.append(mov['标签'].iloc[0])
                
                # 决定当前相位的绿灯供给
                if '🔥核心保卫' in p_tags: g = 45   # 大动脉严重拥堵，必须给足放行时间
                elif '🌊虚假主流' in p_tags: g = 40 # 流量极大但不怎么堵，稍微收紧绿灯
                elif '☠️隐形杀手' in p_tags: g = 30 # 虽然车少但是极其拥堵，必须给够时间排空
                elif '✅常规流向' in p_tags: g = 20 # 没什么存在感，给个行人过街保底绿灯
                else: g = 0 # 没车
                
                if g > 0:
                    cycle_est += g
                    phase_details[phase_name] = f"{g}s"
            
            inter_evals.append({
                '路口名称': inter,
                '病情得分': stress_score,
                '核心矛盾': f"核心保卫x{core_cnt}, 杀手x{killer_cnt}",
                '推演周期(s)': cycle_est,
                '相位预分配基准': phase_details
            })
            
        # 4. 输出该时段的战报
        if inter_evals:
            df_eval = pd.DataFrame(inter_evals)
            # 按病情得分和推演周期降序排列，排在第一的就是“木桶短板”
            df_eval = df_eval.sort_values(by=['病情得分', '推演周期(s)'], ascending=[False, False])
            
            critical_inter = df_eval.iloc[0]['路口名称']
            ccl = df_eval.iloc[0]['推演周期(s)']
            
            print(f"\n📊 【{period_name}】子区各路口问题分析：")
            # 格式化打印大表
            print(df_eval[['路口名称', '问题指数（值越大->问题越严重）', '问题陈列', '推演周期(s)']].to_string(index=False))
            
            print(f"\n💡 决策结论：")
            print(f"   -> 经过延误/占比交叉诊断，本子区【关键瓶颈交叉口】锁定为：🎯 {critical_inter}")
            print(f"   -> 建议整个光明大道子区的公共周期 (CCL) 统一定为：⏳ 【{ccl} 秒】")
            print(f"   -> 关键路口 {critical_inter} 的绿信比安全分配底线：")
            for p, g in df_eval.iloc[0]['相位预分配基准'].items():
                print(f"        {p} : {g}")

if __name__ == "__main__":
    # 自动获取当前脚本所在的目录绝对路径 (例如 E:/深城交数据/上游子区分析)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 拼接出 CSV 文件的绝对路径
    csv_file_path = os.path.join(script_dir, "干线全流向_延误与流量占比明细表.csv")
    
    print(f"📂 尝试读取的数据表路径: {csv_file_path}")
    
    try:
        df_details = load_and_preprocess_data(csv_file_path)
        estimate_subzone_ccl(df_details)
    except Exception as e:
        print(f"运行失败: {e}")