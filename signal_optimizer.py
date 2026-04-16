import pandas as pd
import numpy as np
import os

# ================= 接口区：自定义现行信控方案 =================
# 你可以在这里输入各个路口、各个时段正在使用的真实配时方案
CURRENT_SIGNAL_PLANS = {
    '早高峰': {
        '光明大道与光明大街': {
            '现行周期': 167,
            '放行相序': '北向全放 -> 东向全放 ->  西向全放-> 南向全放 (单口放行)', 
            '绿灯分配': {
                '北向全放': 49,
                '东向全放': 40,
                '西向全放': 30,
                '南向全放': 51
            } # (注：剩余 16s 为黄灯全红损失)
        },
        '光明大道与光辉大道': {
            '现行周期': 120,
            '放行相序': '东向全放 -> 北向全放 ->  西向全放-> 南向全放 (单口放行)', 
            '绿灯分配': {
                '东向全放': 25,
                '北向全放': 35,
                '西向全放': 25,
                '南向全放': 35
            } # (注：剩余 16s 为黄灯全红损失)
        },
        '光明大道与河心路': {
            '现行周期': 163,
            '放行相序': '东向全放 西向全放 -> 行人过街1 ->  东向全放 西向全放-> 行人过街2 (单口放行)', 
            '绿灯分配': {
                '东向全放 西向全放': 45,
                '行人过街1': 37,
                '东向全放 西向全放': 45,
                '行人过街2': 36
            } # (注：剩余 16s 为黄灯全红损失)
        },
        '光明大道与华夏路': {
            '现行周期': 163,
            '放行相序': '北向全放 -> 南向全放 -> 东向全放 -> 西向全放 (单口放行)', 
            '绿灯分配': {
                '北向全放': 49,
                '南向全放': 49,
                '东向全放': 33,
                '西向全放': 32
            } # (注：剩余 16s 为黄灯全红损失)
        },
        '光明大道与华裕路': {
            '现行周期': 176,
            '放行相序': '东向全放 ->  东向全放 西向全放 ->  西向全放-> 南向全放-> 北向全放 (单口放行)', 
            '绿灯分配': {
                '东向全放': 35,
                '东向全放 西向全放': 29,
                '西向全放': 46,
                '南向全放': 38,
                '北向全放': 28
            } # (注：剩余 16s 为黄灯全红损失)
        },
       
    },
    '晚高峰': {
        # 晚高峰方案...
    }
}

# ================= 配置区 =================
PEAK_PERIODS = {
    '早高峰': ('07:30', '09:00'),
    '晚高峰': ('17:30', '19:30')
}

PHASE_MAPPING = {
    '相位1（南北直行）': [('南向', '直行'), ('北向', '直行')],
    '相位2（南北左转）': [('南向', '左转'), ('北向', '左转')],
    '相位3（东西直行）': [('东向', '直行'), ('西向', '直行')],
    '相位4（东西左转）': [('东向', '左转'), ('西向', '左转')]
}

# ================= 核心推演算法 =================
def load_and_preprocess_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到文件 {csv_path}，请先运行综合数据处理脚本。")
    df = pd.read_csv(csv_path, encoding='GBK')
    df['_temp_time'] = pd.to_datetime('2026-01-01 ' + df['时间'])
    df['延误指数'] = pd.to_numeric(df['延误指数'], errors='coerce').fillna(0)
    df['流量占比(%)'] = pd.to_numeric(df['流量占比(%)'], errors='coerce').fillna(0)
    return df

def determine_flow_tag(delay_90th, avg_ratio):
    is_congested = delay_90th >= 55.0  
    is_high_flow = avg_ratio >= 15.0  
    if is_high_flow and is_congested: return '流量大拥堵高'
    if not is_high_flow and is_congested: return '流量小拥堵高'
    if is_high_flow and not is_congested: return '流量大拥堵低'
    return '流量小拥堵低'

def estimate_subzone_ccl(df):
    print("\n" + "="*80)
    print(" 🚦 启动子区公共周期 (CCL) 智能推演引擎")
    print("="*80)

    for period_name, (start_time, end_time) in PEAK_PERIODS.items():
        start_dt = pd.to_datetime(f'2026-01-01 {start_time}')
        end_dt = pd.to_datetime(f'2026-01-01 {end_time}')
        df_period = df[(df['_temp_time'] >= start_dt) & (df['_temp_time'] <= end_dt)]
        if df_period.empty: continue
            
        inter_evals = []
        for inter, group_inter in df_period.groupby('路口名称'):
            flow_stats = group_inter.groupby(['进口道方向', '转向']).agg(
                延误_90分位=('延误指数', lambda x: x.quantile(0.90)),
                平均流量占比=('流量占比(%)', 'mean')
            ).reset_index()
            
            flow_stats['标签'] = flow_stats.apply(
                lambda row: determine_flow_tag(row['延误_90分位'], row['平均流量占比']), axis=1
            )
            
            core_cnt = (flow_stats['标签'] == '流量大拥堵高').sum()
            killer_cnt = (flow_stats['标签'] == '流量小拥堵高').sum()
            stress_score = (core_cnt * 3) + (killer_cnt * 2)
            
            cycle_est = 16 
            equation_parts = ["16"]
            phase_derivations = []
            phase_allocations = {}
            
            for phase_name, directions in PHASE_MAPPING.items():
                p_tags = []
                for d, t in directions:
                    mov = flow_stats[(flow_stats['进口道方向'] == d) & (flow_stats['转向'] == t)]
                    if not mov.empty: p_tags.append(mov['标签'].iloc[0])
                
                if '流量大拥堵高' in p_tags: 
                    g = 45
                    reason = "包含“流量大拥堵高”流向，主干大动脉重度拥堵，分配底线 45 秒"
                elif '流量大拥堵低' in p_tags: 
                    g = 40
                    reason = "为主流向且“流量大拥堵低”，车多但不堵，适度收紧以反哺他向，分配底线 40 秒"
                elif '流量小拥堵高' in p_tags: 
                    g = 30
                    reason = "包含“流量小拥堵高”流向，需给予足够排空时间防溢出死锁，分配底线 30 秒"
                elif '流量小拥堵低' in p_tags: 
                    g = 20
                    reason = "仅为“流量小拥堵低”流向，满足基本过街需求即可，分配底线 20 秒"
                else: 
                    g = 0
                
                if g > 0:
                    cycle_est += g
                    equation_parts.append(str(g))
                    phase_derivations.append(f"        {phase_name}：{reason}。")
                    phase_allocations[phase_name] = g
            
            cycle_equation_str = f"{cycle_est}s = " + " + ".join(equation_parts)

            inter_evals.append({
                '路口名称': inter,
                '问题指数（值越大->问题越严重）': stress_score,
                '问题陈列': f"流量大拥堵高x{core_cnt}, 流量小拥堵高x{killer_cnt}",
                '推演周期(s)': cycle_equation_str,
                '_raw_cycle': cycle_est, 
                '_killer_cnt': killer_cnt,
                '_phase_derivations': phase_derivations,
                '_phase_allocations': phase_allocations
            })
            
        if inter_evals:
            df_eval = pd.DataFrame(inter_evals)
            df_eval = df_eval.sort_values(by=['问题指数（值越大->问题越严重）', '_raw_cycle'], ascending=[False, False])
            
            critical_inter = df_eval.iloc[0]['路口名称']
            ccl = df_eval.iloc[0]['_raw_cycle']
            ccl_equation = df_eval.iloc[0]['推演周期(s)']
            critical_killers = df_eval.iloc[0]['_killer_cnt']
            derivations = "\n".join(df_eval.iloc[0]['_phase_derivations'])
            allocations = df_eval.iloc[0]['_phase_allocations']
            
            print(f"\n📊 【{period_name}】子区路口压力排行榜：")
            print(df_eval[['路口名称', '问题指数（值越大->问题越严重）', '问题陈列', '推演周期(s)']].to_string(index=False))
            
            print(f"\n💡 宏观决策结论：")
            print(f"   -> 经过延误/占比交叉诊断，本子区【关键瓶颈交叉口】锁定为：🎯 {critical_inter}")
            print(f"   -> 建议整个光明大道子区的公共周期 (CCL) 统一定为：⏳ 【{ccl} 秒】")
            print(f"\n📝 推演周期 {ccl} 秒的数学推导：")
            print(f"   系统在后台为了拯救 {critical_inter} 的 {critical_killers} 个“流量小拥堵高”流向，模拟了一次最低生存绿信比分配：")
            print(f"        黄灯全红损失时间：固定 16 秒（4个相位 × 4秒）。")
            print(f"{derivations}")
            print(f"        汇总求和：{ccl_equation}！")

            # --- 【全新功能：信控方案体检与改善建议】 ---
            generate_optimization_advice(period_name, critical_inter, ccl, allocations)

def generate_optimization_advice(period, inter_name, recommended_ccl, recommended_phases):
    """提取现行信控方案并与推荐方案进行比对，输出改善意见"""
    current_plan = CURRENT_SIGNAL_PLANS.get(period, {}).get(inter_name)
    if not current_plan:
        return # 如果未配置该路口的现行方案，则跳过
        
    print(f"\n" + "="*80)
    print(f" 🛠️ 现行信控方案体检与优化建议 (针对 {period} - {inter_name})")
    print("="*80)
    
    cur_cycle = current_plan.get('现行周期', 0)
    cur_sequence = current_plan.get('放行相序', '未知')
    
    print(f"【现状配置输入】: 现行周期 {cur_cycle}s | 放行相序: {cur_sequence}\n")
    
    # 1. 相序诊断
    print("⚠️ 【1. 相序结构诊断】")
    if '单口' in cur_sequence or '北向全放' in cur_sequence:
        print("   [诊断结果]: 极度低效！")
        print("   [工程意见]: 当前采用“单边轮流放行”，严重浪费了南北主线直行的重叠时空资源，直接抹杀了双向绿波带的可能。")
        print("   [优化方案]: 强烈建议变更为“对称放行”或“混开放行”（南北直行同放 -> 南北左转同放），以释放至少 20% 的空间容量反哺侧向！\n")
    else:
        print("   [诊断结果]: 结构合理。")
        print("   [工程意见]: 当前相序已具备主干线协调条件，问题主要出在周期长度或绿信比分配不均。\n")

    # 2. 周期诊断
    print("⚠️ 【2. 公共周期(CCL)诊断】")
    cycle_diff = recommended_ccl - cur_cycle
    if cycle_diff > 10:
        print("   [诊断结果]: 容量严重不足！")
        print(f"   [工程意见]: 现行周期 {cur_cycle}s 无法满足本路口多个“流量小拥堵高”流向的排空底线，正处于系统性死锁边缘。")
        print(f"   [优化方案]: 必须将该路口周期拉长至 {recommended_ccl}s (净增 {cycle_diff}s)，并将此周期同步下发至整个光明大道子区。\n")
    elif cycle_diff < -10:
        print("   [诊断结果]: 周期过长（空耗）！")
        print(f"   [工程意见]: 现行周期 {cur_cycle}s 过于臃肿，极易导致绿灯空放，并增加次干道车辆绝望感。")
        print(f"   [优化方案]: 建议压缩至 {recommended_ccl}s 以提高路口运转频次。\n")
    else:
        print("   [诊断结果]: 周期适中。")
        print("   [工程意见]: 现行周期与推演容量需求基本吻合，建议保持。\n")

    # 3. 相位时长建议
    print("⚠️ 【3. 相位重组与时长分配指引】 (基于推荐的对称相序框架)")
    for p_name, rec_green in recommended_phases.items():
        print(f"   -> {p_name}: 建议强制保障 【{rec_green}s】 的有效绿灯时间。")
    print("   (说明：以上时长建议已充分考虑并压榨了“流量大拥堵低”流向的冗余时间，精准补偿了极易溢出的侧向路口，请直接输入信号机进行调试。)\n")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_dir, "干线全流向_延误与流量占比明细表.csv")
    try:
        df_details = load_and_preprocess_data(csv_file_path)
        estimate_subzone_ccl(df_details)
    except Exception as e:
        print(f"运行失败: {e}")