import pandas as pd
import numpy as np
import os

# ================= 接口区：自定义现行信控方案 =================
CURRENT_SIGNAL_PLANS = {
    '早高峰': {
        '光明大道与光明大街': {
            '现行周期': 167,
            '放行相序': '北向全放 -> 东向全放 ->  西向全放-> 南向全放 (单口放行)'
        },
        '光明大道与光辉大道': {
            '现行周期': 120,
            '放行相序': '东向全放 -> 北向全放 ->  西向全放-> 南向全放 (单口放行)'
        },
        '光明大道与河心路': {
            '现行周期': 163,
            '放行相序': '东向全放 西向全放 -> 行人过街1 ->  东向全放 西向全放-> 行人过街2 (单口放行)'
        },
        '光明大道与华夏路': {
            '现行周期': 163,
            '放行相序': '北向全放 -> 南向全放 -> 东向全放 -> 西向全放 (单口放行)'
        },
        '光明大道与华裕路': {
            '现行周期': 176,
            '放行相序': '东向全放 ->  东向全放 西向全放 ->  西向全放-> 南向全放-> 北向全放 (单口放行)'
        },
    },
    '晚高峰': {}
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
    
    if 'pass_flow' in df.columns:
        df['pass_flow'] = pd.to_numeric(df['pass_flow'], errors='coerce').fillna(0)
    else:
        df['pass_flow'] = 0

    # 严格过滤掉“聚合”行，避免干扰最大值计算
    df = df[~df['进口道方向'].astype(str).str.contains('聚合')]
    df = df[~df['转向'].astype(str).str.contains('聚合')]

    # 计算历史最大分均流量 (用以近似替代饱和流率 S)
    hist_max_df = df.groupby(['路口名称', '进口道方向', '转向'])['pass_flow'].max().reset_index()
    hist_max_df.rename(columns={'pass_flow': '历史最大分均流量'}, inplace=True)
    
    # 合并回原表
    df = pd.merge(df, hist_max_df, on=['路口名称', '进口道方向', '转向'], how='left')
    
    return df

def modified_webster_cycle(Y_total, L_total):
    """
    修正的 Webster 最小延误法
    公式: C_0 = (alpha*L + beta) / (1 - theta*Y)
    加入双向反馈约束：上限触顶调 theta，下限触底调 beta
    """
    alpha = 1.5
    beta = 5.0
    theta = 0.90 # 初始阻尼系数

    # 防止极端拥堵时分母出现负数或 0
    if Y_total * theta >= 0.95:
        Y_total = 0.95 / theta

    C_0 = (alpha * L_total + beta) / (1 - theta * Y_total)

    # 现实周期约束调参
    MAX_CYCLE = 180.0
    MIN_CYCLE = 60.0

    if C_0 > MAX_CYCLE:
        # 触碰上限（过饱和）：反推调整阻尼 theta
        theta = (1 - (alpha * L_total + beta) / MAX_CYCLE) / Y_total
        C_0 = MAX_CYCLE
    elif C_0 < MIN_CYCLE:
        # 触碰下限（保护行人）：反推增加安全缓冲 beta
        beta_new = MIN_CYCLE * (1 - theta * Y_total) - (alpha * L_total)
        beta = round(max(beta_new, beta), 2)
        C_0 = MIN_CYCLE
        
    return int(round(C_0)), alpha, beta, round(theta, 3), Y_total

def estimate_subzone_ccl(df):
    print("\n" + "="*80)
    print(" 🚦 启动子区公共周期 (CCL) - 修正 Webster 精算引擎")
    print("="*80)

    for period_name, (start_time, end_time) in PEAK_PERIODS.items():
        start_dt = pd.to_datetime(f'2026-01-01 {start_time}')
        end_dt = pd.to_datetime(f'2026-01-01 {end_time}')
        df_period = df[(df['_temp_time'] >= start_dt) & (df['_temp_time'] <= end_dt)]
        if df_period.empty: continue
            
        inter_evals = []
        for inter, group_inter in df_period.groupby('路口名称'):
            # 聚合各流向统计指标（保留平均分均流量逻辑）
            flow_stats = group_inter.groupby(['进口道方向', '转向']).agg(
                延误_90分位=('延误指数', lambda x: x.quantile(0.90)),
                平均流量占比=('流量占比(%)', 'mean'),
                平均分均流量=('pass_flow', 'mean'),
                历史最大分均流量=('历史最大分均流量', 'first')
            ).reset_index()
            
            # --- 简化版：交叉口综合拥堵指数 ---
            # 采用直白逻辑：各个流向的 (α × 流量占比) + (β × 延误指数) 的总和
            # 设置权重 alpha_weight=0.4 (流量权重), beta_weight=0.6 (延误权重，代表延误对拥堵影响更剧烈)
            alpha_weight = 0.4
            beta_weight = 0.6
            flow_stats['流向综合得分'] = (alpha_weight * flow_stats['平均流量占比']) + (beta_weight * flow_stats['延误_90分位'])
            stress_score = flow_stats['流向综合得分'].sum()
            
            # 提取经验流量比 y_i 并记录
            phase_y_dict = {}
            phase_flow_details = {} 
            L_total = 16 # 假设黄闪+全红基础总损失为 16s
            
            for phase_name, directions in PHASE_MAPPING.items():
                max_y = 0.0
                best_detail = ""
                for d, t in directions:
                    mov = flow_stats[(flow_stats['进口道方向'] == d) & (flow_stats['转向'] == t)]
                    if not mov.empty:
                        cur_flow = mov['平均分均流量'].iloc[0]
                        max_flow = mov['历史最大分均流量'].iloc[0]
                        
                        y_val = cur_flow / max_flow if max_flow > 0 else 0.0
                            
                        if y_val >= max_y:
                            max_y = y_val
                            best_detail = f"{d}{t} (均量:{cur_flow:.1f}/最大:{max_flow:.1f} = y:{y_val:.3f})"
                
                phase_y_dict[phase_name] = max_y
                phase_flow_details[phase_name] = best_detail

            # 修正的 Webster 独立推演
            Y_sum = sum(phase_y_dict.values())
            ccl_est, alpha, beta, theta, Y_adj = modified_webster_cycle(Y_sum, L_total)

            inter_evals.append({
                '路口名称': inter,
                '综合拥堵指数': stress_score,
                '独立推演周期': ccl_est,
                '_Y_adj': Y_adj,
                '_alpha': alpha, '_beta': beta, '_theta': theta,
                '_L_total': L_total,
                '_phase_details': phase_flow_details
            })
            
        if inter_evals:
            df_eval = pd.DataFrame(inter_evals)
            # 排序：先比综合拥堵指数，打平比周期
            df_eval = df_eval.sort_values(by=['综合拥堵指数', '独立推演周期'], ascending=[False, False])
            
            critical_inter = df_eval.iloc[0]['路口名称']
            subzone_ccl = df_eval.iloc[0]['独立推演周期']
            Y_val = df_eval.iloc[0]['_Y_adj']
            alp = df_eval.iloc[0]['_alpha']
            bet = df_eval.iloc[0]['_beta']
            the = df_eval.iloc[0]['_theta']
            L_val = df_eval.iloc[0]['_L_total']
            details = df_eval.iloc[0]['_phase_details']
            cici_score = df_eval.iloc[0]['综合拥堵指数']
            
            print(f"\n📊 【{period_name}】子区路口压力与周期需求排行榜：")
            display_df = df_eval[['路口名称', '综合拥堵指数', '独立推演周期']].copy()
            display_df['综合拥堵指数'] = display_df['综合拥堵指数'].round(2)
            print(display_df.to_string(index=False))
            
            print(f"\n💡 宏观决策结论：")
            print(f"   -> 经过简化的直白加权评估，本子区【关键瓶颈交叉口】锁定为：🎯 {critical_inter}")
            print(f"      (入选理由：综合拥堵指数最高达 {cici_score:.2f})")
            print(f"   -> 以该瓶颈为基准，建议整个光明大道子区的公共周期 (CCL) 统一定为：⏳ 【{subzone_ccl} 秒】")
            
            print(f"\n📝 【{critical_inter}】修正 Webster 周期推导模型解析：")
            print(f"   计算公式：C_0 = (αL + β) / (1 - θY)")
            print(f"   [提取经验流量比 y (使用历史最大分均流量近似饱和流率)]：")
            for phase, info in details.items():
                print(f"     - {phase}: 取关键流向 {info}")
            print(f"   [参数代入与双向代数反馈约束]：")
            print(f"     - 交叉口经验流率比总和 (Y) = {Y_val:.3f}")
            print(f"     - 交叉口总损失时间 (L) = {L_val}s")
            print(f"     - 稳定控制系数：α = {alp}, β = {bet}, θ = {the}")
            print(f"       *(注:若β变大，说明在为行人过街逆向补帧; 若θ变小，说明在防过饱和周期爆炸)*")
            print(f"   [精算结果]：({alp} × {L_val} + {bet}) / (1 - {the} × {Y_val:.3f}) ≈ {subzone_ccl}s")

            # --- 方案体检与周期评价 ---
            generate_optimization_advice(period_name, critical_inter, subzone_ccl)

def generate_optimization_advice(period, inter_name, recommended_ccl):
    current_plan = CURRENT_SIGNAL_PLANS.get(period, {}).get(inter_name)
    if not current_plan: return 
        
    print(f"\n" + "="*80)
    print(f" 🛠️ 现行信控方案体检与优化建议 (针对 {period} - {inter_name})")
    print("="*80)
    
    cur_cycle = current_plan.get('现行周期', 0)
    cur_sequence = current_plan.get('放行相序', '未知')
    
    print(f"【现状配置输入】: 现行周期 {cur_cycle}s | 放行相序: {cur_sequence}\n")
    
    print("⚠️ 【1. 相序结构诊断】")
    if '单口' in cur_sequence or '北向全放' in cur_sequence:
        print("   [诊断结果]: 极度低效！")
        print("   [工程意见]: 当前采用“单边轮流放行”，严重浪费了交叉口重叠时空资源，直接抹杀了双向绿波带的可能。")
        print("   [优化方案]: 强烈建议变更为“对称放行”或“混开放行”（南北直行同放 -> 南北左转同放），以释放路口容量！\n")
    else:
        print("   [诊断结果]: 结构合理。")
        print("   [工程意见]: 当前相序已具备主干线协调条件。\n")

    print("⚠️ 【2. 公共周期(CCL)宏观诊断】")
    cycle_diff = recommended_ccl - cur_cycle
    if cycle_diff > 10:
        print("   [诊断结果]: 容量严重不足！")
        print(f"   [工程意见]: 现行周期 {cur_cycle}s 无法满足本路口当前关键流量比(Y)的承载需求，极易诱发排队激增。")
        print(f"   [优化方案]: 必须将该路口周期拉长至推荐的最佳周期 {recommended_ccl}s，并作为 CCL 同步下发至全线。\n")
    elif cycle_diff < -10:
        print("   [诊断结果]: 周期过长（空耗）！")
        print(f"   [工程意见]: 现行周期 {cur_cycle}s 超出了修正 Webster 的最优延误极值点，容易导致绿灯空放。")
        print(f"   [优化方案]: 建议压缩至推荐的最佳周期 {recommended_ccl}s 以提高路口运转频次。\n")
    else:
        print("   [诊断结果]: 周期适中。")
        print(f"   [工程意见]: 现行周期与修正 Webster 推演出的最佳周期 ({recommended_ccl}s) 基本吻合，建议保持。\n")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_dir, "干线全流向_延误与流量占比明细表.csv")
    try:
        df_details = load_and_preprocess_data(csv_file_path)
        estimate_subzone_ccl(df_details)
    except Exception as e:
        print(f"运行失败: {e}")