import pandas as pd
import numpy as np
import os
import re

# ================= 配置区 =================
# 自定义分析日期
TARGET_DATE = '2026-03-25'  

# 早高峰时段自定义
PEAK_PERIODS = {
    '早高峰': ('07:30', '09:00')
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
    
    # 暴力清洗列名
    df.columns = df.columns.str.strip()
    
    # 【修复核心1】：先从全量数据中，清洗 pass_flow 并找出真正的全局历史最大值！
    pass_flow_cols = [c for c in df.columns if 'pass_flow' in c.lower() or '分均流量' in c]
    if pass_flow_cols:
        p_col = pass_flow_cols[0]
        df['pass_flow'] = df[p_col].astype(str).str.replace(',', '', regex=False)
        df['pass_flow'] = df['pass_flow'].str.extract(r'(\d+\.?\d*)')[0]
        df['pass_flow'] = pd.to_numeric(df['pass_flow'], errors='coerce').fillna(0)
    else:
        df['pass_flow'] = 0

    df['进口道方向'] = df['进口道方向'].astype(str).str.strip()
    df['转向'] = df['转向'].astype(str).str.strip()

    # 过滤聚合行
    df_clean = df[~df['进口道方向'].str.contains('聚合')]
    df_clean = df_clean[~df_clean['转向'].str.contains('聚合')]

    # ---> 在全量、所有日期的数据集上，寻找真实的物理极限最大流率 <---
    hist_max_df = df_clean.groupby(['路口名称', '进口道方向', '转向'])['pass_flow'].max().reset_index()
    hist_max_df.rename(columns={'pass_flow': '历史最大分均流量'}, inplace=True)
    
    # 将真实的全局极值合并回原表
    df = pd.merge(df, hist_max_df, on=['路口名称', '进口道方向', '转向'], how='left')

    # 【修复核心2】：算完全局极值后，再来进行特定日期的切分过滤！
    if 'create_time' in df.columns:
        df['_parsed_time'] = pd.to_datetime(df['create_time'], errors='coerce')
        df['_date_str'] = df['_parsed_time'].dt.strftime('%Y-%m-%d')
        
        df_target = df[df['_date_str'] == TARGET_DATE].copy()
        if df_target.empty:
            available_dates = df['_date_str'].dropna().unique()
            raise ValueError(f"⚠️ 错误：未找到日期 {TARGET_DATE} 的数据！\n💡 你的CSV中实际包含的日期有: {available_dates}")
        
        df = df_target
        print(f"✅ 成功锁定目标日期：{TARGET_DATE}，共截取 {len(df)} 条数据。")
        df['_temp_time'] = pd.to_datetime('2026-01-01 ' + df['_parsed_time'].dt.strftime('%H:%M:%S'))
        
    elif '时间' in df.columns:
        df['_temp_time'] = pd.to_datetime('2026-01-01 ' + df['时间'].astype(str).str.strip())
    else:
        raise ValueError("⚠️ 致命错误：CSV 中无法找到时间列！")

    df['延误指数'] = pd.to_numeric(df['延误指数'], errors='coerce').fillna(0)
    df['流量占比(%)'] = pd.to_numeric(df['流量占比(%)'], errors='coerce').fillna(0)
    
    return df
    
    # 智能匹配 pass_flow 列名并使用正则暴力提取纯数字
    pass_flow_cols = [c for c in df.columns if 'pass_flow' in c.lower() or '分均流量' in c]
    if pass_flow_cols:
        p_col = pass_flow_cols[0]
        df['pass_flow'] = df[p_col].astype(str).str.replace(',', '', regex=False)
        df['pass_flow'] = df['pass_flow'].str.extract(r'(\d+\.?\d*)')[0]
        df['pass_flow'] = pd.to_numeric(df['pass_flow'], errors='coerce').fillna(0)
    else:
        print("⚠️ 警告：CSV中未找到 'pass_flow' 相关列，将启用降级估算模式！")
        df['pass_flow'] = 0

    # 清洗方向字符串，防止首尾空格导致匹配不到
    df['进口道方向'] = df['进口道方向'].astype(str).str.strip()
    df['转向'] = df['转向'].astype(str).str.strip()

    # 严格过滤掉“聚合”行
    df = df[~df['进口道方向'].str.contains('聚合')]
    df = df[~df['转向'].str.contains('聚合')]

    # 计算该日期的历史最大分均流量 (用以近似替代饱和流率 S)
    hist_max_df = df.groupby(['路口名称', '进口道方向', '转向'])['pass_flow'].max().reset_index()
    hist_max_df.rename(columns={'pass_flow': '历史最大分均流量'}, inplace=True)
    
    # 合并回原表
    df = pd.merge(df, hist_max_df, on=['路口名称', '进口道方向', '转向'], how='left')
    
    return df

def modified_webster_cycle(Y_total, L_total):
    """
    修正的 Webster 最小延误法
    """
    alpha = 1.5
    beta = 5.0
    theta = 0.90 # 初始阻尼系数

    if Y_total * theta >= 0.95:
        Y_total = 0.95 / theta

    C_0 = (alpha * L_total + beta) / (1 - theta * Y_total)

    MAX_CYCLE = 180.0
    MIN_CYCLE = 60.0

    if C_0 > MAX_CYCLE:
        theta = (1 - (alpha * L_total + beta) / MAX_CYCLE) / Y_total
        C_0 = MAX_CYCLE
    elif C_0 < MIN_CYCLE:
        beta_new = MIN_CYCLE * (1 - theta * Y_total) - (alpha * L_total)
        beta = round(max(beta_new, beta), 2)
        C_0 = MIN_CYCLE
        
    return int(round(C_0)), alpha, beta, round(theta, 3), Y_total

def estimate_subzone_ccl(df):
    print("\n" + "="*80)
    print(f" 子区公共周期- 使用修正的 Webster 计算方法 (日期: {TARGET_DATE})")
    print("="*80)

    for period_name, (start_time, end_time) in PEAK_PERIODS.items():
        start_dt = pd.to_datetime(f'2026-01-01 {start_time}')
        end_dt = pd.to_datetime(f'2026-01-01 {end_time}')
        df_period = df[(df['_temp_time'] >= start_dt) & (df['_temp_time'] <= end_dt)]
        if df_period.empty: 
            print(f"⚠️ {period_name} 时段 ({start_time}-{end_time}) 暂无数据！")
            continue
            
        inter_evals = []
        for inter, group_inter in df_period.groupby('路口名称'):
            flow_stats = group_inter.groupby(['进口道方向', '转向']).agg(
                延误_90分位=('延误指数', lambda x: x.quantile(0.90)),
                平均流量占比=('流量占比(%)', 'mean'),
                平均分均流量=('pass_flow', 'mean'),
                历史最大分均流量=('历史最大分均流量', 'first')
            ).reset_index()
            
            # 极简版：交叉口综合拥堵指数 = (0.4 × 流量占比) + (0.6 × 延误指数)
            alpha_weight = 0.4
            beta_weight = 0.6
            flow_stats['流向综合得分'] = (alpha_weight * flow_stats['平均流量占比']) + (beta_weight * flow_stats['延误_90分位'])
            stress_score = flow_stats['流向综合得分'].sum()
            
            inter_evals.append({
                '路口名称': inter,
                '综合拥堵指数': stress_score,
                '_flow_stats': flow_stats 
            })
            
        if inter_evals:
            df_eval = pd.DataFrame(inter_evals)
            # 仅仅通过综合拥堵指数进行排序
            df_eval = df_eval.sort_values(by=['综合拥堵指数'], ascending=[False])
            
            # ========== 选出关键瓶颈路口 ==========
            critical_inter = df_eval.iloc[0]['路口名称']
            cici_score = df_eval.iloc[0]['综合拥堵指数']
            critical_flow_stats = df_eval.iloc[0]['_flow_stats']
            
            print(f"【{period_name}】子区路口综合延误状况：")
            display_df = df_eval[['路口名称', '综合拥堵指数']].copy()
            display_df['综合拥堵指数'] = display_df['综合拥堵指数'].round(2)
            print(display_df.to_string(index=False))
            
            print(f"关键路口定位：")
            print(f"   -> 经过延误指数与流量状况加权评估，子区关键路口为：🎯 {critical_inter}")
            print(f"      (入选理由：综合延误状况最高达 {cici_score:.2f})")
            
            # ========== 为关键瓶颈路口推演公共周期 ==========
            phase_y_dict = {}
            phase_flow_details = {} 
            L_total = 16 
            
            for phase_name, directions in PHASE_MAPPING.items():
                max_y = 0.0
                best_detail = ""
                for d, t in directions:
                    mov = critical_flow_stats[(critical_flow_stats['进口道方向'] == d) & (critical_flow_stats['转向'] == t)]
                    if not mov.empty:
                        cur_flow = mov['平均分均流量'].iloc[0]
                        max_flow = mov['历史最大分均流量'].iloc[0]
                        
                        # ----- 核心兜底逻辑 -----
                        if max_flow > 0:
                            y_val = cur_flow / max_flow
                            detail_str = f"均量:{cur_flow:.1f}/最大:{max_flow:.1f} = y:{y_val:.3f}"
                        else:
                            ratio = mov['平均流量占比'].iloc[0]
                            y_val = (ratio / 100.0) * 1.5
                            detail_str = f"绝量缺失, 按占比{ratio:.1f}%折算 = y:{y_val:.3f}"
                            
                        if y_val >= max_y:
                            max_y = y_val
                            best_detail = f"{d}{t} ({detail_str})"
                
                phase_y_dict[phase_name] = max_y
                phase_flow_details[phase_name] = best_detail

            Y_sum = sum(phase_y_dict.values())
            subzone_ccl, alp, bet, the, Y_val = modified_webster_cycle(Y_sum, L_total)

            print(f"   -> 以该瓶颈路口为基准，建议光明大道子区公共周期：【{subzone_ccl} 秒】")
            
            print(f"【{critical_inter}】修正的 Webster 周期推导模型：")
            print(f"   计算公式：C_0 = (αL + β) / (1 - θY)")
            print(f"   [提取经验流量比 y]：")
            for phase, info in phase_flow_details.items():
                print(f"     - {phase}: 取关键流向 {info}")
            print(f"   [交叉口控制系数]：")
            print(f"     - 交叉口经验流率比总和 (Y) = {Y_val:.3f}")
            print(f"     - 交叉口总损失时间 (L) = {L_total}s")
            print(f"     - 稳定控制系数：α = {alp}, β = {bet}, θ = {the}")
            print(f"   [周期计算结果]：({alp} × {L_total} + {bet}) / (1 - {the} × {Y_val:.3f}) ≈ {subzone_ccl}s")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_dir, "干线全流向_延误与流量占比明细表.csv")
    try:
        df_details = load_and_preprocess_data(csv_file_path)
        estimate_subzone_ccl(df_details)
    except Exception as e:
        print(f"运行失败: {e}")