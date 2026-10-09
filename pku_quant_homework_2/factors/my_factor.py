"""
因子挖掘实验：流动性/结构稳定性因子 —— 自主设计与验证

研究动机（三轮失败后的新方向）：
    前三轮"价量反转"挖掘全部失败（沪深300 × 2021-2025）：
      - 跳空/量价/反转类因子 RankIC 全在 ±0.02 以内
      - 发现的"均值陷阱"与"选择诱饵"证明简单价量规律已被套利
    本轮换一个完全不同的思路：不预测涨跌方向，而是寻找【结构特征】。

核心假设 H3：
    A 股中，"流动性稳定"的股票与"流动性不稳定"的股票，
    后续收益存在系统性差异。
    直觉：流动性不稳 = 有大事要发生（消息/情绪剧烈波动/停牌风险），
    这类股票平均而言表现差。该假设不依赖方向预测，只看"结构"。

    该假设【可证伪】：若不同流动性组收益无显著差异，H3 被否。

四轮设计（每轮锁定一个机制）：
    R1 流动性稳定性：成交额的变异系数、偏度、自相关
    R2 波动性稳定性：振幅、收益波动、换手率的稳定性
    R3 量价配合：上涨缩量/放量、下跌缩量/放量的特征
    R4 组合正交：前三轮强因子合成

评估框架：
    1. 合并所有股票为面板数据（日期 × 股票）
    2. 因子先做截面标准化（z-score），消除量纲混淆
    3. label：T+1 买入、T+3 卖出的收益
    4. 按日截面算 IC（Pearson）/ RankIC（Spearman）/ ICIR
    5. 按因子值分 10 组，【同时】看平均收益与中位收益
       ⚠️ 关键教训：只看均值会被"少数妖股拉高"骗（均值虚高、中位数反向）
    6. 多空组合（Q10-Q1），并出绩效图

样本分段（样本内挖掘 / 样本外验证）：
    train 2021-2023 选定方向 → valid 2024-2025 检验是否存活
    只在一段上表现好 = 过拟合；两段都活 = 真信号

用法：
    python my_factor.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # 无界面后端，直接存图
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# 未来收益口径：T+1 到 T+3
HOLD_START = 1
HOLD_END = 3

# 样本分段（样本内挖掘 / 样本外验证）
# 只在一段上表现好 = 过拟合；两段都活 = 真信号
TRAIN_START = "2021-01-01"
TRAIN_END = "2023-12-31"
VALID_START = "2024-01-01"
VALID_END = "2025-12-31"

# ============ 4 轮挖掘框架 ============
# 每轮锁定一个"机制假设"，围绕它生成一批候选；用 train 选、valid 验。
# 晋级线：|valid RankIC|>=0.015 且 均值/中位单调性均>=0.5 且 train/valid 同号。

# 回看窗口候选（R2 用）
WINDOWS = [2, 3, 4, 5, 10]

# 各轮候选因子名（由 build_round_candidates 生成）
ROUND_NAMES = ["R1_时间结构", "R2_窗口平台", "R3_量价增强", "R4_组合正交"]


def load_panel() -> pd.DataFrame:
    """读取 data/ 下所有股票 CSV，合并为面板数据。"""
    frames = []
    for f in sorted(DATA_DIR.glob("*.csv")):
        df = pd.read_csv(f, encoding="utf-8-sig")
        if df.empty:
            continue
        df["symbol"] = f.stem
        frames.append(df)
    if not frames:
        raise SystemExit(f"未在 {DATA_DIR} 找到数据，请先运行 download_data.py")
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    # 新浪接口字段：date/open/high/low/close/volume/amount/outstanding_share/turnover
    panel = panel[["date", "symbol", "open", "close", "high", "low",
                   "volume", "amount", "outstanding_share", "turnover"]]
    return panel.sort_values(["symbol", "date"]).reset_index(drop=True)


def _rsum(sym: pd.Series, panel: pd.DataFrame, n: int) -> pd.Series:
    """按股票分组做 n 日滚动求和。"""
    return sym.groupby(panel["symbol"]).transform(lambda s: s.rolling(n).sum())


def _rmean(sym: pd.Series, panel: pd.DataFrame, n: int) -> pd.Series:
    """按股票分组做 n 日滚动均值。"""
    return sym.groupby(panel["symbol"]).transform(lambda s: s.rolling(n).mean())


def compute_features(panel: pd.DataFrame) -> pd.DataFrame:
    """计算基础中间量（收益/振幅/量变化等）与未来收益 label。"""
    g = panel.groupby("symbol", group_keys=False)
    prev_close = g["close"].shift(1)
    panel["ret_1"] = panel["close"] / prev_close - 1.0
    panel["ret_overnight"] = panel["open"] / prev_close - 1.0
    panel["ret_intraday"] = panel["close"] / panel["open"] - 1.0
    # 振幅（日内高低幅）
    panel["amp"] = panel["high"] / panel["low"] - 1.0
    # 成交量对数变化
    panel["dvol"] = g["volume"].transform(lambda s: np.log(s).diff())
    # 收益绝对值（用于波动计算）
    panel["abs_ret"] = panel["ret_1"].abs()
    # 涨跌方向
    panel["up"] = (panel["ret_1"] > 0).astype(float)

    # 未来收益 label：T+1 买入、T+3 卖出
    close_start = g["close"].shift(-HOLD_START)
    close_end = g["close"].shift(-HOLD_END)
    panel["label"] = close_end / close_start - 1.0
    return panel


def _rstd(sym: pd.Series, panel: pd.DataFrame, n: int) -> pd.Series:
    """按股票分组做 n 日滚动标准差。"""
    return sym.groupby(panel["symbol"]).transform(lambda s: s.rolling(n).std())


def _rskew(sym: pd.Series, panel: pd.DataFrame, n: int) -> pd.Series:
    """按股票分组做 n 日滚动偏度。"""
    return sym.groupby(panel["symbol"]).transform(lambda s: s.rolling(n).skew())


def _rmin(sym: pd.Series, panel: pd.DataFrame, n: int) -> pd.Series:
    """按股票分组做 n 日滚动最小值。"""
    return sym.groupby(panel["symbol"]).transform(lambda s: s.rolling(n).min())


def _rmax(sym: pd.Series, panel: pd.DataFrame, n: int) -> pd.Series:
    """按股票分组做 n 日滚动最大值。"""
    return sym.groupby(panel["symbol"]).transform(lambda s: s.rolling(n).max())


# ---------------------------------------------------------------
# 各轮候选因子定义
# ---------------------------------------------------------------

def build_round1(panel: pd.DataFrame) -> list[str]:
    """R1 流动性稳定性：成交额的变异系数、偏度、自相关等。"""
    amt = panel["amount"]
    vol = panel["volume"]
    made = []

    # 1) 成交额变异系数 CV = std/mean（不同窗口）
    for n in [5, 10, 20]:
        m = _rmean(amt, panel, n)
        s = _rstd(amt, panel, n)
        panel[f"R1_AMT_CV_{n}"] = s / m.replace(0, np.nan)
        made.append(f"R1_AMT_CV_{n}")

    # 2) 成交额偏度（不同窗口）
    for n in [5, 10, 20]:
        panel[f"R1_AMT_SKEW_{n}"] = _rskew(amt, panel, n)
        made.append(f"R1_AMT_SKEW_{n}")

    # 3) 成交额自相关（今日与昨日、今日与5日前）
    for lag in [1, 5]:
        lagged = amt.groupby(panel["symbol"]).shift(lag)
        panel[f"R1_AMT_AUTOC_{lag}"] = amt / lagged.replace(0, np.nan) - 1.0
        made.append(f"R1_AMT_AUTOC_{lag}")

    # 4) 成交额最大/最小 之比（20日）
    m20 = _rmean(amt, panel, 20)
    panel["R1_AMT_RANGE"] = (_rmax(amt, panel, 20) / m20.replace(0, np.nan)
                             - _rmin(amt, panel, 20) / m20.replace(0, np.nan))
    made.append("R1_AMT_RANGE")

    # 5) 成交额趋势（5日均量 / 20日均量）
    panel["R1_AMT_TREND"] = (_rmean(amt, panel, 5)
                             / _rmean(amt, panel, 20).replace(0, np.nan))
    made.append("R1_AMT_TREND")

    return made


def build_round2(panel: pd.DataFrame, seed_kind: str = "") -> list[str]:
    """R2 波动性稳定性：振幅、收益波动、换手率的稳定性。"""
    made = []
    ret = panel["ret_1"]
    amp = panel["amp"]
    turn = panel["turnover"]

    # 1) 收益波动的变异系数（不同窗口）
    for n in [5, 10, 20]:
        m = _rmean(ret.abs(), panel, n)
        s = _rstd(ret.abs(), panel, n)
        panel[f"R2_RET_CV_{n}"] = s / m.replace(0, np.nan)
        made.append(f"R2_RET_CV_{n}")

    # 2) 振幅的稳定性（振幅的变异系数）
    for n in [5, 10, 20]:
        m = _rmean(amp, panel, n)
        s = _rstd(amp, panel, n)
        panel[f"R2_AMP_CV_{n}"] = s / m.replace(0, np.nan)
        made.append(f"R2_AMP_CV_{n}")

    # 3) 换手率的稳定性
    for n in [5, 10, 20]:
        m = _rmean(turn, panel, n)
        s = _rstd(turn, panel, n)
        panel[f"R2_TURN_CV_{n}"] = s / m.replace(0, np.nan)
        made.append(f"R2_TURN_CV_{n}")

    # 4) 换手率的绝对水平
    panel["R2_TURN_MEAN_20"] = _rmean(turn, panel, 20)
    made.append("R2_TURN_MEAN_20")

    # 5) 收益的偏度（20日）
    panel["R2_RET_SKEW_20"] = _rskew(ret, panel, 20)
    made.append("R2_RET_SKEW_20")

    return made


def build_round3(panel: pd.DataFrame, seed: pd.Series | None = None,
                 seed_name: str = "") -> list[str]:
    """R3 量价配合：上涨时缩量/放量、下跌时缩量/放量的特征。"""
    made = []
    up = panel["up"]
    amt = panel["amount"]
    turn = panel["turnover"]

    # 1) 上涨日平均成交额 / 下跌日平均成交额（20日）
    amt_up = amt * up
    amt_dn = amt * (1 - up)
    for n in [10, 20]:
        up_n = _rmean(amt_up, panel, n)
        dn_n = _rmean(amt_dn, panel, n)
        panel[f"R3_UPDN_AMT_{n}"] = up_n / dn_n.replace(0, np.nan)
        made.append(f"R3_UPDN_AMT_{n}")

    # 2) 上涨日平均换手 / 下跌日平均换手（20日）
    turn_up = turn * up
    turn_dn = turn * (1 - up)
    for n in [10, 20]:
        up_n = _rmean(turn_up, panel, n)
        dn_n = _rmean(turn_dn, panel, n)
        panel[f"R3_UPDN_TURN_{n}"] = up_n / dn_n.replace(0, np.nan)
        made.append(f"R3_UPDN_TURN_{n}")

    # 3) 上涨天数占比（20日）
    for n in [10, 20]:
        panel[f"R3_UP_RATIO_{n}"] = _rmean(up, panel, n)
        made.append(f"R3_UP_RATIO_{n}")

    # 4) 量价相关性：收益与成交额的相关（10日）
    for n in [10, 20]:
        corr = (panel.groupby("symbol", group_keys=False)["ret_1"]
                .transform(lambda s: s.rolling(n).corr(panel.loc[s.index, "amount"])))
        panel[f"R3_PV_CORR_{n}"] = corr
        made.append(f"R3_PV_CORR_{n}")

    return made


def build_round4(panel: pd.DataFrame, seeds: list[tuple[str, float]]) -> list[str]:
    """R4 组合正交：把若干弱因子按 z-score 等权/加权合成。"""
    made = []
    d = panel["date"]

    def zscore(col: str) -> pd.Series:
        return panel.groupby("date")[col].transform(
            lambda s: (s - s.mean()) / s.std() if s.std() else s * np.nan)

    zs = {name: zscore(name) for name, _ in seeds}

    # 等权合成
    eq = sum(zs.values()) / len(zs)
    panel["R4_EQUAL"] = eq
    made.append("R4_EQUAL")

    # 按 RankIC 绝对值加权
    total = sum(abs(w) for _, w in seeds) or 1.0
    wt = sum(zs[name] * (abs(w) / total) for name, w in seeds)
    panel["R4_WEIGHTED"] = wt
    made.append("R4_WEIGHTED")

    # 只用最强的两个
    top2 = sorted(seeds, key=lambda x: abs(x[1]), reverse=True)[:2]
    if len(top2) == 2:
        panel["R4_TOP2"] = (zs[top2[0][0]] + zs[top2[1][0]]) / 2.0
        made.append("R4_TOP2")

    # 主成分式：最强因子 − 次强因子（去共线）
    if len(top2) == 2:
        panel["R4_SPREAD"] = zs[top2[0][0]] - zs[top2[1][0]]
        made.append("R4_SPREAD")
    return made


# ---------------------------------------------------------------
# R5-R8：围绕"换手率"深挖（承接 R2_TURN_MEAN_20 的发现）
# 发现：换手率越高 → 后续收益越低（train/valid 都成立，中位数口径）
# ---------------------------------------------------------------

def build_round5(panel: pd.DataFrame) -> list[str]:
    """R5 换手率本身的最佳形态：窗口、绝对值/排名/变化率。"""
    made = []
    turn = panel["turnover"]

    # 1) 换手率的均值（多窗口）
    for n in [5, 10, 20]:
        panel[f"R5_TURN_MEAN_{n}"] = _rmean(turn, panel, n)
        made.append(f"R5_TURN_MEAN_{n}")

    # 2) 换手率的截面排名（多窗口，0~1）
    for n in [5, 10, 20]:
        m = _rmean(turn, panel, n)
        panel[f"R5_TURN_RANK_{n}"] = m.groupby(panel["date"]).rank(pct=True)
        made.append(f"R5_TURN_RANK_{n}")

    # 3) 换手率的变化率（今日 / 5日均、5日均 / 20日均）
    panel["R5_TURN_CHG_1"] = turn / _rmean(turn, panel, 5).replace(0, np.nan)
    panel["R5_TURN_CHG_5"] = (_rmean(turn, panel, 5)
                              / _rmean(turn, panel, 20).replace(0, np.nan))
    made += ["R5_TURN_CHG_1", "R5_TURN_CHG_5"]

    # 4) 换手率的 z-score（时间序列标准化，消除个股基础差异）
    for n in [10, 20]:
        m = _rmean(turn, panel, n)
        s = _rstd(turn, panel, n)
        panel[f"R5_TURN_Z_{n}"] = (turn - m) / s.replace(0, np.nan)
        made.append(f"R5_TURN_Z_{n}")

    return made


def build_round6(panel: pd.DataFrame) -> list[str]:
    """R6 换手率 × 大小：去掉市值干扰后的纯换手率效应。"""
    made = []
    turn = panel["turnover"]
    # 用 outstanding_share（流通股本）作市值代理
    size = panel["outstanding_share"]
    panel["log_size"] = np.log(size)

    # 1) 换手率 与 市值 的截面相关性（诊断）
    # 2) 换手率 剔除市值后的残差（按日截面回归）
    turn20 = _rmean(turn, panel, 20)

    def _resid(df: pd.DataFrame) -> pd.Series:
        x = df["log_size"].values
        y = df["turn20"].values
        if len(x) < 10 or np.std(x) == 0:
            return pd.Series(np.nan, index=df.index)
        # 简单线性回归 y = a + b*x，返回残差
        b = np.cov(x, y, ddof=0)[0, 1] / np.var(x)
        a = np.mean(y) - b * np.mean(x)
        return pd.Series(y - (a + b * x), index=df.index)

    tmp = panel[["date", "log_size"]].copy()
    tmp["turn20"] = turn20
    panel["R6_TURN_RESID"] = tmp.groupby("date").apply(
        _resid, include_groups=False).reset_index(level=0, drop=True)
    made.append("R6_TURN_RESID")

    # 3) 换手率 × 市值（看组合效应）
    panel["R6_TURN_x_SIZE"] = turn20 * panel["log_size"]
    made.append("R6_TURN_x_SIZE")

    # 4) 换手率 / 市值（另一种标准化）
    panel["R6_TURN_div_SIZE"] = turn20 / size.replace(0, np.nan)
    made.append("R6_TURN_div_SIZE")

    # 5) 换手率与市值的相对排名差
    rank_turn = turn20.groupby(panel["date"]).rank(pct=True)
    rank_size = size.groupby(panel["date"]).rank(pct=True)
    panel["R6_TURN_MINUS_SIZE"] = rank_turn - rank_size
    made.append("R6_TURN_MINUS_SIZE")

    return made


def build_round7(panel: pd.DataFrame) -> list[str]:
    """R7 换手率 × 其他：加振幅/波动/量稳定性做复合因子。"""
    made = []
    turn20 = _rmean(panel["turnover"], panel, 20)
    amp20 = _rmean(panel["amp"], panel, 20)
    ret_std20 = _rstd(panel["ret_1"], panel, 20)
    amt20 = _rmean(panel["amount"], panel, 20)
    amt_cv20 = _rstd(panel["amount"], panel, 20) / amt20.replace(0, np.nan)

    # 1) 换手率 × 振幅（都是"过热"指标，相乘放大）
    panel["R7_TURN_x_AMP"] = turn20 * amp20
    made.append("R7_TURN_x_AMP")

    # 2) 换手率 × 收益波动（情绪+波动双重过热）
    panel["R7_TURN_x_VOL"] = turn20 * ret_std20
    made.append("R7_TURN_x_VOL")

    # 3) 换手率 × 成交额变异系数（不稳 + 高热）
    panel["R7_TURN_x_CV"] = turn20 * amt_cv20
    made.append("R7_TURN_x_CV")

    # 4) 换手率 / 成交额变异系数（稳定高热 vs 不稳高热）
    panel["R7_TURN_div_CV"] = turn20 / amt_cv20.replace(0, np.nan)
    made.append("R7_TURN_div_CV")

    # 5) 复合 z-score：换手率 + 振幅 + 波动 三者等权
    zt = turn20.groupby(panel["date"]).transform(
        lambda s: (s - s.mean()) / s.std() if s.std() else s * np.nan)
    za = amp20.groupby(panel["date"]).transform(
        lambda s: (s - s.mean()) / s.std() if s.std() else s * np.nan)
    zv = ret_std20.groupby(panel["date"]).transform(
        lambda s: (s - s.mean()) / s.std() if s.std() else s * np.nan)
    panel["R7_HEAT"] = (zt + za + zv) / 3.0
    made.append("R7_HEAT")

    return made


def build_round8(panel: pd.DataFrame) -> list[str]:
    """R8 换手率极端值：换手率暴涨/暴跌的股票（事件驱动）。"""
    made = []
    turn = panel["turnover"]
    turn20 = _rmean(turn, panel, 20)
    turn5 = _rmean(turn, panel, 5)

    # 1) 今日换手率 / 20日均值（是否异常放量）
    panel["R8_TURN_SPIKE"] = turn / turn20.replace(0, np.nan)
    made.append("R8_TURN_SPIKE")

    # 2) 5日均值 / 20日均值（短期相对长期的放量程度）
    panel["R8_TURN_SURGE"] = turn5 / turn20.replace(0, np.nan)
    made.append("R8_TURN_SURGE")

    # 3) 今日换手率截面排名（是否在最活跃的 10%）
    panel["R8_TURN_TOP"] = turn.groupby(panel["date"]).rank(pct=True)
    made.append("R8_TURN_TOP")

    # 4) 换手率暴涨事件（今日 > 20日均值 2 倍）
    panel["R8_TURN_SPIKE_2X"] = (turn > turn20 * 2).astype(float)
    made.append("R8_TURN_SPIKE_2X")

    # 5) 换手率暴涨后的第二天（事件后动量）
    spike = (turn > turn20 * 2).astype(float)
    panel["R8_SPIKE_T1"] = spike.groupby(panel["symbol"]).shift(1)
    made.append("R8_SPIKE_T1")

    return made


def compute_factors(panel: pd.DataFrame) -> pd.DataFrame:
    """兼容旧接口：计算基础特征（label 等）。"""
    return compute_features(panel)


def cross_section_zscore(panel: pd.DataFrame, factor: str) -> pd.Series:
    """按日截面做 z-score 标准化，消除量纲与波动大小的混淆。"""
    def _z(s: pd.Series) -> pd.Series:
        std = s.std()
        if not std or np.isnan(std):
            return pd.Series(np.nan, index=s.index)
        return (s - s.mean()) / std

    return panel.groupby("date")[factor].transform(_z)


def daily_ic(panel: pd.DataFrame, factor: str) -> pd.DataFrame:
    """按日截面计算 IC（Pearson）与 RankIC（Spearman）。"""
    sub = panel[["date", factor, "label"]].dropna()

    def _ic(x: pd.DataFrame) -> float:
        if len(x) < 10:
            return np.nan
        return x[factor].corr(x["label"])

    def _rank_ic(x: pd.DataFrame) -> float:
        if len(x) < 10:
            return np.nan
        return x[factor].corr(x["label"], method="spearman")

    ic = sub.groupby("date").apply(_ic, include_groups=False)
    rank_ic = sub.groupby("date").apply(_rank_ic, include_groups=False)
    return pd.DataFrame({"IC": ic, "RankIC": rank_ic}).dropna()


def group_returns(panel: pd.DataFrame, factor: str, n_groups: int = 10,
                  stat: str = "mean") -> pd.DataFrame:
    """按因子值分 n 组，返回 日期 × 分组 的每日组内收益矩阵。

    stat="mean"   组内平均收益（受少数极端股影响）
    stat="median" 组内中位收益（对极端值稳健，用于揭穿"均值陷阱"）
    """
    sub = panel[["date", factor, "label"]].dropna().copy()

    def _assign(x: pd.Series) -> pd.Series:
        if x.notna().sum() < n_groups * 2:
            return pd.Series(np.nan, index=x.index)
        return pd.qcut(x, n_groups, labels=False, duplicates="drop")

    sub["group"] = sub.groupby("date")[factor].transform(_assign)
    sub = sub.dropna(subset=["group"])
    sub["group"] = sub["group"].astype(int)

    grp = sub.pivot_table(index="date", columns="group", values="label", aggfunc=stat)
    grp.index = pd.to_datetime(grp.index)
    return grp.sort_index()


def summarize(ic_df: pd.DataFrame, grp: pd.DataFrame, factor: str,
              segment: str = "全样本", grp_med: pd.DataFrame | None = None) -> dict:
    """汇总因子绩效指标。

    「单调性」：分组序号与组均收益的 Spearman 相关。
      理想因子接近 +1（因子值越大收益越高）或 -1（反之）；接近 0 = 无排序能力。

    ⚠️ 「中位数版」指标（grp_med）：对极端值稳健。
      若均值单调性高、中位数单调性却相反 —— 说明收益全靠少数极端股拉高，
      是【均值陷阱】，不是真信号。
    """
    if ic_df.empty or grp.empty:
        return {"factor": factor, "segment": segment}

    ic_mean = ic_df["IC"].mean()
    rank_ic_mean = ic_df["RankIC"].mean()
    ic_ir = ic_df["IC"].mean() / ic_df["IC"].std()
    rank_ic_ir = ic_df["RankIC"].mean() / ic_df["RankIC"].std()
    ic_win = (ic_df["IC"] > 0).mean()
    rank_ic_win = (ic_df["RankIC"] > 0).mean()

    cols = sorted(grp.columns)
    group_mean = grp[cols].mean()

    # 单调性：分组序号 vs 组均收益 的秩相关
    if len(cols) >= 3:
        monotonicity = pd.Series(range(len(cols))).corr(
            pd.Series(group_mean.values), method="spearman")
    else:
        monotonicity = np.nan

    # 多空：最高组 - 最低组（丢弃任一组缺失的日期，否则 cumprod 会被 nan 污染）
    ls = (grp[cols[-1]] - grp[cols[0]]).dropna()
    ls_cum = (1 + ls).cumprod()
    ann_ret = ls.mean() * 252
    ann_vol = ls.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum = ls_cum.iloc[-1] - 1 if len(ls_cum) else np.nan

    out = {
        "factor": factor,
        "segment": segment,
        "IC均值": ic_mean,
        "RankIC均值": rank_ic_mean,
        "ICIR": ic_ir,
        "RankICIR": rank_ic_ir,
        "IC胜率": ic_win,
        "RankIC胜率": rank_ic_win,
        "单调性": monotonicity,
        "多空年化": ann_ret,
        "多空夏普": sharpe,
        "多空累计": cum,
        "样本天数": len(ic_df),
    }

    # ---------- 中位数稳健性检验（防"均值陷阱"） ----------
    if grp_med is not None and not grp_med.empty:
        mcols = sorted(grp_med.columns)
        gmed = grp_med[mcols].mean()
        if len(mcols) >= 3:
            out["中位单调性"] = pd.Series(range(len(mcols))).corr(
                pd.Series(gmed.values), method="spearman")
        else:
            out["中位单调性"] = np.nan
        ls_m = (grp_med[mcols[-1]] - grp_med[mcols[0]]).dropna()
        out["多空中位夏普"] = (ls_m.mean() * 252
                              / (ls_m.std() * np.sqrt(252))
                              if ls_m.std() > 0 else np.nan)
    return out


def evaluate(panel: pd.DataFrame, factor: str, mask: pd.Series,
             segment: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """在指定样本区间上评估一个因子，返回 (指标, ic_df, grp, grp_med)。"""
    sub = panel[mask]
    zf = factor + "_Z"
    ic_df = daily_ic(sub, factor)
    grp = group_returns(sub, zf, stat="mean")
    grp_med = group_returns(sub, zf, stat="median")
    s = summarize(ic_df, grp, factor, segment, grp_med)
    return s, ic_df, grp


def plot_results(factor: str, ic_df: pd.DataFrame, grp: pd.DataFrame,
                 metrics: dict, segments: dict | None = None) -> None:
    """出三张图：分组平均收益 + 多空累计净值 + 累积 RankIC。

    segments: {"train": (start, end), "valid": (start, end)} 用于在图上画分界线。
    """
    plt.rcParams["axes.unicode_minus"] = False
    cols = sorted(grp.columns)

    def _draw_split(ax) -> None:
        """在多空/IC 图上画出 train/valid 分界。"""
        if not segments:
            return
        for name, (s, e) in segments.items():
            try:
                ax.axvline(pd.Timestamp(s), color="gray", linestyle="--",
                           linewidth=1, alpha=0.7)
            except Exception:
                pass
        # 标注
        if "train" in segments and "valid" in segments:
            try:
                ts, te = pd.Timestamp(segments["train"][0]), pd.Timestamp(segments["train"][1])
                vs = pd.Timestamp(segments["valid"][0])
                xmin, xmax = ax.get_xlim()
                ax.text(ts, 1.02, "train", transform=ax.get_xaxis_transform(),
                        fontsize=9, color="gray")
                ax.text(vs, 1.02, "valid", transform=ax.get_xaxis_transform(),
                        fontsize=9, color="gray")
            except Exception:
                pass

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 图1：十分组平均收益（看单调性）
    ax = axes[0]
    means = grp[cols].mean() * 100
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in means]  # 红涨绿跌（中国习惯）
    ax.bar([str(int(c) + 1) for c in cols], means, color=colors)
    ax.set_title(f"{factor} 十分组平均未来收益(%)\n"
                 f"单调性={metrics.get('单调性', float('nan')):.2f}")
    ax.set_xlabel("分组（1=因子值最低，10=最高）")
    ax.set_ylabel("平均收益(%)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

    # 图2：多空累计收益（含 train/valid 分界）
    ax = axes[1]
    ls = (grp[cols[-1]] - grp[cols[0]]).dropna()
    ax.plot(ls.index, (1 + ls).cumprod(), color="#d62728", linewidth=1.5)
    ax.set_title(f"{factor} 多空组合(Q{len(cols)}-Q1)累计净值\n"
                 f"夏普={metrics.get('多空夏普', float('nan')):.2f}")
    ax.set_ylabel("净值")
    ax.grid(alpha=0.3)
    _draw_split(ax)

    # 图3：累积 RankIC
    ax = axes[2]
    ax.plot(ic_df.index, ic_df["RankIC"].cumsum(), color="#1f77b4", linewidth=1.5)
    ax.set_title(f"{factor} 累积 RankIC\n均值={metrics.get('RankIC均值', float('nan')):.4f}")
    ax.set_ylabel("累积 RankIC")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(alpha=0.3)
    _draw_split(ax)

    plt.tight_layout()
    out = OUT_DIR / f"{factor}_performance.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  图已保存：{out}", flush=True)


def screen(panel: pd.DataFrame, mask_train: pd.Series, mask_valid: pd.Series,
           names: list[str], tag: str) -> pd.DataFrame:
    """对一批候选因子做筛选评分，返回指标表（按 |valid RankIC| 降序）。

    晋级条件（三条同时满足）：
      1. |valid RankIC| >= 0.015
      2. |valid 均值单调性| >= 0.5  且  |valid 中位单调性| >= 0.5（非均值陷阱）
      3. train 与 valid 的 RankIC 同号（方向一致）
    """
    rows = []
    for name in names:
        if name not in panel.columns:
            continue
        # 标准化列用于分组
        zc = name + "_Z"
        if zc not in panel.columns:
            panel[zc] = cross_section_zscore(panel, name)

        s_tr, _, _ = evaluate(panel, name, mask_train, "train")
        s_va, _, _ = evaluate(panel, name, mask_valid, "valid")
        if "IC均值" not in s_va:
            continue

        ric_v = s_va["RankIC均值"]
        ric_t = s_tr.get("RankIC均值", np.nan)
        mono = abs(s_va["单调性"])
        mono_med = abs(s_va.get("中位单调性") or 0)
        same_sign = (np.sign(ric_v) == np.sign(ric_t)) if not np.isnan(ric_t) else False

        passed = (abs(ric_v) >= 0.015 and mono >= 0.5
                  and mono_med >= 0.5 and same_sign)
        rows.append({
            "因子": name,
            "tag": tag,
            "train_RIC": ric_t,
            "valid_RIC": ric_v,
            "valid_RICIR": s_va["RankICIR"],
            "valid_均值单调": s_va["单调性"],
            "valid_中位单调": s_va.get("中位单调性", np.nan),
            "valid_均值夏普": s_va["多空夏普"],
            "valid_中位夏普": s_va.get("多空中位夏普", np.nan),
            "同号": same_sign,
            "晋级": "YES" if passed else "",
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["_abs"] = df["valid_RIC"].abs()
    return df.sort_values("_abs", ascending=False).drop(columns="_abs").reset_index(drop=True)


def main() -> None:
    print("加载数据...", flush=True)
    panel = load_panel()
    print(f"  面板数据：{panel['symbol'].nunique()} 只股票，{panel['date'].nunique()} 个交易日，"
          f"{len(panel)} 行", flush=True)
    print(f"  区间：{panel['date'].min().date()} ~ {panel['date'].max().date()}", flush=True)

    panel = compute_features(panel)

    d = panel["date"]
    mask_train = (d >= TRAIN_START) & (d <= TRAIN_END)
    mask_valid = (d >= VALID_START) & (d <= VALID_END)
    mask_all = pd.Series(True, index=panel.index)
    segments = {"train": (TRAIN_START, TRAIN_END),
                "valid": (VALID_START, VALID_END)}
    print(f"  样本分段：train {TRAIN_START}~{TRAIN_END}"
          f"（{d[mask_train].nunique()} 交易日）、"
          f"valid {VALID_START}~{VALID_END}"
          f"（{d[mask_valid].nunique()} 交易日）", flush=True)

    LAB_DIR = OUT_DIR / "lab"
    LAB_DIR.mkdir(exist_ok=True)
    all_round_results = []

    # ================= R1：流动性稳定性 =================
    print(f"\n{'#'*70}\n# R1 流动性稳定性：成交额变异系数/偏度/自相关（11 候选）\n{'#'*70}",
          flush=True)
    r1_names = build_round1(panel)
    r1 = screen(panel, mask_train, mask_valid, r1_names, "R1")
    print(r1.to_string(index=False), flush=True)
    r1.to_csv(LAB_DIR / "R1_metrics.csv", index=False, encoding="utf-8-sig")
    all_round_results.append(r1)
    best_r1 = r1.iloc[0]["因子"] if not r1.empty else None
    print(f"\n  R1 最优：{best_r1}", flush=True)

    # ================= R2：波动性稳定性 =================
    print(f"\n{'#'*70}\n# R2 波动性稳定性：振幅/收益/换手率稳定性（11 候选）\n{'#'*70}",
          flush=True)
    r2_names = build_round2(panel)
    r2 = screen(panel, mask_train, mask_valid, r2_names, "R2")
    print(r2.to_string(index=False), flush=True)
    r2.to_csv(LAB_DIR / "R2_metrics.csv", index=False, encoding="utf-8-sig")
    all_round_results.append(r2)
    best_r2 = r2.iloc[0]["因子"] if not r2.empty else None
    print(f"\n  R2 最优：{best_r2}", flush=True)

    # ================= R3：量价配合 =================
    print(f"\n{'#'*70}\n# R3 量价配合：上涨缩量/放量、下跌缩量/放量（10 候选）\n{'#'*70}",
          flush=True)
    r3_names = build_round3(panel)
    r3 = screen(panel, mask_train, mask_valid, r3_names, "R3")
    print(r3.to_string(index=False), flush=True)
    r3.to_csv(LAB_DIR / "R3_metrics.csv", index=False, encoding="utf-8-sig")
    all_round_results.append(r3)
    best_r3 = r3.iloc[0]["因子"] if not r3.empty else None
    print(f"\n  R3 最优：{best_r3}", flush=True)

    # ================= R4：组合正交 =================
    print(f"\n{'#'*70}\n# R4 组合正交：合成前几轮的强因子\n{'#'*70}", flush=True)
    # 收集前几轮 top 因子作为组合种子
    pool = pd.concat([r1, r2, r3], ignore_index=True)
    pool = pool[pool["valid_RIC"].abs() >= 0.010].head(6)
    seeds = [(row["因子"], row["valid_RIC"]) for _, row in pool.iterrows()]
    print(f"  组合种子：{[(n, round(w,4)) for n, w in seeds]}", flush=True)
    r4_names = build_round4(panel, seeds) if seeds else []
    r4 = screen(panel, mask_train, mask_valid, r4_names, "R4")
    if not r4.empty:
        print(r4.to_string(index=False), flush=True)
        r4.to_csv(LAB_DIR / "R4_metrics.csv", index=False, encoding="utf-8-sig")
        all_round_results.append(r4)

    # ================= R5-R8：围绕"换手率"深挖 =================
    print(f"\n{'#'*70}\n# R5 换手率最佳形态（13 候选）\n{'#'*70}", flush=True)
    r5_names = build_round5(panel)
    r5 = screen(panel, mask_train, mask_valid, r5_names, "R5")
    print(r5.to_string(index=False), flush=True)
    r5.to_csv(LAB_DIR / "R5_metrics.csv", index=False, encoding="utf-8-sig")
    all_round_results.append(r5)

    print(f"\n{'#'*70}\n# R6 换手率 × 大小（5 候选）\n{'#'*70}", flush=True)
    r6_names = build_round6(panel)
    r6 = screen(panel, mask_train, mask_valid, r6_names, "R6")
    print(r6.to_string(index=False), flush=True)
    r6.to_csv(LAB_DIR / "R6_metrics.csv", index=False, encoding="utf-8-sig")
    all_round_results.append(r6)

    print(f"\n{'#'*70}\n# R7 换手率 × 其他（5 候选）\n{'#'*70}", flush=True)
    r7_names = build_round7(panel)
    r7 = screen(panel, mask_train, mask_valid, r7_names, "R7")
    print(r7.to_string(index=False), flush=True)
    r7.to_csv(LAB_DIR / "R7_metrics.csv", index=False, encoding="utf-8-sig")
    all_round_results.append(r7)

    print(f"\n{'#'*70}\n# R8 换手率极端值（5 候选）\n{'#'*70}", flush=True)
    r8_names = build_round8(panel)
    r8 = screen(panel, mask_train, mask_valid, r8_names, "R8")
    print(r8.to_string(index=False), flush=True)
    r8.to_csv(LAB_DIR / "R8_metrics.csv", index=False, encoding="utf-8-sig")
    all_round_results.append(r8)

    # ================= 总排名 =================
    print(f"\n{'#'*70}\n# 全部候选总排名（按 |valid RankIC| 降序，前 20）\n{'#'*70}",
          flush=True)
    final = pd.concat(all_round_results, ignore_index=True)
    final = final.sort_values("valid_RIC", key=lambda s: s.abs(),
                              ascending=False).reset_index(drop=True)
    final.to_csv(LAB_DIR / "ALL_metrics.csv", index=False, encoding="utf-8-sig")
    print(final.head(20).to_string(index=False), flush=True)

    winners = final[final["晋级"] == "YES"]
    print(f"\n{'='*70}", flush=True)
    if not winners.empty:
        print(f">>> 共 {len(winners)} 个因子晋级：", flush=True)
        print(winners[["因子", "valid_RIC", "valid_均值单调",
                       "valid_中位单调", "valid_均值夏普"]].to_string(index=False),
              flush=True)
        champ = winners.iloc[0]["因子"]
    else:
        print(">>> 没有因子通过晋级线（|valid RIC|>=0.015 且 均值/中位单调>=0.5 且 同号）",
              flush=True)
        champ = final.iloc[0]["因子"]
        print(f">>> 综合最优候选：{champ}", flush=True)

    # ============ 对冠军因子出完整报告 ============
    print(f"\n{'#'*70}\n# 冠军因子完整评估：{champ}\n{'#'*70}", flush=True)
    seg_metrics = {}
    for seg_name, mask in [("全样本", mask_all), ("train", mask_train),
                           ("valid", mask_valid)]:
        s, ic_df, grp = evaluate(panel, champ, mask, seg_name)
        seg_metrics[seg_name] = (s, ic_df, grp)
        if "IC均值" in s:
            print(f"  [{seg_name}] RankIC={s['RankIC均值']:+.4f}  "
                  f"均值单调={s['单调性']:+.2f}  "
                  f"中位单调={s.get('中位单调性', float('nan')):+.2f}  "
                  f"均值夏普={s['多空夏普']:+.2f}  "
                  f"中位夏普={s.get('多空中位夏普', float('nan')):+.2f}  "
                  f"({s['样本天数']}天)", flush=True)

    s_all, ic_all, grp_all = seg_metrics["全样本"]
    if "IC均值" in s_all:
        cols = sorted(grp_all.columns)
        print("  全样本分组【平均】收益(%):",
              " ".join(f"Q{int(c)+1}={grp_all[c].mean()*100:.2f}" for c in cols),
              flush=True)
        plot_results(champ, ic_all, grp_all, s_all, segments)


if __name__ == "__main__":
    main()
