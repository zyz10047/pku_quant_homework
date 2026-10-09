"""
下载沪深300成分股日线数据（akshare 新浪接口）。

策略：
    - 取沪深300成分股列表，取前 N 只（默认 100 只）
    - 逐只下载近若干年日线（前复权），用新浪接口 stock_zh_a_daily
    - 每只存为一个 CSV 到 data/ 目录
    - 内置重试与容错

⚠️ 关键：本环境有会话级代理（访问国内数据源会失败），故在请求层禁用代理。

用法：
    python download_data.py
"""

from pathlib import Path
from time import sleep

import requests

# ===== 禁用代理（本机会话代理会导致访问新浪/东财失败）=====
_orig_request = requests.Session.request


def _no_proxy_request(self, *args, **kwargs):
    kwargs.setdefault("proxies", {"http": None, "https": None})
    return _orig_request(self, *args, **kwargs)


requests.Session.request = _no_proxy_request

import akshare as ak  # noqa: E402  （必须在 patch 之后导入）

# ===== 配置 =====
N_STOCKS = 300              # 下载前 N 只成分股
START_DATE = "20210101"     # 起始日期（约 5 年，覆盖多种市场状态）
END_DATE = "20251231"       # 结束日期
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def to_sina_symbol(code: str) -> str:
    """把 6 位代码转为新浪格式：6/9 开头→sh，其余→sz。"""
    if code.startswith(("6", "9")):
        return f"sh{code}"
    return f"sz{code}"


def get_hs300_symbols(n: int) -> list[str]:
    """获取沪深300成分股代码列表（取前 n 只）。"""
    df = ak.index_stock_cons_csindex(symbol="000300")
    symbols = df["成分券代码"].astype(str).str.zfill(6).tolist()
    return symbols[:n]


def download_one(code: str, retries: int = 3) -> bool:
    """下载单只股票日线，存为 CSV。返回是否成功。"""
    out = DATA_DIR / f"{code}.csv"
    if out.exists() and out.stat().st_size > 0:
        return True

    sina = to_sina_symbol(code)
    for i in range(retries):
        try:
            df = ak.stock_zh_a_daily(
                symbol=sina,
                start_date=START_DATE,
                end_date=END_DATE,
                adjust="qfq",
            )
            if df is None or df.empty:
                return False
            df.to_csv(out, index=False, encoding="utf-8-sig")
            return True
        except Exception:
            sleep(1 + i)
    return False


def main() -> None:
    print(f"[1/2] 获取沪深300成分股列表（取前 {N_STOCKS} 只）...", flush=True)
    symbols = get_hs300_symbols(N_STOCKS)
    print(f"      共 {len(symbols)} 只：{symbols[:5]} ...", flush=True)

    print(f"[2/2] 开始下载日线（{START_DATE} ~ {END_DATE}）...", flush=True)
    ok, fail = 0, 0
    for i, sym in enumerate(symbols, 1):
        if download_one(sym):
            ok += 1
        else:
            fail += 1
        if i % 10 == 0 or i == len(symbols):
            print(f"      [{i}/{len(symbols)}] 成功 {ok}，失败 {fail}", flush=True)
        sleep(0.2)

    print(f"\n完成：成功 {ok}，失败 {fail}，数据目录 {DATA_DIR}", flush=True)


if __name__ == "__main__":
    main()
