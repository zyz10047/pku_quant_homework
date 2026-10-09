"""
最小化 CTP 交易程序：连接 -> 订阅行情 -> 打印推送 -> 下单闭环。

对应作业要求（老师课堂口头布置）：
    "自己跑一个 CTP 小交易程序"

本程序的闭环设计（极简版）：
    1. 连接交易/行情柜台并登录
    2. 订阅合约行情，持续打印
    3. 收到第一个目标合约的行情后，以对手价买入开仓 1 手
    4. 成交回报到达后，以对手价卖出平仓 1 手
    5. 打印"闭环成功"（持仓归零、无活动委托）

设计说明：
    - 下单用"对手价"（买用卖一价、卖用买一价），确保立即成交
    - 只操作一次（用 order_done 标志位防止重复下单）
    - 账户密码从 config_local.py 读取（已 gitignore，不外泄）

用法：
    python ctp_demo.py
    （按 Ctrl+C 退出）
"""

from datetime import datetime
from pathlib import Path
from time import sleep

from vnpy_ctp.api import MdApi, TdApi
from vnpy_ctp.api.ctp_constant import (
    THOST_FTDC_D_Buy,
    THOST_FTDC_D_Sell,
    THOST_FTDC_OF_Open,
    THOST_FTDC_OF_Close,
    THOST_FTDC_OPT_LimitPrice,
    THOST_FTDC_TC_GFD,
    THOST_FTDC_VC_AV,
    THOST_FTDC_HF_Speculation,
    THOST_FTDC_CC_Immediately,
)

# 从本地配置读取账户信息（config_local.py 已被 .gitignore 排除，不入库）
try:
    import config_local as CFG
except ImportError:
    raise SystemExit(
        "未找到 config_local.py！\n"
        "请先复制 config_template.py 为 config_local.py，并填入你的账号密码。"
    )


# 要下单的合约（选一个最活跃的做闭环）
TRADE_SYMBOL = "rb2601"

# 流文件目录（CTP 底层要求一个可写的本地目录存放流文件）
FLOW_PATH = Path(__file__).parent / "flow"
FLOW_PATH.mkdir(exist_ok=True)


def now() -> str:
    """返回当前时间字符串（精确到毫秒），用作日志前缀。"""
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def log(msg: str) -> None:
    """带时间戳打印一行日志。"""
    print(f"[{now()}] {msg}", flush=True)


def make_login_req() -> dict:
    """构造登录请求体（交易/行情两个 API 的登录体完全一样，共用一份）。"""
    return {
        "UserID": CFG.USERID,
        "Password": CFG.PASSWORD,
        "BrokerID": CFG.BROKERID,
    }


class SimpleTdApi(TdApi):
    """交易 API：连接 -> 认证 -> 登录 -> 下单 -> 闭环。"""

    def __init__(self) -> None:
        super().__init__()                 # 必须调用：C++ 扩展对象初始化
        self.reqid: int = 0                # 请求编号，每次请求自增
        self.login_status: bool = False    # 是否登录成功
        self.connect_status: bool = False  # 是否已发起过连接
        self.order_done: bool = False      # 是否已完成开仓（防重复下单）
        self.last_price: float = 0.0       # 最近一次下单价格（备用）
        # 供行情 API 回调触发下单用
        self.pending_open: bool = False    # 是否待开仓

    # ---------- 连接与登录 ----------

    def connect(self) -> None:
        """发起网络连接（非阻塞，结果在 onFrontConnected 回调返回）。"""
        if not self.connect_status:
            self.createFtdcTraderApi(str(FLOW_PATH / "Td").encode("GBK"), True)
            self.registerFront(CFG.TD_ADDRESS)
            self.init()
            self.connect_status = True

    def authenticate(self) -> None:
        """发起终端认证（SimNow 要求，需 AppID + AuthCode）。"""
        req = {
            "UserID": CFG.USERID,
            "BrokerID": CFG.BROKERID,
            "AppID": CFG.APPID,
            "AuthCode": CFG.AUTH_CODE,
        }
        self.reqid += 1
        self.reqAuthenticate(req, self.reqid)

    def login(self) -> None:
        """发起登录请求。"""
        self.reqid += 1
        self.reqUserLogin(make_login_req(), self.reqid)

    # ---------- 回调 ----------

    def onFrontConnected(self) -> None:
        log("交易前置连接成功，开始终端认证")
        self.authenticate()

    def onFrontDisconnected(self, reason: int) -> None:
        log(f"交易前置连接断开，原因码 {reason}")

    def onRspAuthenticate(self, data: dict, error: dict, reqid: int, last: bool) -> None:
        if not error["ErrorID"]:
            log("终端认证成功，开始登录")
            self.login()
        else:
            log(f"终端认证失败：{error['ErrorID']} {error['ErrorMsg']}")

    def onRspUserLogin(self, data: dict, error: dict, reqid: int, last: bool) -> None:
        if not error["ErrorID"]:
            self.login_status = True
            log(f"交易账号登录成功，交易日 {data.get('TradingDay')}")
        else:
            log(f"交易登录失败：{error['ErrorID']} {error['ErrorMsg']}")

    # ---------- 下单 ----------

    def send_order(self, direction: str, offset: str, price: float, volume: int) -> int:
        """发送限价委托，返回报单引用（OrderRef）。"""
        self.reqid += 1
        order_ref = str(self.reqid)

        req = {
            "BrokerID": CFG.BROKERID,
            "InvestorID": CFG.USERID,
            "InstrumentID": TRADE_SYMBOL,
            "OrderRef": order_ref,
            "UserID": CFG.USERID,
            "OrderPriceType": THOST_FTDC_OPT_LimitPrice,
            "Direction": direction,
            "CombOffsetFlag": offset,
            "CombHedgeFlag": THOST_FTDC_HF_Speculation,
            "LimitPrice": price,
            "VolumeTotalOriginal": volume,
            "TimeCondition": THOST_FTDC_TC_GFD,
            "VolumeCondition": THOST_FTDC_VC_AV,
            "MinVolume": 1,
            "ContingentCondition": THOST_FTDC_CC_Immediately,
            "StopPrice": 0.0,
            "ForceCloseReason": "0",
            "IsAutoSuspend": 0,
            "RequestID": self.reqid,
        }
        self.reqOrderInsert(req, self.reqid)

        action = "买入开仓" if direction == THOST_FTDC_D_Buy else "卖出平仓"
        log(f"发出委托：{action} {TRADE_SYMBOL} {volume} 手 @ {price}（OrderRef={order_ref}）")
        return int(order_ref)

    def open_long(self, price: float) -> None:
        """买入开仓 1 手。"""
        self.send_order(THOST_FTDC_D_Buy, THOST_FTDC_OF_Open, price, 1)

    def close_long(self, price: float) -> None:
        """卖出平仓 1 手。"""
        self.send_order(THOST_FTDC_D_Sell, THOST_FTDC_OF_Close, price, 1)

    # ---------- 委托与成交回报 ----------

    def onRspOrderInsert(self, data: dict, error: dict, reqid: int, last: bool) -> None:
        """回报：报单被 CTP 拒绝（本地校验失败）。"""
        if error and error["ErrorID"]:
            log(f"报单被拒：{error['ErrorID']} {error['ErrorMsg']}")

    def onErrRtnOrderInsert(self, data: dict, error: dict) -> None:
        """回报：报单被交易所拒绝。"""
        if error and error["ErrorID"]:
            log(f"报单被交易所拒绝：{error['ErrorID']} {error['ErrorMsg']}")

    def onRtnOrder(self, data: dict) -> None:
        """回报：委托状态变化。"""
        log(f"委托回报：{data.get('InstrumentID')} 状态={data.get('OrderStatus')} "
            f"已成交={data.get('VolumeTraded')}/{data.get('VolumeTotalOriginal')}")

    def onRtnTrade(self, data: dict) -> None:
        """回报：成交。开仓成交后立即发起平仓，完成闭环。"""
        log(f"成交回报：{data.get('InstrumentID')} 方向={data.get('Direction')} "
            f"开平={data.get('OffsetFlag')} 价={data.get('Price')} 量={data.get('Volume')}")

        offset = data.get("OffsetFlag")

        # 开仓成交 -> 触发平仓
        if offset == THOST_FTDC_OF_Open and not self.order_done:
            self.order_done = True
            price = float(data.get("Price", 0))
            log(f"开仓成交，准备平仓（对手价 {price}）")
            # 稍等片刻，避免报单过快被拒
            self.reqid += 1
            sleep(0.5)
            self.close_long(price)

        # 平仓成交 -> 闭环完成
        elif offset == THOST_FTDC_OF_Close:
            log("★ 平仓成交，闭环完成：开仓 -> 平仓 -> 持仓归零")


class SimpleMdApi(MdApi):
    """行情 API：连接 -> 登录 -> 订阅 -> 打印推送 ->（首条行情触发下单）。"""

    def __init__(self, td_api: SimpleTdApi = None) -> None:
        super().__init__()
        self.reqid: int = 0
        self.login_status: bool = False
        self.connect_status: bool = False
        self.subscribed: list = []
        self.td_api = td_api               # 用于行情触发的下单联动

    def connect(self) -> None:
        if not self.connect_status:
            self.createFtdcMdApi(str(FLOW_PATH / "Md").encode("GBK"), True)
            self.registerFront(CFG.MD_ADDRESS)
            self.init()
            self.connect_status = True

    def login(self) -> None:
        self.reqid += 1
        self.reqUserLogin(make_login_req(), self.reqid)

    def subscribe(self, symbols: list) -> None:
        self.subscribed.extend(symbols)

    def onFrontConnected(self) -> None:
        log("行情前置连接成功，开始登录")
        self.login()

    def onFrontDisconnected(self, reason: int) -> None:
        self.login_status = False
        log(f"行情前置连接断开，原因码 {reason}")

    def onRspUserLogin(self, data: dict, error: dict, reqid: int, last: bool) -> None:
        if not error["ErrorID"]:
            self.login_status = True
            log(f"行情账号登录成功，交易日 {data.get('TradingDay')}，开始订阅 {self.subscribed}")
            for symbol in self.subscribed:
                self.subscribeMarketData(symbol)
        else:
            log(f"行情登录失败：{error['ErrorID']} {error['ErrorMsg']}")

    def onRspSubMarketData(self, data: dict, error: dict, reqid: int, last: bool) -> None:
        if error and error["ErrorID"]:
            log(f"订阅失败 {data.get('InstrumentID')}：{error['ErrorID']} {error['ErrorMsg']}")

    def onRtnDepthMarketData(self, data: dict) -> None:
        """行情推送：打印快照；首条目标合约行情触发开仓。"""
        def fmt(v, nd=1):
            try:
                return f"{v:.{nd}f}"
            except (TypeError, ValueError):
                return str(v)

        symbol = data["InstrumentID"]
        print(
            f"[{now()}] {symbol}"
            f" 最新={fmt(data['LastPrice'])}"
            f" 买一={fmt(data['BidPrice1'])}x{data['BidVolume1']}"
            f" 卖一={fmt(data['AskPrice1'])}x{data['AskVolume1']}"
            f" 成交量={data['Volume']} 持仓量={data['OpenInterest']}"
            f" 时间={data['UpdateTime']}.{data['UpdateMillisec']}",
            flush=True,
        )

        # 首条目标合约行情 -> 触发买入开仓（用卖一价，确保成交）
        if (
            self.td_api is not None
            and self.td_api.login_status
            and self.td_api.pending_open
            and symbol == TRADE_SYMBOL
        ):
            self.td_api.pending_open = False
            ask_price = float(data.get("AskPrice1", 0))
            if ask_price > 0:
                self.td_api.open_long(ask_price)


def main() -> None:
    """主流程：连接 -> 订阅 -> 打印行情 -> 触发下单闭环（不自动退出）。"""
    log(f"账户 {CFG.USERID} @ broker {CFG.BROKERID}")
    log(f"交易前置 {CFG.TD_ADDRESS}")
    log(f"行情前置 {CFG.MD_ADDRESS}")
    log(f"下单目标合约 {TRADE_SYMBOL}（首条行情触发买入开仓，成交后自动平仓）")

    td_api = SimpleTdApi()
    md_api = SimpleMdApi(td_api)

    log("发起交易前置连接...")
    td_api.connect()
    log("发起行情前置连接...")
    md_api.connect()

    md_api.subscribe(CFG.SYMBOLS)

    # 开启下单触发（等行情到达时执行）
    td_api.pending_open = True

    log("主线程进入等待，持续接收行情推送，按 Ctrl+C 退出")
    try:
        while True:
            sleep(1)
    except KeyboardInterrupt:
        log("退出")
        td_api.exit()
        md_api.exit()


if __name__ == "__main__":
    main()
