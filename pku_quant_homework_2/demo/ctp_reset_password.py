"""
SimNow 首次登录强制改密 —— 一次性脚本。

背景：
    新注册的 SimNow 账号，首次通过 CTP 交易系统登录时会报：
        140 CTP:首次登录必须修改密码，请修改密码后重新登录。
    这是 CTP 的一次性安全要求 —— 网站能登 ≠ 交易系统能登，两者密码体系分开。

本脚本做什么：
    1. 连接交易前置 -> 终端认证 -> 调用 ReqUserPasswordUpdate
    2. 把密码"改成它自己"（旧密码 == 新密码），只为走完首次改密流程
    3. 打印结果后退出

安全说明：
    - 密码只从 config_local.py 读取，不硬编码、不打印、不写日志、不上传
    - 脚本读取的 NEW_PASSWORD 默认与 PASSWORD 相同（即"改成原密码"）

用法：
    python ctp_reset_password.py
"""

from datetime import datetime
from pathlib import Path
from time import sleep

from vnpy_ctp.api import TdApi

try:
    import config_local as CFG
except ImportError:
    raise SystemExit("未找到 config_local.py，请先确认本地配置存在。")

# 新密码：默认与原密码相同（走完流程即可）
NEW_PASSWORD = getattr(CFG, "NEW_PASSWORD", None) or CFG.PASSWORD

FLOW_PATH = Path(__file__).parent / "flow"
FLOW_PATH.mkdir(exist_ok=True)


def now() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


class PasswordUpdater(TdApi):
    """只做一件事：连接 -> 认证 -> 改密。"""

    def __init__(self) -> None:
        super().__init__()
        self.reqid: int = 0
        self.done: bool = False

    def connect(self) -> None:
        self.createFtdcTraderApi(str(FLOW_PATH / "TdReset").encode("GBK"), True)
        self.registerFront(CFG.TD_ADDRESS)
        self.init()

    def authenticate(self) -> None:
        req = {
            "UserID": CFG.USERID,
            "BrokerID": CFG.BROKERID,
            "AppID": CFG.APPID,
            "AuthCode": CFG.AUTH_CODE,
        }
        self.reqid += 1
        self.reqAuthenticate(req, self.reqid)

    def update(self) -> None:
        """发起改密请求：旧密码 -> 新密码。"""
        req = {
            "UserID": CFG.USERID,
            "BrokerID": CFG.BROKERID,
            "OldPassword": CFG.PASSWORD,
            "NewPassword": NEW_PASSWORD,
        }
        self.reqid += 1
        self.reqUserPasswordUpdate(req, self.reqid)

    # ---------- 回调 ----------

    def onFrontConnected(self) -> None:
        log("交易前置连接成功，开始终端认证")
        self.authenticate()

    def onFrontDisconnected(self, reason: int) -> None:
        log(f"交易前置断开，原因码 {reason}")

    def onRspAuthenticate(self, data: dict, error: dict, reqid: int, last: bool) -> None:
        if error and error.get("ErrorID", 0) != 0:
            log(f"终端认证失败：[{error['ErrorID']}] {error['ErrorMsg']}")
            self.done = True
            return
        log("终端认证成功，发起改密请求")
        self.update()

    def onRspUserPasswordUpdate(self, data: dict, error: dict, reqid: int, last: bool) -> None:
        if error and error.get("ErrorID", 0) != 0:
            log(f"改密失败：[{error['ErrorID']}] {error['ErrorMsg']}")
        else:
            log("改密成功！该账号已完成 CTP 首次改密流程。")
            log("现在可以用 ctp_demo.py 正常登录交易系统了。")
        self.done = True


def main() -> None:
    log(f"账户 {CFG.USERID} @ {CFG.TD_ADDRESS}")
    log("执行 CTP 首次登录改密流程（旧密码 -> 新密码）")
    if NEW_PASSWORD == CFG.PASSWORD:
        log("  （新密码与旧密码相同：仅为走完流程）")
    else:
        log("  （新密码与旧密码不同：已在 config_local.py 配置 NEW_PASSWORD）")

    api = PasswordUpdater()
    api.connect()

    # 最多等 60 秒
    for _ in range(600):
        if api.done:
            break
        sleep(0.1)
    else:
        log("超时：60 秒内未收到响应")

    api.exit()
    log("结束")


if __name__ == "__main__":
    main()
