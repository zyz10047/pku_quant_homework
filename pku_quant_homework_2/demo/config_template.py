"""
SimNow 仿真账户配置 —— 模板（TEMPLATE）

本文件是「模板」，可以安全地提交到 git 仓库。
请勿把真实密码写进本文件。

用法：
    复制本文件为 config_local.py，把里面的占位值换成你的真实信息。
    config_local.py 已加入 .gitignore，不会被提交。

获取信息的地方：https://www.simnow.com.cn/
    - 用户ID / 经纪商代码：登录后「个人中心」查看
    - 服务器地址：SimNow 官方公开地址（见下方默认值）
"""

# ===== 账户凭据 =====
USERID = "your_userid"          # 例：277642
PASSWORD = "your_password"      # ⚠️ 真实密码只写在 config_local.py
BROKERID = "9999"               # SimNow 固定为 9999

# ===== 服务器地址 =====
# 第一组（需在交易时段使用）
TD_ADDRESS = "tcp://180.168.146.187:10130"
MD_ADDRESS = "tcp://180.168.146.187:10131"

# 备用地址（7x24 环境，随时可连，测试推荐）
TD_ADDRESS_7X24 = "tcp://180.168.146.187:10140"
MD_ADDRESS_7X24 = "tcp://180.168.146.187:10141"

# ===== CTP 认证信息（SimNow 固定值）=====
APPID = "simnow_client_test"
AUTH_CODE = "0000000000000000"

# ===== 要订阅的合约（成交活跃）=====
# 螺纹钢、豆粕、白糖、甲醇等主力合约
SYMBOLS = [
    "rb2601",    # 螺纹钢
    "m2601",     # 豆粕
    "SR601",     # 白糖
    "MA601",     # 甲醇
]
