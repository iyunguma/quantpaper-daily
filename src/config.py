"""集中設定。想調整抓取範圍、排序偏好或模型，改這個檔案就好。"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"
SITE_DIR = ROOT / "site"


def _env(name: str, default: str) -> str:
    # GitHub Actions 裡沒設定的 Variables 會以空字串傳進來，視同未設定
    return os.getenv(name) or default


# ---------------------------------------------------------------- 抓取

# 每天最多選幾篇
DAILY_COUNT = int(_env("DAILY_COUNT", "10"))

# 只考慮最近幾天內首次投稿的論文。arXiv 週五、週六不發佈新論文，
# 往回看一週可以讓週末那幾期用前幾天沒選上的好文章補位。
LOOKBACK_DAYS = int(_env("LOOKBACK_DAYS", "7"))

# 每組查詢向 arXiv 要幾篇（依投稿時間由新到舊）
QFIN_FETCH = 150
CROSS_FETCH = 100

# arXiv 要求兩次請求間隔至少 3 秒，並建議標明身分
ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_DELAY_SECONDS = 3.0
# GitHub Actions 會自動填入 repo 網址；本地執行時用預設值
USER_AGENT = os.getenv("ARXIV_USER_AGENT") or "quant-paper-daily/2.0 (daily digest, 2 requests/day)"

QFIN_CATEGORIES = [
    "q-fin.TR",  # Trading and Market Microstructure
    "q-fin.CP",  # Computational Finance
    "q-fin.ST",  # Statistical Finance
    "q-fin.PM",  # Portfolio Management
    "q-fin.RM",  # Risk Management
    "q-fin.MF",  # Mathematical Finance
    "q-fin.PR",  # Pricing of Securities
]

# 資工、統計分類中，摘要提到這些詞組才納入候選
CROSS_CATEGORIES = ["cs.LG", "cs.AI", "cs.CE", "stat.ML", "econ.EM"]
CROSS_TERMS = [
    "trading strategy", "algorithmic trading", "limit order book",
    "market making", "portfolio optimization", "asset pricing",
    "stock prediction", "stock return", "financial time series",
    "cryptocurrency", "volatility forecasting", "backtest",
]

CATEGORY_LABELS = {
    "q-fin.TR": "交易與市場微結構",
    "q-fin.CP": "計算金融",
    "q-fin.ST": "統計金融",
    "q-fin.PM": "投組管理",
    "q-fin.RM": "風險管理",
    "q-fin.MF": "數理金融",
    "q-fin.PR": "證券定價",
    "q-fin.GN": "一般金融",
    "q-fin.EC": "經濟",
    "cs.LG": "機器學習",
    "cs.AI": "人工智慧",
    "cs.CE": "計算科學與工程",
    "stat.ML": "統計機器學習",
    "stat.ME": "統計方法",
    "econ.EM": "計量經濟",
    "econ.GN": "一般經濟",
}

# ---------------------------------------------------------------- 第一階段：關鍵字粗篩
# 以完整單字比對（"alpha" 不會命中 "alphabetical"），容許複數形。
# 標題命中算兩倍。這一階段只負責從上百篇候選中縮小範圍，
# 最終排序交給第二階段的語意重排。

KEYWORD_WEIGHTS = {
    # 策略與訊號
    "alpha": 3, "trading strategy": 8, "algorithmic trading": 8,
    "market making": 9, "market maker": 8, "statistical arbitrage": 9,
    "pairs trading": 7, "momentum": 5, "mean reversion": 6,
    "factor model": 7, "factor investing": 7, "cross-sectional": 4,
    "trading signal": 6, "backtest": 6, "backtesting": 6,
    "optimal execution": 8, "execution cost": 6, "slippage": 5,
    "transaction cost": 6,
    # 市場微結構
    "limit order book": 9, "order book": 7, "order flow": 8,
    "market microstructure": 8, "high-frequency": 7, "high frequency": 7,
    "liquidity": 4, "bid-ask spread": 5, "price impact": 7,
    # 方法
    "reinforcement learning": 7, "deep learning": 5, "transformer": 5,
    "neural network": 3, "large language model": 5, "llm": 5,
    "graph neural network": 4, "gradient boosting": 4,
    # 投組與風險
    "portfolio optimization": 7, "portfolio selection": 6, "risk parity": 6,
    "drawdown": 5, "sharpe ratio": 7, "hedging": 4,
    "volatility forecasting": 6, "realized volatility": 5,
    "implied volatility": 4, "regime": 4, "covariance estimation": 5,
    # 資產類別
    "cryptocurrency": 4, "bitcoin": 4, "futures": 3, "stock": 3,
    "equity market": 3, "option pricing": 3,
    # 實證品質訊號
    "empirical": 3, "out-of-sample": 6, "live trading": 7,
}

NEGATIVE_KEYWORDS = {
    "existence and uniqueness": -8, "viscosity solution": -7,
    "well-posedness": -7, "axiomatic": -5, "survey": -3,
    "procurement": -6, "vaccine": -8,
}

CATEGORY_BONUS = {
    "q-fin.TR": 10, "q-fin.CP": 7, "q-fin.PM": 7,
    "q-fin.ST": 6, "q-fin.RM": 4, "q-fin.MF": 1, "q-fin.PR": 2,
}

# 沒有語意重排時（例如未設 API key），關鍵字分數低於此值的不選
MIN_KEYWORD_SCORE = 8

# ---------------------------------------------------------------- 模型
# 使用 Gemini API 免費額度。免費額度以模型為單位計算，失敗的請求也算在內。
# 預設重排與摘要都用 Flash-Lite：2026-10 實測 gemini-3.8-flash 免費額度每天只有 20 次，
# 而且經常回傳 503 過載，一天就把額度耗光；Flash-Lite 則穩定、幾秒就回應。
# Google 汰換模型很快，模型被停用時到 https://ai.google.dev/gemini-api/docs/models
# 查新名稱，用 repo 的 Variables 覆蓋即可，不用改程式。
#
# 思考程度 minimal / low / medium / high：越高越仔細，但越慢、越容易用完額度。
# 各模型支援的等級不同，例如 gemini-3.8-flash 不支援 minimal。

# ---------------------------------------------------------------- 第二階段：語意重排
# 關鍵字分數最高的 RERANK_POOL 篇交給輕量模型逐篇評 0–10 分相關度，
# 一次請求即可。失敗時自動退回關鍵字排序。

RERANK_ENABLED = os.getenv("RERANK_ENABLED", "1") != "0"
RERANK_MODEL = _env("RERANK_MODEL", "gemini-3.5-flash-lite")
RERANK_THINKING = _env("RERANK_THINKING", "minimal")
RERANK_POOL = 40
RERANK_MAX_TOKENS = 6000

# 語意相關度低於此值的不選。寧可少於十篇，也不硬湊不相關的論文。
# 設成 0 就會永遠湊滿 DAILY_COUNT。
MIN_RELEVANCE = int(_env("MIN_RELEVANCE", "4"))

# ---------------------------------------------------------------- 摘要

SUMMARY_MODEL = _env("SUMMARY_MODEL", "gemini-3.5-flash-lite")
SUMMARY_THINKING = _env("SUMMARY_THINKING", "medium")

# 思考 token 也算在這個上限內，要留足空間
SUMMARY_MAX_TOKENS = 8000

# 同時摘要幾篇。免費額度的回應常要排隊，同時送出可以縮短總時間；
# 碰到每分鐘上限時程式會等待重試，調太高只會多等，不會出錯。
SUMMARY_CONCURRENCY = 3

# "abstract"：只讀 arXiv 摘要。
# "pdf"：下載全文 PDF 給模型讀，重點與數字精準得多，但輸入量大約是 50 倍，
#        免費額度的每分鐘 token 上限比較容易碰到。
SUMMARY_SOURCE = _env("SUMMARY_SOURCE", "abstract")
PDF_MAX_BYTES = 25 * 1024 * 1024

# 當天摘要失敗比例超過這個值就中止，不覆蓋網站，並讓 GitHub Actions 標成失敗
MAX_FAILURE_RATIO = 0.3

# ---------------------------------------------------------------- 網站

SITE_TITLE = "量化研究日報"
SITE_TAGLINE = "每天從 arXiv 挑出量化交易論文，讀完再告訴你重點"
SITE_BASE_URL = os.getenv("SITE_BASE_URL", "").rstrip("/")
FEED_DAYS = 14
