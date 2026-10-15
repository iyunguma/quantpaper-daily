"""共用測試資料。論文 ID、標題、分類取自 2026-10-15 arXiv q-fin 當日列表，
摘要為精簡改寫，僅供測試排序與解析用。"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from xml.sax.saxutils import escape

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

NOW = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)

REAL_PAPERS = [
    ("2610.20037", "Data-Driven Dynamic Portfolio Selection with Vine Copula Simulation and Reinforcement Learning",
     "q-fin.PM", "Vine-copula scenario generation is combined with reinforcement learning to build a dynamic portfolio policy penalising CVaR. On a seven-asset panel the ensemble outperforms classical benchmarks out-of-sample while reducing turnover and transaction costs.",
     "2026-10-14T16:02:11Z"),
    ("2609.34004", "RICE-Alpha: Reliability-Informed Correction with Event Graphs for LLM-Agent Stock Forecasting",
     "cs.LG", "A point-in-time stock scoring framework pairs a multi-view forecast with an event-based residual correction. On Nasdaq-100 and Hang Seng panels from 2024 to 2026 it beats LLM agents and momentum on portfolio metrics including Sharpe ratio.",
     "2026-10-13T09:12:00Z"),
    ("2610.20366", "Regime Detection in Latent Factor Models -- A Bayesian Decision-Theoretic Approach",
     "q-fin.MF", "Online regime detection with latent factors is framed as a Bayesian decision problem trading off false alarms and delay, solved with particle filtering and Monte Carlo dynamic programming.",
     "2026-10-14T11:40:00Z"),
    ("2610.19428", "Common Signals, Spurious Cross-Excitation, and Identification in Multivariate Hawkes Processes",
     "stat.ME", "Latent common signals can produce apparent cross-excitation in multivariate Hawkes models. Third-order cumulants aid identification; the method is applied to cryptocurrency transactions.",
     "2026-10-13T20:00:00Z"),
    ("2610.19254", "Beyond Time Induction: Structured Conditional Neural Solvers for Parametric Stochastic-Volatility Pricing",
     "cs.LG", "A structured conditional neural architecture solves parametric PDEs; tested on the Heston PDE it improves option value and sensitivity accuracy.",
     "2026-10-14T05:00:00Z"),
    ("2610.20000", "Interior smoothness and zero-variance geometry for finite-maturity Heston American puts",
     "q-fin.MF", "We study the exercise boundary of American puts under the Heston model without the Feller condition, establishing existence and uniqueness and smoothness at positive variance.",
     "2026-10-14T18:30:00Z"),
    ("2610.19045", "Long Term Pediatric Vaccine Tender Scheduling with Quantity Discounts",
     "econ.GN", "A vaccine market model schedules multi-year procurement tenders for low-income countries using UNICEF demand forecasts.",
     "2026-10-14T13:00:00Z"),
    ("2610.18921", "Creative destruction: Human-AI competition in an experimental market",
     "econ.GN", "In an experiment with 1,025 participants, AI-designed games outperformed those by humans, including professionals.",
     "2026-10-13T15:00:00Z"),
]

# 邊界案例
OLD_STYLE = ("q-fin/0701001", "Order Book Dynamics in Old Style Identifiers",
             "q-fin.TR", "A limit order book model with empirical high-frequency data.", "2026-10-14T10:00:00Z")
TOO_OLD = ("2609.11111", "Statistical Arbitrage with Limit Order Book Signals",
           "q-fin.TR", "A statistical arbitrage strategy on limit order book data with live trading results.",
           "2026-09-20T10:00:00Z")


def entry_xml(arxiv_id, title, primary, abstract, published, version=1):
    return f"""
  <entry>
    <id>http://arxiv.org/abs/{arxiv_id}v{version}</id>
    <updated>{published}</updated>
    <published>{published}</published>
    <title>{escape(title)}</title>
    <summary>  {escape(abstract)}
    </summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
    <link href="http://arxiv.org/abs/{arxiv_id}v{version}" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/{arxiv_id}v{version}" rel="related" type="application/pdf"/>
    <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="{primary}" scheme="http://arxiv.org/schemas/atom"/>
    <category term="{primary}" scheme="http://arxiv.org/schemas/atom"/>
  </entry>"""


ERROR_ENTRY = """
  <entry>
    <id>http://arxiv.org/api/errors#incorrect_id_format_for_1234</id>
    <title>Error</title>
    <summary>incorrect id format for 1234</summary>
  </entry>"""


def feed_xml(entries, total=None):
    body = "".join(entries)
    total = len(entries) if total is None else total
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <opensearch:totalResults xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">{total}</opensearch:totalResults>
  {body}
</feed>"""


@pytest.fixture
def real_feed():
    return feed_xml([entry_xml(*p) for p in REAL_PAPERS + [OLD_STYLE, TOO_OLD]] + [ERROR_ENTRY])


@pytest.fixture
def candidates(real_feed):
    from fetch_papers import parse_feed
    papers, _ = parse_feed(real_feed)
    return papers


# ---------------------------------------------------------------- 假的 Anthropic client

def text_response(text, stop_reason="end_turn", thinking=True):
    blocks = [SimpleNamespace(type="thinking", thinking="…")] if thinking else []
    blocks.append(SimpleNamespace(type="text", text=text))
    return SimpleNamespace(content=blocks, stop_reason=stop_reason)


class FakeClient:
    """依序回傳預先排好的回應；回應可以是函式（接收 kwargs）或例外。"""

    def __init__(self, responder):
        self.calls: list[dict] = []
        self._responder = responder
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        result = self._responder(kwargs)
        if isinstance(result, Exception):
            raise result
        return result
