# 量化研究日報

每天早上自動從 arXiv 抓量化交易相關論文，挑出最值得看的（最多十篇），用 Claude 產生繁體中文重點，發佈成靜態網站與 RSS。

沒有伺服器、沒有資料庫。排程跑在 GitHub Actions，網站放在 GitHub Pages，每日資料以 JSON 存回 repo。

## 運作流程

```
GitHub Actions（每天台北時間 07:17）
  │
  ├─ fetch_papers.py   向 arXiv 查兩組：q-fin 全分類、資工/統計分類中與交易相關者
  │                    → 約 250 篇候選
  │
  ├─ rank.py           1. 排除已收錄、超過 7 天的論文
  │                    2. 關鍵字粗篩，取前 40 篇
  │                    3. Claude Haiku 5.5 逐篇評 0–10 分交易相關度（一次請求）
  │                    4. 相關度 ≥ 4 的取前 10 篇；量少的日子寧缺勿濫
  │                    （語意評分失敗時自動退回關鍵字排序）
  │
  ├─ summarize.py      Claude Sonnet 5.5 以 structured outputs 產生固定格式的重點：
  │                    一句話結論、問題、方法、發現、交易意涵、限制、落地可用性 1–5
  │
  ├─ store.py          寫入 data/papers/YYYY-MM-DD.json ← 唯一的資料來源
  │
  └─ build_site.py     產生今日頁、每日歸檔、歸檔索引、RSS，部署到 GitHub Pages
```

哪些論文已經收錄過，直接從 `data/papers/` 推算，不另外維護清單，所以資料不會不同步。

## 設定步驟

1. **建立 repo**：把這個資料夾推上 GitHub。public repo 的 Actions 與 Pages 都免費。
2. **放 API key**：Settings → Secrets and variables → Actions → New repository secret，名稱 `ANTHROPIC_API_KEY`，值為你在 https://platform.claude.com 申請的金鑰。
3. **開啟 Pages**：Settings → Pages → Source 選 **GitHub Actions**。
4. **跑第一次**：Actions → 「每日論文更新」→ Run workflow。約三分鐘後網站上線，網址會顯示在執行結果頁面上。之後每天自動更新。

網站網址會自動帶入 RSS，不需要另外設定。

## 手動執行的選項

在 Actions 頁面按 Run workflow 時可以選：

| 選項 | 用途 |
|---|---|
| daily | 產生今天這一期。今天已經產生過就只重新部署，不會換一批論文 |
| force | 重做今天這一期（例如改了排序設定想重選） |
| repair | 把所有摘要失敗的論文重新摘要一次 |
| build | 只重建網站（改了樣式或模板時用） |

執行失敗時（arXiv 連不上、API key 失效、摘要失敗超過三成），網站會維持前一天的內容，GitHub 會寄信通知你。

## 本地測試

```bash
pip install -r requirements-dev.txt
python -m pytest                       # 45 項測試，不需要網路與 API key

# 看看今天會選到哪些論文，不產生摘要、不寫任何檔案
python src/run_daily.py --dry-run      # 沒有 API key 時只用關鍵字排序

export ANTHROPIC_API_KEY=sk-ant-...
python src/run_daily.py --dry-run      # 有 key 時會加上語意評分，並列出每篇的評分理由
python src/run_daily.py                # 完整跑一次
python -m http.server 8000 -d site     # 在 http://localhost:8000 預覽
```

## 常見調整

設定都在 `src/config.py`，大多也能用 repo 的 Variables（Settings → Secrets and variables → Actions → Variables）覆蓋，不用改程式。

| 想做的事 | 怎麼改 |
|---|---|
| 讀全文 PDF，重點更精準 | Variables 新增 `SUMMARY_SOURCE` = `pdf`（費用見下方） |
| 一定要湊滿十篇 | `MIN_RELEVANCE` 改成 `0` |
| 每天篇數 | `DAILY_COUNT` |
| 偏重某個主題 | 調高 `KEYWORD_WEIGHTS` 裡對應的權重，並在 `rank.py` 的 `RERANK_SYSTEM` 寫明你的偏好 |
| 摘要更快更省 | `SUMMARY_EFFORT` 改成 `low`，或 `SUMMARY_MODEL` 改成 `claude-haiku-5-5` |
| 摘要品質更高 | `SUMMARY_MODEL` 改成 `claude-opus-5-5` |
| 改執行時間 | `.github/workflows/daily.yml` 的 cron（UTC，台北時間減 8 小時） |

**關於執行時間**：arXiv 的新論文大約在美東午夜才進 API，也就是台北中午 12 點到 1 點。早上 7 點執行時拿到的是前一批，等於延遲一天，但不會漏掉任何論文。想要最新的一批，把 cron 改成 `"17 5 * * *"`（台北 13:17）。

## 費用估算

依 2026 年 10 月官方價目（Sonnet 5.5 每百萬 token 輸入 $2、輸出 $10；Haiku 5.5 更便宜一個數量級）粗估，實際依論文長度與思考量浮動：

| 模式 | 每天 | 每月 |
|---|---|---|
| 讀摘要（預設） | 約 US$0.2–0.4 | 約 US$6–12 |
| 讀全文 PDF | 約 US$1.5–3 | 約 US$45–90 |

語意排序每天一次，費用可以忽略。最新價格見 https://platform.claude.com/docs/en/about-claude/pricing 。GitHub Actions 與 Pages 在 public repo 免費。

## 從第一版升級

第一版有幾個會讓系統實際上無法運作的錯誤（模型名稱錯誤導致摘要全部失敗、Pages 部署設定缺漏）。如果你已經部署過：

1. 用這一版的檔案整個覆蓋 repo。
2. 刪除 `data/seen.json`（已不再使用）。
3. 在 Actions 執行一次 **repair**，把舊資料中失敗的摘要補上。

## 已知限制

- **只涵蓋 arXiv**。SSRN 與期刊沒有開放 API。arXiv q-fin 已涵蓋大部分量化預印本。
- **預設只讀摘要**。摘要沒寫的數字不會出現在重點裡；需要的話改用 PDF 模式。
- **相關度由模型判斷**，偶爾會有見仁見智的取捨。dry-run 會列出每篇的評分理由，方便你調整提示詞。
- **arXiv 週五、週六不發佈新論文**，週末那幾期由前幾天沒選上的論文補位，數量可能少於十篇。
