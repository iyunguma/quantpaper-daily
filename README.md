# 量化研究日報

每天早上從 arXiv 過去一年的量化交易相關論文中，挑出主題對口、內容也扎實的（最多十篇），用 Gemini 產生繁體中文重點，發佈成靜態網站與 RSS。

沒有伺服器、沒有資料庫。排程跑在 GitHub Actions，網站放在 GitHub Pages，每日資料以 JSON 存回 repo。模型使用 Gemini API 的免費額度，整套系統不用花錢。

## 運作流程

```
GitHub Actions（每天台北時間 07:17）
  │
  ├─ fetch_papers.py   向 arXiv 分頁查兩組過去一年的論文：q-fin 全分類、資工/統計分類中與交易相關者
  │                    → 約 2,500 篇候選，每天約 4 次請求
  │
  ├─ rank.py           1. 排除已收錄的論文
  │                    2. 還沒評估過的論文做關鍵字粗篩，取前 40 篇
  │                    3. Gemini 3.5 Flash-Lite 逐篇標主題與實證依據、評兩個 0–10 分（一次請求）：
  │                       交易相關度，以及內容價值（實證是否扎實、有沒有新意、結論可不可信）
  │                       只靠模擬資料的內容價值最高 6 分；績效好到不可信的最高 4 分（不會入選）
  │                    4. 連同先前評過、還沒收錄的論文一起排名：相關度 ≥ 4 且內容價值 ≥ 5 的，
  │                       依兩者總分取前 10 篇，同一主題最多 3 篇；不夠就寧缺勿濫
  │                    （評估失敗時改用先前的評分，再不行才退回關鍵字排序）
  │
  ├─ summarize.py      Gemini 3.5 Flash-Lite 以 structured outputs 逐篇產生固定格式的重點：
  │                    一句話結論、問題、方法、發現、交易意涵、限制、落地可用性 1–5
  │
  ├─ store.py          寫入 data/papers/YYYY-MM-DD.json ← 網站內容的唯一來源
  │                    評分另存 data/evaluations/YYYY-MM-DD.json
  │
  └─ build_site.py     產生今日頁、每日歸檔、歸檔索引、RSS，部署到 GitHub Pages
```

哪些論文已經收錄過，直接從 `data/papers/` 推算，不另外維護清單，所以資料不會不同步。評過分的論文記在 `data/evaluations/`，不會重評；當天沒選上的好論文，隔天繼續和新評的論文競爭。

## 設定步驟

1. **建立 repo**：把這個資料夾推上 GitHub。public repo 的 Actions 與 Pages 都免費。
2. **放 API key**：到 https://aistudio.google.com/apikey 用 Google 帳號免費建立金鑰（不用綁信用卡）。回到 repo 的 Settings → Secrets and variables → Actions → New repository secret，名稱 `GEMINI_API_KEY`，值貼上金鑰。
3. **開啟 Pages**：Settings → Pages → Source 選 **GitHub Actions**。
4. **跑第一次**：Actions → 「每日論文更新」→ Run workflow。Gemini 免費額度的回應有時要排隊，大約 5 到 20 分鐘後網站上線，網址會顯示在執行結果頁面上。之後每天自動更新。

網站網址會自動帶入 RSS，不需要另外設定。

## 手動執行的選項

在 Actions 頁面按 Run workflow 時可以選：

| 選項 | 用途 |
|---|---|
| daily | 產生今天這一期。今天已經產生過就只重新部署，不會換一批論文 |
| force | 重做今天這一期（例如改了排序設定想重選） |
| repair | 把所有摘要失敗的論文重新摘要一次 |
| build | 只重建網站（改了樣式或模板時用） |

執行失敗時（arXiv 連不上、API key 失效、免費額度用完、摘要失敗超過三成），網站會維持前一天的內容，GitHub 會寄信通知你。

## 本地測試

```bash
pip install -r requirements-dev.txt
python -m pytest                       # 86 項測試，不需要網路與 API key

# 看看今天會選到哪些論文，不產生摘要、不寫任何檔案
python src/run_daily.py --dry-run      # 沒有 API key 時只用關鍵字排序

export GEMINI_API_KEY=...
python src/run_daily.py --dry-run      # 有 key 時會加上語意評估，列出每篇的分數、主題與理由，以及分數分布
python src/run_daily.py                # 完整跑一次
python -m http.server 8000 -d site     # 在 http://localhost:8000 預覽
```

## 常見調整

設定都在 `src/config.py`。下表標「Variables」的項目，也可以在 repo 的 Settings → Secrets and variables → Actions → Variables 新增同名變數覆蓋，不用改程式；刪掉變數就回到預設值。

| 想做的事 | 怎麼改 |
|---|---|
| 讀全文 PDF，重點更精準 | Variables：`SUMMARY_SOURCE` = `pdf`（較容易碰到額度上限，見下方） |
| 一定要湊滿十篇 | Variables：`MIN_RELEVANCE` = `0`、`MIN_VALUE` = `0` |
| 對內容價值更挑剔 | Variables：`MIN_VALUE` 調高（預設 5，滿分 10） |
| 往回找多久 | Variables：`LOOKBACK_DAYS`（預設 365 天） |
| 同一主題每天最多幾篇 | Variables：`MAX_PER_TOPIC`（預設 3）。主題清單在 `rank.py` 的 `TOPICS` |
| 改了評分標準想全部重評 | 把 `rank.py` 的 `RUBRIC_VERSION` 加一，舊的評分會作廢 |
| 網站上出現中國大陸用語 | 在 `src/terms.py` 的 `TERM_FIXES` 加一行，摘要與評選理由都會自動替換 |
| 每天篇數 | Variables：`DAILY_COUNT` |
| 摘要更快、更省額度 | Variables：`SUMMARY_THINKING` = `low` |
| 摘要更仔細 | Variables：`SUMMARY_THINKING` = `high` |
| 換模型（例如 Google 停用了預設模型） | Variables：`SUMMARY_MODEL`、`RERANK_MODEL`，名稱查 https://ai.google.dev/gemini-api/docs/models |
| 偏重某個主題 | 調高 `KEYWORD_WEIGHTS` 裡對應的權重，並在 `rank.py` 的 `RERANK_SYSTEM` 寫明你的偏好 |
| 改執行時間 | `.github/workflows/daily.yml` 的 cron（UTC，台北時間減 8 小時） |

思考程度（`SUMMARY_THINKING`、`RERANK_THINKING`）可設 `minimal` / `low` / `medium` / `high`，但各模型支援的不同，例如 `gemini-3.8-flash` 不支援 `minimal`。換模型時一併確認。

**關於執行時間**：arXiv 的新論文大約在美東午夜才進 API，也就是台北中午 12 點到 1 點。早上 7 點執行時拿到的是前一批，等於延遲一天，但不會漏掉任何論文。想要最新的一批，把 cron 改成 `"17 5 * * *"`（台北 13:17）。

## 免費額度

每天的用量是語意排序 1 次加摘要約 10 次，預設都用 `gemini-3.5-flash-lite`，共用同一個模型的每日額度。免費額度以模型為單位計算，**失敗的請求也算在內**。GitHub Actions 與 Pages 在 public repo 也免費。

摘要原本想用品質稍好的 `gemini-3.8-flash`，但 2026 年 10 月實測它的免費額度每天只有 20 次，而且經常回傳 503 過載，重試幾輪就把當天額度耗光。想試試看的話，在 Variables 設 `SUMMARY_MODEL` = `gemini-3.8-flash`。

Google 沒有公開固定的免費額度數字，而且調整過好幾次。你的實際額度請看 https://aistudio.google.com/rate-limit 。程式對兩種上限的處理方式：

- **每分鐘上限**：依 Gemini 回傳的建議秒數等待後重試，通常不影響當天結果。
- **今日額度用完**：立刻停止呼叫，剩下的論文標成摘要失敗。失敗超過三成時當天不發佈，網站維持前一天；沒超過就照常發佈，之後在 Actions 執行 **repair** 補上。

如果額度常常不夠，可以在 Google AI Studio 為這個專案啟用付費，改為依用量計費，價格見 https://ai.google.dev/gemini-api/docs/pricing 。

## 從第一版升級

第一版有幾個會讓系統實際上無法運作的錯誤（模型名稱錯誤導致摘要全部失敗、Pages 部署設定缺漏）。如果你已經部署過：

1. 用這一版的檔案整個覆蓋 repo。
2. 刪除 `data/seen.json`（已不再使用）。
3. 在 Actions 執行一次 **repair**，把舊資料中失敗的摘要補上。

## 已知限制

- **只涵蓋 arXiv**。SSRN 與期刊沒有開放 API。arXiv q-fin 已涵蓋大部分量化預印本。
- **預設只讀摘要**。摘要沒寫的數字不會出現在重點裡；需要的話改用 PDF 模式。
- **相關度與內容價值由模型判斷**，偶爾會有見仁見智的取捨。內容價值只根據摘要與 arXiv 附註（頁數、接受的會議、期刊），看不到全文的實驗細節。網站上每篇都列出評選理由，dry-run 也會列出，方便你調整提示詞。
- **前幾週會先消化過去一年累積的好論文**。之後能選的主要是每天新公告的論文，夠格的不一定有十篇，數量可能變少。
- **免費額度的內容會被 Google 用於改進產品**（Gemini API 價目頁標示；付費方案不會）。這裡送出的只有公開的 arXiv 標題、摘要與 PDF，以及 `src/` 裡的提示詞。
- **模型名稱會過期**。Google 汰換模型很快，預設模型被停用時，摘要會全部失敗並在 Actions 記錄裡顯示 404。照「常見調整」用 Variables 換成新模型即可。
