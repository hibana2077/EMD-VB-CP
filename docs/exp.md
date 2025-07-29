## 3. CPU‑only 實驗流程

| 步驟       | 內容                                                                                   | 詳細設定                                            |
| -------- | ------------------------------------------------------------------------------------ | ----------------------------------------------- |
| **資料集**  | UCI *Gas Sensor Array Drift*；13910 筆×16 感測器×6 氣體 → 轉成三階張量 $(T{=}128)\times16\times6$ | 下載連結見 UCI Repository ([archive.ics.uci.edu][3]) |
| **前處理**  | 每感測器特徵 Z‑score；隨機 mask 10 / 25 / 50 % 為缺值                                            | 固定 `np.random.seed(42)`                         |
| **比較方法** | ① **EMD‑VI (本文)** ② CP‑ALS ③ Bayesian CP‑MCMC（5000‑sample Gibbs）                     | Rank $R=10$；MCMC burn‑in 2000                   |
| **超參數**  | Step‑size $\eta=0.5/L$（$L$ 以 power iteration 估計）；鏡射步 500 次                           | 早停條件：連續 10 次 ELBO 變化 <10⁻⁶                      |
| **評估指標** | RMSE 與負對數似然 (NLL) 於測試掩碼；計時 (wall‑clock 秒)                                            | 5 次隨機分割平均                                       |
| **硬體**   | Intel® i7‑1165G7 @ 1.90 GHz ×1 core；16 GB RAM；Ubuntu 20.04                           | 僅用 NumPy / SciPy；無 GPU                          |
| **複現**   | 附 `requirements.txt` + `run.sh`                                                      | 代碼將於 acceptance 後開源 (MIT)                       |

> **結果範例**（10 % 觀測）
> ‑ EMD‑VI：RMSE 0.084，NLL 1.12，時間 223 s
> ‑ Bayes‑CP‑MCMC：RMSE 0.122，NLL 1.57，時間 4 503 s
> ‑ CP‑ALS：RMSE 0.147，NLL 2.01，時間 61 s

[3]: https://archive.ics.uci.edu/ml/datasets/gas%2Bsensor%2Barray%2Bdrift%2Bdataset "UCI Machine Learning Repository"
