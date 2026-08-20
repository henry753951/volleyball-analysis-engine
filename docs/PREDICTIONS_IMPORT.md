# Predictions JSON 流式匯入

這套工具是獨立入口，不會取代或修改原本的 `volleyball-analysis-worker` 推論流程。它只讀取既有 predictions JSON，先建立約數 MB 的 frame offset 索引，再把每個 rally 片段轉成正式 Provider Work `ANALYSIS` 結果。

資料流程：

1. `plan` 串流掃描 JSON，以 pass/set/spike 類 play label 建立 rally；
   `l_winpoint/r_winpoint` 視為終止狀態，並保留前 2 秒、後 1 秒畫面。
2. `submit` 透過 Annotation WebSocket 建立 START / END / SUBMIT，Central 依序產生 clip 與 AI job。
3. `worker` 使用 AI Worker Token 連到 `/api/v2/ai/providers/ws`；每收到一個 job，只 seek/read 對應 JSON frame 範圍。
4. 每個 rally 完成後，Central 會建立 VAD1、frame chunks、pose evidence 與 UI processing update。人物框、逐幀動作與球位置可立即用於 Overlay；穩定的 pass / set / spike phase 會轉成可校正的 AI 擊球時間點與接球／舉球／攻擊球種。Court60 keypoints 會沿用正式 worker 的 homography、左右場判定與 path builder，產生可分析球路。UI 不必等整場 JSON 完成。

## 1. 建立索引與匯入計畫

Ubuntu / Docker 容器內路徑請換成實際 mount path：

```bash
python scripts/import_predictions.py plan \
  --predictions /data/match_predictions.json \
  --index /work/import/predictions.index.npz \
  --plan /work/import/plan.json
```

`plan.json` 會保存每個片段的 source frame 範圍、`rally_id` 與 `submission_id`。中斷後可直接重跑：已送出的片段會略過；若 START 已成功但 END 尚未完成，工具會從 WebSocket snapshot 接續該 draft。媒體索引暫時未就緒時，Annotation command 也會在 `--media-wait-seconds` 範圍內重試。來源 JSON 不會被改寫。

## 2. 連線並自動送出

本機開發環境可省略 `VOLLYAI_USER_TOKEN`。正式環境若 Annotation API 需要使用者 Bearer Token，請另外設定；AI Worker Token 不能取代使用者登入。

```bash
export VOLLYAI_TOKEN='UI 建立後只顯示一次的 AI Worker Token'
export VOLLYAI_USER_TOKEN='正式環境的使用者 Token（本機 DEV_AUTH 可省略）'

python scripts/import_predictions.py all \
  --plan /work/import/plan.json \
  --match-id 1322399f-91fa-42c8-92b2-683067101f3b \
  --capture-session-id 05d989a7-03ce-48a2-a450-006f2dc740a4 \
  --server-http https://volleyai.hsulab.net \
  --annotation-ws wss://volleyai.hsulab.net/ws/annotations \
  --server-ws 'wss://volleyai.hsulab.net/api/v2/ai/providers/ws?match_id=1322399f-91fa-42c8-92b2-683067101f3b' \
  --workspace /work/import/worker
```

先只驗證一個片段：

```bash
python scripts/import_predictions.py all \
  --plan /work/import/plan.json \
  --match-id 1322399f-91fa-42c8-92b2-683067101f3b \
  --capture-session-id 05d989a7-03ce-48a2-a450-006f2dc740a4 \
  --workspace /work/import/worker \
  --limit 1
```

也可拆成兩個 process，先啟動 `worker`，再執行 `submit`。這適合長時間匯入與容器重啟：

```bash
python scripts/import_predictions.py worker \
  --plan /work/import/plan.json \
  --server-ws 'wss://volleyai.hsulab.net/api/v2/ai/providers/ws?match_id=1322399f-91fa-42c8-92b2-683067101f3b' \
  --workspace /work/import/worker

python scripts/import_predictions.py submit \
  --plan /work/import/plan.json \
  --match-id 1322399f-91fa-42c8-92b2-683067101f3b \
  --capture-session-id 05d989a7-03ce-48a2-a450-006f2dc740a4 \
  --server-http https://volleyai.hsulab.net \
  --annotation-ws wss://volleyai.hsulab.net/ws/annotations
```

`match_id` WebSocket scope 很重要：這個 predictions worker 只會收到該場賽事的
`ANALYSIS` 工作，不會租用或拒絕其他場次原本排隊中的工作。

## 3. 重新產生既有匯入場次的球路

若場次已經匯入，但要用目前的 worker function 重新產生擊球時間、球種與球路，
可建立新的版本化 `AnalysisRun`。這不會改寫既有 `RallySubmission`、人工標記或快捷鍵流程：

```bash
export VOLLYAI_USER_TOKEN='正式環境的使用者 Token（本機 DEV_AUTH 可省略）'

python scripts/reprocess_predictions.py \
  --plan /work/import/plan.json \
  --match-id 1322399f-91fa-42c8-92b2-683067101f3b \
  --server-http https://volleyai.hsulab.net \
  --server-ws wss://volleyai.hsulab.net/api/v2/ai/providers/ws \
  --workspace /work/import/reprocess \
  --worker-count 4
```

若沒有設定 `VOLLYAI_TOKEN`，腳本會透過 operations API 建立暫時 AI Worker Token，
完成或失敗後都會自動刪除。`--match-id` 會自動加入 WebSocket URL，限制 Worker
只接收這一場的工作；多個 Worker 只提高同場的平行處理量，不會碰其他場次。

## 匯入限制

- JSON 沒有穩定的 person `track_id`，因此每個 rally 內以幾何 tracker 建立 analysis-local track；不宣稱跨 rally 身分一致。
- JSON 的 group activity 用於切 rally，也會經過最短持續時間、信心門檻、短雜訊合併與 ball-flight 對齊後，產生 `ai_detected` 擊球點；這些時間點與擊球球員仍是待人工校正的 AI 結果。
- group activity 不直接改比分。自動建立的 rally 保留 pending outcome。
- `l_*` 固定代表左場、`r_*` 固定代表右場；隊名由該 rally 的場上配置解析。例如本場左場為 IRI、右場為 ALG，不把隊名硬寫進通用匯入器。
- canonical player court positions 由 JSON 內的 Court60 keypoints 使用正式 worker 的 RANSAC homography 投影；相鄰擊球都有代表位置時，球路會標成 `complete`。投影證據不足時仍會保守標成 `unavailable`。
- UI 請同時開啟 `Overlay` 與設定中的 `下載 AI 分析資料`。選取已完成回合後，左上 `FRAME IDX` 應切換成該 clip 的局部 frame；若仍顯示整場 frame，重新選取回合或重新整理頁面。
- AI Worker WebSocket 是控制面；VAD1 與 evidence 依既有 job callback 以 HTTP multipart 上傳。影片與大型 JSON 不會塞進 WebSocket。
- 內部 evidence contract 仍會自動產生內容雜湊；CLI 不要求操作者提供模型 SHA，也不會用 SHA 下載模型。
