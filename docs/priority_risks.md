# 缺口與風險 — 優先順序（高 → 低）

依 **IR-AAP 常見情境**（內網模擬、`run_live`、儀表板、PyBullet）排序；若你的部署不同，請自行調整。

---

## P0 — 最高（對外或跨信任邊界時必做）

| 序 | 項目 | 風險 | 建議動作 |
|----|------|------|----------|
| 1 | **API 無認證、CORS 寬鬆** | 未授權遙控模擬／設定、內網誤暴露 | 僅內網或 VPN；若需對外：反向代理 + API Key / JWT、縮小 CORS、TLS |
| 2 | **機密經設定檔／環境變數外洩** | `.env`、token 被 commit | 維持 `.gitignore`、CI secret scan、文件提醒勿提交憑證 |

---

## P1 — 高（穩定性與誤用）

| 序 | 項目 | 風險 | 建議動作 |
|----|------|------|----------|
| 3 | **REST 與 World 類型不一致** | `GridWorld` 或精簡 world 掛同一套 API 可能例外或 500（例如 `_obstacle_ids`、`set_robot_position` 簽名） | 路由內統一 `hasattr` / 明確 400 說明；README 標註「REST 完整功能以 PhysicsWorld / MockPhysicsWorld 為準」 |
| 4 | **PyBullet 多進程 / 多 worker** | 崩潰、非決定性、GUI 衝突 | 維持 **單一** sim 進程；文件與 Docker 註明禁止水平擴展同一實例 |
| 5 | **操作與物理不同步認知** | 儀表板「瞬移／場景預設」與真機語意不同，現場誤判 | 文件區分「模擬專用 API」與真機適配層（`RealWorldAdapter`） |

---

## P2 — 中（品質與維護成本）

| 序 | 項目 | 風險 | 建議動作 |
|----|------|------|----------|
| 6 | **儀表板無 E2E** | 改版回歸靠人工、易漏 | 最小 Playwright：開 `/dashboard`、斷言說明鈕或 `/health` 連線狀態 |
| 7 | **發版煙霧僅文件** | 版本間環境差異未抓 | 沿用 `CONTRIBUTING.md` 的 Release smoke；CI 至少 `pytest` + `ruff` |
| 8 | **Gymnasium / SB3 警告** | 觀測空間無界、無 `Monitor` 影響實驗可比性 | RL 路線再收斂 `RobotEnv` 與 wrapper |

---

## P3 — 低（擴展與最佳化）

| 序 | 項目 | 風險 | 建議動作 |
|----|------|------|----------|
| 9 | **真機 / ROS / 網路延遲** | 單元測試無法涵蓋 | 整合測試環境、乾跑（dry-run）策略、重試與逾時文件化 |
| 10 | **負載與高頻 step** | WS 多開、API 狂打未驗證 | 有需求時再做 locust/k6；預設單使用者儀表板通常足夠 |

---

## 執行順序建議（一句話）

1. **先確認 API 暴露範圍** → 若不只內網，先做 **P0**。  
2. 日常開發：**P1** 的 world/API 一致性與 PyBullet 單實例。  
3. 專案成熟後補 **P2** E2E 與 RL 收斂。  
4. **P3** 依真機與上線規模再投入。

---

相關文件：[simulation_physics.md](simulation_physics.md)、[events.md](events.md)、根目錄 `CONTRIBUTING.md`（Release smoke）。
