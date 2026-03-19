# 模擬場景、測試物與物理 | Simulation, test objects & physics

工業機器人／手臂在模擬裡通常需要：**可布置的場景**、**可偵測的物件**、以及**接近真實的物理**（重力、碰撞、摩擦、時間步長等）。IR-AAP 以 **PyBullet** 為核心，並用 **YAML profile + REST** 組裝場景。

---

## 1. 場景裡有什麼？

| 類型 | 說明 | 典型用途 |
|------|------|----------|
| **靜態障礙** `obstacles` | `place_obstacle` / YAML，**質量為 0** 的方塊 | 牆、棧板、固定設備 |
| **可動測試物** `dynamics` | `spawn_dynamic_box` / YAML，**有質量**，受重力與碰撞影響 | 自由落體、滾落、堆疊、抓取前後位移 |
| **機器人（球體 AGV）** | 多台命名機器人，外力移動 | 導航、避障、多車 |
| **URDF 手臂** | `URDFRobot` 與現有 `PhysicsWorld` 共用 client | 關節控制、內建 `box_arm_with_gripper`（2 旋轉關節 + `gripper_slide` 滑動關節） |

目前預設 **AGV 本體**在模擬裡是**球體**；**手臂**需透過 URDF 另載入，與球體可並存於同一 PyBullet client。

---

## 2. 在 YAML 裡布置（profile）

### 靜態障礙（不會倒）

```yaml
world:
  obstacles:
    - {x: 4, y: 0, z: 0.5}
```

### 物理參數（時間步長、重力）

較小 `timestep` → 較穩定、較慢；`gravity_z` 預設約 **-9.81**（m/s² 量級，依你的單位一致即可）。

```yaml
world:
  physics:
    timestep: 0.004167    # 約 1/240 s
    gravity_z: -9.81
```

### 可動測試物（例：自由落體）

從空中放一個方塊，模擬迴圈每步 `step_physics()` 後會下落並與地面／障礙碰撞。

```yaml
world:
  dynamics:
    - {x: 0.5, y: 0, z: 2.0, mass: 0.4, half_extent: 0.06}
```

- `mass`：公斤級（與 PyBullet 慣例一致時行為較直覺）  
- `half_extent`：半邊長（立方體）

執行：`iraap serve --profile your.yaml`。API **`GET /api/v1/state`** 會多回傳 **`dynamics`**（位置、線速度），方便儀表板或外掛做「偵測」邏輯。

### 可選：YAML 載入手臂（與 REST 夾爪）

在 profile 頂層加 **`manipulator`**（僅 **PyBullet**、`client_id >= 0` 時生效；**mock** profile 會自動略過）：

```yaml
manipulator:
  enabled: true
  id: demo_arm
  x: 2.0
  y: -1.0
  z: 0.15
  urdf_path: null   # null = 內建 URDF
```

啟動後可用：

- **`GET /api/v1/manipulator/state`** — 手臂狀態（含 `joint_positions`）  
- **`POST /api/v1/manipulator/gripper`** — JSON `{"openness": 0.0}`～`1.0` 控制 `gripper_slide`（示範用，無真實夾持約束）

Legacy 前綴 **`/api/manipulator/...`** 同樣可用。

---

## 3. 執行期再加物件

- **靜態障礙**：`POST /api/v1/obstacles/add`、`DELETE /api/v1/obstacles/clear`  
- **可動測試物（方塊）**：`POST /api/v1/dynamics/box`，JSON 例如  
  `{"x":0,"y":0,"z":2,"mass":0.3,"half_extent":0.05}`  
- **清空所有可動物**：`DELETE /api/v1/dynamics`  
- **手臂夾爪**（需 profile 已啟用 `manipulator` 且 `create_app` 帶入 `urdf_arm`）：`GET/POST /api/v1/manipulator/state`、`/api/v1/manipulator/gripper`

（Legacy 路徑同樣掛在 `/api/dynamics/box`、`/api/dynamics`。）亦可於程式直接呼叫 `PhysicsWorld.spawn_dynamic_box(...)`。

---

## 4. 感測與「對物件偵測」

- **距離 / 雷達**：`physics_distance`、`physics_lidar` 對場景射線，可打到靜態障礙與動態剛體（視 PyBullet 碰撞形狀）。  
- **相機**：`PhysicsCamera` 等插件從模擬取影像/深度類資料（見 `plugins/physics_camera.py`）。  
- **語意偵測**：可在插件裡對 `dynamics` 的 id 或標籤做對應，或發佈自訂事件到 Event Bus。

---

## 5. 物理層面你還會遇到什麼？

Bullet 常見調整方向（進階需直接 PyBullet API 或擴充 `PhysicsWorld`）：

- **摩擦 / 恢復係數**：`changeDynamics`  
- **關節馬達、阻尼**：URDF + `URDFRobot` 控制模式  
- **軟體、皮帶**：超出預設剛體範圍，需另選引擎或簡化模型  

**自由落體**只是動力學的一例；同一套 **`dynamics`** 也可測碰撞彈開、斜坡滑動（需加斜面幾何）等。

---

## 6. 建議工作流（整合商／測試）

1. 用 **`/wizard` 或 `iraap init`** 起一份 YAML。  
2. 加上 **`physics`** 與 **`dynamics`** 區塊做重現案例。  
3. 用 **Recorder** 錄事件，對照 `state.dynamics` 與感測事件除錯。  
4. 真機時：保留 **同一事件契約**，把模擬外掛換成硬體外掛。

瀏覽 **`/dashboard`** 時，**World** 分頁可圖形化操作：瞬移球體機器人（`POST /api/v1/robots/position`）、生成／清空 dynamics、讀寫夾爪（若已啟用 manipulator），並可開啟定時同步場景狀態。

更多 API 列表見專案 **README**；事件型別見 **`docs/events.md`**。
