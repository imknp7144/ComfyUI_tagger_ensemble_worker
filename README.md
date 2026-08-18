# Tagger Ensemble Worker ユーザーマニュアル

ComfyUI本体プロセス内で、8種類のHeavyタガー(CUDA/ONNX/timmベースの画像タグ推定モデル)を
切り替えて使うためのカスタムノード集です。本ドキュメントは実機での動作確認・デバッグを経て、
実際の挙動に基づいて作成しています。

---

## 1. ノード一覧

### `Tagger Ensemble Worker Setup`(内部名: `TaggerWorkerSetup`)

モデルの配置状況確認・VRAM予算設定を行う管理ノード。ワークフローの先頭に1つ置き、
実行して`status_report`を確認する使い方を想定しています。

| パラメータ | 型 | 既定値 | 説明 |
|---|---|---|---|
| `max_vram_gb` | FLOAT | 4.0 | 全Heavyタガー合計のVRAM予算(目安)。この値を超えそうになると、最も長く使われていないモデルからアンロードされる(LRU方式) |
| `enable_gpl_models` | BOOLEAN | False | GPL-3.0ライセンスのモデル(`at_eva02`, `at_convnext_huge`)を有効化するか。ONにする場合はGPLがこの拡張機能の配布物に伝播する可能性があることに同意した上で行うこと |

`status_report`出力には、各モデルの状態(`## モデル状態`)と現在のVRAM使用状況(`## VRAM使用状況`)
が表示されます。

### `Tagger Worker (Heavy)`(内部名: `TaggerWorkerHeavy`)

実際に画像を1枚タグ付けするノード。

| パラメータ | 型 | 既定値 | 説明 |
|---|---|---|---|
| `image` | IMAGE | - | 入力画像(バッチの場合は先頭1枚のみ処理) |
| `model_id` | COMBO | - | 使用するモデル(下記「4. モデル一覧」参照) |
| `use_compile` | BOOLEAN | False | `torch.compile()`を使うか(timmバックエンドのモデルのみ有効。`at_eva02`/`at_convnext_huge`) |
| `use_best_threshold` | BOOLEAN | False | ONにすると、以下の`threshold_*`を全て無視し、モデル配布元が計測した単一の推奨閾値(`best_threshold`)を使う |
| `threshold_general` | FLOAT | 0.35 | 一般タグの採用閾値(`use_best_threshold=False`のとき有効) |
| `threshold_character` | FLOAT | 0.60 | キャラクタータグの採用閾値 |
| `threshold_copyright` | FLOAT | 0.50 | 作品名(著作権)タグの採用閾値 |
| `top_n_raw_scores` | INT | 30 | `raw_scores_json`出力に含める上位スコアの件数(0で出力なし) |
| `device` | COMBO | AUTO | 実行デバイス。詳細は「6. deviceについて」参照 |
| `max_tags` | INT | 200 | `tags`出力に含める最大タグ数(0で無制限)。閾値を超えたタグ数がこれより多い場合、スコア上位から切り詰める |

出力は`tags`(カンマ区切りの文字列)と`raw_scores_json`(JSON Lines形式、`{"tag":..., "score":...}`)の2つです。
`tags`は他のノード(例: [ComfyUI-Danbooru-Prompt-Formatter](https://github.com/imknp7144/ComfyUI-Danbooru-Prompt-Formatter)
の`combine_mode`)に接続して複数モデルの結果を統合できます。**このノード自体は複数モデルの合議は行いません。**

---

## 2. 初回セットアップ手順

1. `Tagger Ensemble Worker Setup`ノードをワークフローに追加し、一度実行する
2. `status_report`に`[ACTION REQUIRED]`と表示されたモデル(同意が必要な配布元のもの)を、
   下記「4. モデル一覧」の配布元から手動ダウンロードし、指定パスに配置する
   (配置先は拡張機能フォルダ直下の`models/<model_id>/`)
3. 再度`Setup`ノードを実行し、対象モデルが`[OK]`・`state=LOADABLE`になっていることを確認する
4. `Tagger Worker (Heavy)`ノードで`model_id`を選び、画像を接続して実行する

`state`は「ファイルが揃って登録できたか」を示すもので、「実際にCUDAで動くか」とは別です。
実際の動作可否(`VALIDATED`/`FAILED`)と使用デバイス(`provider=CUDA_READY`等)は、
`Tagger Worker (Heavy)`で一度ロードを試みるまで反映されません。

---

## 3. モデル一覧と特性

| model_id | 語彙数 | gated | ライセンス | 特徴・注意点 |
|---|---|---|---|---|
| `cl_v1` | 約51,213 | ❌自動DL | Apache-2.0 | 入力サイズ448px(384pxではない点に注意) |
| `cl_v2` | 約106,536 | ✅要手動配置 | 独自ライセンス(再配布禁止) | 語彙数が非常に多く、統計的に閾値超えタグが多くなりやすい。`max_tags`での切り詰めや`use_best_threshold`の活用を推奨 |
| `dtq_l16` | 約11,424 | ✅要手動配置 | DINOv3 License(Meta) | DINOv3ベース |
| `dtq_b16` | 約11,424 | ✅要手動配置 | DINOv3 License(Meta) | 同上、より軽量 |
| `oppai_v11` | 約19,294 | ❌自動DL | Apache-2.0 | **タグのcategory情報を持たない(全タグgeneral扱い)。`threshold_character`/`threshold_copyright`は実質効かない** |
| `wd_eva02_l` | 約10,861 | ❌自動DL | Apache-2.0 | category情報あり、閾値は正常に機能 |
| `at_eva02` | 約12,476 | ✅要手動配置+GPL同意 | **GPL-3.0** | timmバックエンド(ONNXではない) |
| `at_convnext_huge` | - | ✅要手動配置+GPL同意 | **GPL-3.0** | timmバックエンド、比較的重い |

配布元URLと配置ファイル名の詳細は次章「4. 配布元・配置ファイル一覧」を参照してください。

---

## 4. 配布元・配置ファイル一覧

`[ACTION REQUIRED]`(gated、要手動配置)のモデルは、以下の配布元から該当ファイルをダウンロードし、
拡張機能フォルダ直下の`models/<model_id>/`へ、記載のファイル名で配置してください。
❌自動DLのモデルは`Setup`ノード実行時に自動でダウンロードされるため、手動作業は不要です。

| model_id | 配布元 | 配置するファイル | 配置先 |
|---|---|---|---|
| `cl_v1` ❌自動DL | [cella110n/cl_tagger](https://huggingface.co/cella110n/cl_tagger)(`cl_tagger_1_02/`フォルダ) | `model.onnx`, `tag_mapping.json` | `models/cl_v1/` |
| `cl_v2` ✅要手動配置 | [cella110n/cl_tagger_v2](https://huggingface.co/cella110n/cl_tagger_v2) | `model.onnx`, `model.onnx.data`(ONNX外部データ、忘れずに), `model_vocabulary.json` | `models/cl_v2/` |
| `dtq_l16` ✅要手動配置 | [realphongha/danbooru-tag-query](https://huggingface.co/realphongha/danbooru-tag-query)(`models/DanbooruTagQuery_l16_448x448/`フォルダ) | `model.onnx`, `tag_to_id.json`, `tag_category.json` | `models/dtq_l16/` |
| `dtq_b16` ✅要手動配置 | [realphongha/danbooru-tag-query](https://huggingface.co/realphongha/danbooru-tag-query)(`models/DanbooruTagQuery_b16_448x448/`フォルダ) | `model.onnx`, `tag_to_id.json`, `tag_category.json` | `models/dtq_b16/` |
| `oppai_v11` ❌自動DL | [Grio43/OppaiOracle](https://huggingface.co/Grio43/OppaiOracle)(`V1.1_onnx/`フォルダ) | `model.onnx`, `selected_tags.csv` | `models/oppai_v11/` |
| `wd_eva02_l` ❌自動DL | [SmilingWolf/wd-eva02-large-tagger-v3](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3) | `model.onnx`, `selected_tags.csv` | `models/wd_eva02_l/` |
| `at_eva02` ✅要手動配置+GPL同意 | [animetimm/eva02_large_patch14_448.dbv4-full](https://huggingface.co/animetimm/eva02_large_patch14_448.dbv4-full) | `model.safetensors`, `selected_tags.csv` | `models/at_eva02/` |
| `at_convnext_huge` ✅要手動配置+GPL同意 | [animetimm/convnextv2_huge.dbv4-full](https://huggingface.co/animetimm/convnextv2_huge.dbv4-full) | `model.safetensors`, `selected_tags.csv` | `models/at_convnext_huge/` |

`dtq_l16`/`dtq_b16`はDINOv3ベースのため、配布元Hugging Faceページでの利用規約への同意
(Meta社のDINOv3 License)が必要です。`at_eva02`/`at_convnext_huge`はGPL-3.0ライセンスのため、
`Setup`ノードの`enable_gpl_models`をONにする必要があります(GPLがこの拡張機能の配布物に
伝播する可能性があることに同意した上で行ってください)。

各モデルのライセンス種別は上記「3. モデル一覧と特性」の表を参照してください。

---

## 5. deviceについて

`AUTO` / `GPU` / `CPU`の3択です。

- **AUTO**: CUDA→DirectMLの順で使えるものを自動選択し、無ければCPU
- **GPU**: 明示要求。同じ優先順で探し、見つからなければ警告を出してCPUへフォールバック
- **CPU**: 強制CPU

CUDAとDirectMLは同じonnxruntimeインストール内には共存できません(どちらか一方のビルドしか
入れられない)。そのため`GPU`を選んでも、実際にどちらが使われるかは環境(`pip install
onnxruntime-gpu`か`onnxruntime-directml`か)によって決まります。timmバックエンドの
モデル(`at_eva02`/`at_convnext_huge`)はCUDA/CPUのみ対応です(DirectMLは`torch-directml`が
別途必要なため未対応。指定すると自動でAUTOへ読み替えられます)。

GPU EPでのセッション生成に失敗した場合は自動でCPUへ再試行するため、ノードがクラッシュすることは
ありません。ただしCPU動作は速度が大きく落ちます。

---

## 6. ログの読み方

コンソールログには`[TEW][種別]`のプレフィックスが付きます。トラブル報告時はこの`[TEW]`行を
含めて共有してください。

| プレフィックス | 内容 |
|---|---|
| `[TEW][LOAD]` | モデルのロード開始/検証/コミット/ロールバック |
| `[TEW][ONNX]` | 実際に選択されたExecution Provider・ONNX入出力形状 |
| `[TEW][TORCH]` | timmバックエンドの実行デバイス |
| `[TEW][TAGS]` | タグリスト・カテゴリの読み込み結果(形式自動判定の結果) |
| `[TEW][PREPROCESS]` | 前処理仕様(入力サイズ・正規化パラメータ等) |
| `[TEW][INFER]` | 推論1回ごとの入出力形状・所要時間・スコア分布 |
| `[TEW][THRESHOLD]` | 閾値通過タグ数(`kept=X/Y`)。タグ内容そのものはログに出ない |
| `[TEW][VRAM]` | VRAM使用量(実測/推定の別を明示) |
| `[TEW][UNLOAD]` | モデルのアンロード |

---

## 7. トラブルシューティング

### CUDAが使えず`CPU_FALLBACK`になる

1. `[TEW][ONNX]`ログの`diagnosis=`行を確認する
2. `No registered plugin EP device found`が出る場合、CUDA Toolkitは入っていても
   **cuDNNが別途必要**(CUDA Toolkitのインストーラには含まれない)。以下で確認できる:
   ```
   python -c "import torch; import onnxruntime as ort; ort.print_debug_info()"
   ```
   `List of loaded DLLs`に`cudnn64_9.dll`等が含まれているか確認する
3. それでも解決しない場合、GPU EPでのセッション生成自体が失敗している可能性がある。
   その場合は自動でCPUへフォールバックし処理は継続されるため、実害としては速度低下のみ
4. onnxruntime-gpuのバージョンとCUDA/cuDNNのバージョン対応は
   [公式互換表](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
   を参照

### `tags`出力が大量(数百〜数千件)になる

- 特に`cl_v2`(語彙数106,536)で起きやすい。語彙が大きいほど、閾値をたまたま超えるタグの
  絶対数が統計的に増えるため
- `max_tags`(既定200)で自動的に上位N件へ切り詰められる。それでも多いと感じる場合は
  `threshold_general`を上げるか、`use_best_threshold=True`にしてモデル推奨の閾値を使う
- `[TEW][THRESHOLD] kept=X/Y`ログで、実際に何件が閾値を超えたか確認できる

### `threshold_character`/`threshold_copyright`を上げても効果がない

- `oppai_v11`は既知の制限としてcategory情報を持たないため、常に`threshold_general`のみが
  適用される(上記「3. モデル一覧」参照)
- 他のモデルで同様の症状が出る場合は、`[TEW][TAGS] category schema=...`ログで
  `category_count`や`value_distribution`を確認する。`0`件、または全て`{0: 全件数}`に
  なっている場合はカテゴリ判定に失敗している可能性がある

### 既存ワークフローのノードでウィジェットの値がずれて表示される

- ComfyUIは保存済みワークフローのウィジェット値を配列の位置(インデックス)で復元するため、
  拡張機能のアップデートでウィジェットが増減すると位置がずれることがある
- 該当ノードを一度削除し、ノードメニューから新規に配置し直せば解消する
  (新規配置ノードは現在のウィジェット定義に基づいて作られるため、位置ズレが起きない)

---

## 8. 既知の制限事項

- **`oppai_v11`**: タグのcategory情報を持たないため、`threshold_character`/
  `threshold_copyright`による絞り込みができない(全タグがgeneral扱い)。この点はモデル配布物側の
  制約であり、拡張機能側での修正はできない
- **`cl_v1`/`cl_v2`のcategoryラベル**: 一部タグに`"Model"`/`"Quality"`のような未知のラベルが
  少数(数十件程度)存在し、この分だけカテゴリ未設定(general扱い)になる。全体からすると
  ごく僅かであり実用上の影響は小さい
- **`cl_v1`の前処理パラメータ**: 配布元に正規化パラメータの明記が無いため、`cl_v2`と同じ設定
  (SigLIP2想定)を暫定採用している。入力サイズ(448px)は実機のエラーメッセージから逆算して
  特定済みだが、mean/std等の細部は未検証
- **`at_eva02`/`at_convnext_huge`の前処理**: 公式の`dghs-imgutils`パイプラインを
  「正方形パディング+直接リサイズ」で近似している。より正確な前処理が必要な場合は
  `dghs-imgutils`の`create_torchvision_transforms`への置き換えを検討すること
