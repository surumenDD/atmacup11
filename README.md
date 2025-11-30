# atmacup11 study repo

AtmaCup#11 の画像回帰コンペを題材に、ベースラインからモデル拡張・スタッキングまで一連の学習パイプラインを整理した個人向けリポジトリです。PyTorch Lightning と Hydra を用いた実験スクリプト、OOF 生成、2nd レベルの Ridge スタッキング、探索用ノートブックを含みます。

## リポジトリ構成
- `experiments/`: 各モデルの学習スクリプトと Hydra 設定。ResNet ベースの回帰 (`exp001_resnet18d_reg` など) や ViT/DINO などの派生実験をディレクトリ単位で管理しています。【F:experiments/exp001_resnet18d_reg/run.py†L1-L208】【F:experiments/exp001_resnet18d_reg/config.yaml†L1-L24】
- `stacking/`: ベースモデルの OOF とテスト予測を集約し、Ridge 回帰でメタ学習するスクリプト。【F:stacking/stack_reg_v1.py†L13-L116】
- `utils/`: パス管理 (`EnvConfig`)、ロギング、処理時間計測などの共通ユーティリティ。【F:utils/env.py†L1-L15】
- `input/`: 公式配布データを置く場所 (Git 管理外)。
- `outputs/`: 学習済みモデル、OOF、サブミッション、スタッキング結果の保存先 (Git 管理外)。
- `notebook/`: 特徴量確認やアイデア検証用のノートブック。

## 依存関係と実行環境
- Python, PyTorch, timm, PyTorch Lightning, Hydra, scikit-learn, pandas, PIL などを利用します。【F:experiments/exp001_resnet18d_reg/run.py†L1-L52】
- Weights & Biases にロギングするため、API キーを設定しておくと記録が残ります (デバッグ時は `exp.debug=true` で無効化可能)。【F:experiments/exp001_resnet18d_reg/run.py†L128-L150】

## データ配置
1. `input/` 直下に学習・評価用メタデータ (`train.csv`, `test.csv`) と `atmaCup#11_sample_submission.csv` を配置します。【F:experiments/exp001_resnet18d_reg/run.py†L214-L262】
2. 画像ファイルは `input/photos/` に `object_id.jpg` 形式で置きます (JPEG/PNG 拡張子を自動検出)。【F:experiments/exp001_resnet18d_reg/run.py†L70-L114】【F:experiments/exp001_resnet18d_reg/run.py†L276-L316】
3. デフォルトの入出力パスは `utils.env.EnvConfig` または Hydra 設定から上書きできます。【F:utils/env.py†L1-L15】【F:experiments/exp001_resnet18d_reg/config.yaml†L9-L23】

## 実験の流れ
1. **学習/OOF 生成**: 各実験ディレクトリの `run.py` を Hydra で実行します。
   ```bash
   # 例: ResNet18d 回帰 (fold=0 のみ実行、設定は config.yaml を使用)
   python experiments/exp001_resnet18d_reg/run.py

   # 例: すべての fold を学習したい場合
   python experiments/exp001_resnet18d_reg/run.py exp.folds=[0,1,2,3,4]
   ```
   - StratifiedGroupKFold で fold 分割を作成し、fold ごとに Lightning で学習します。【F:experiments/exp001_resnet18d_reg/run.py†L214-L316】
   - OOF 予測と fold 別 RMSE を計算し、`outputs/experiments/<exp_name>/.../oof_*.csv` に保存します。【F:experiments/exp001_resnet18d_reg/run.py†L318-L358】
   - ベスト checkpoint を使ってテスト推論し、サブミッションを `submission_*.csv` として出力します。【F:experiments/exp001_resnet18d_reg/run.py†L360-L394】

2. **メタ学習 (スタッキング)**:
   - `stacking/stack_reg_v1.py` で BASE_EXPERIMENTS に列挙した実験の OOF/Sub を読み込み、Ridge 回帰で 2nd レベル学習を行います。
   - スタック後の OOF と `submission_stack_reg_v1.csv` を `outputs/stacking/stack_reg_v1/` に書き出します。【F:stacking/stack_reg_v1.py†L13-L116】【F:stacking/stack_reg_v1.py†L118-L181】

## 実験ディレクトリの扱い
- 新しいモデルを試したい場合は、既存ディレクトリをコピーし、`ExpConfig` のハイパーパラメータや transforms を調整してください。【F:experiments/exp001_resnet18d_reg/run.py†L19-L77】【F:experiments/exp001_resnet18d_reg/config.yaml†L19-L23】
- `expXXX_...` というディレクトリ名は、そのまま出力名やスタッキング設定のキーに使われます。`stack_reg_v1.py` の `BASE_EXPERIMENTS` を合わせて更新してください。【F:stacking/stack_reg_v1.py†L13-L42】

## 出力物の目安
- `outputs/experiments/<exp>/<hydra_run>/oof_*.csv`: object_id, fold, target, oof_pred を含む OOF 予測。【F:experiments/exp001_resnet18d_reg/run.py†L331-L358】
- `outputs/experiments/<exp>/<hydra_run>/submission_*.csv`: サンプルサブミッション形式の予測。【F:experiments/exp001_resnet18d_reg/run.py†L360-L394】
- `outputs/stacking/stack_reg_v1/`: スタック済み OOF / サブミッション。【F:stacking/stack_reg_v1.py†L118-L181】

## メモ
- 画像サイズ・学習率・エポック数・fold などは Hydra CLI で上書き可能です。再現性のため seed も設定しています。【F:experiments/exp001_resnet18d_reg/run.py†L19-L208】
- `notebook/` には特徴確認や前処理アイデアのメモを残しています。必要に応じて入力パスを自分の環境に合わせてください。
