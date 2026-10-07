# Selfplay-Transformer-based

三者間SAOP用のTransformer方策をPPOで学習し、永続Opponent Pool、PFSP、凍結snapshot、checkpoint/resumeを組み込んだ自己対戦実装です。

`Transformer-based`を移植元として使用していますが、実行時のsibling importはありません。`Selfplay-mipn-based`からは自己対戦の考え方とデータ構造だけを移植しています。元の2リポジトリは変更しません。

これはPPOによる第一段階の実装です。quantile critic、DSAC、スタイル制御、自然言語生成、SL warm start、KL正則化は含まれないため、論文版α-Negoの完全再現ではありません。

## 主な修正と機能

- 行動を `0 = Accept`、`1..N = joint bid`、`Discrete(N + 1)`へ統一
- `env.step()`をterminalまたは次の学習者手番まで進め、実行されない行動をPPOへ保存しない
- episodeごとにObserverを完全reset
- 最新の有効turnと全issueを集約するTransformer方策
- rolloutとPPO再評価で共通のAccept mask・log probabilityを使用
- scripted、historical snapshot、current frozen self-playのsource mixture
- 学習者のagreement、utility、lengthだけで更新するPFSP
- 左右slot、domain、utility case、scripted anchorを均衡化したpool評価
- atomic snapshot、pool JSON、version付きcheckpoint
- model、optimizer、pool、Python/NumPy/Torch CPU/CUDA、sampler RNG、scheduleのresume
- 既存集計と互換なTSV先頭7列

## セットアップ

Python 3.10を使用します。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

学習と評価はリポジトリルートから実行してください。通常は同梱済みの事前計算埋め込みを使用するため、OpenAI APIキーは不要です。新しいドメインの埋め込みを生成するときだけ`OPENAI_API_KEY`が必要です。

## 学習

今回の実験で使用するutility caseは`case1`〜`case3`だけです。タプルの左から学習エージェント、opponent 1、opponent 2のutilityを表します。

| case | 学習エージェント | opponent 1 | opponent 2 |
|---|---|---|---|
| case1 | utility1 | utility2 | utility3 |
| case2 | utility2 | utility3 | utility1 |
| case3 | utility3 | utility1 | utility2 |

`case4`〜`case6`はCLIと環境の両方で受け付けません。

### Expertモデル

```bash
python train.py \
  -a Boulware Linear \
  -i Laptop \
  --model-type expert \
  --utility-cases case1 \
  --total-timesteps 100000 \
  --save-path results/case1
```

### Generalモデル

```bash
python train.py \
  -a Boulware Conceder Linear Atlas3 \
  -i Laptop ItexvsCypress IS_BT_Acquisition Grocery thompson Car EnergySmall_A \
  --model-type general \
  --utility-cases case1 \
  --total-timesteps 300000 \
  --save-path results/case1
```

複数caseを一つの学習scheduleで回す場合は、例えば`--utility-cases case1 case2 case3`と指定します。同条件比較用には、他手法と同様にcaseごとに別checkpointを作ることを推奨します。

既定のsource mixtureは次のとおりです。

```text
current self-play : 0.0
scripted          : 0.5
snapshot          : 0.5
```

snapshotがまだ存在しない間はscriptedへ自動的に再正規化されます。論文比率に近い比較条件は次のように指定できます。

```bash
--self-play-probability 0.2 \
--scripted-probability 0.3 \
--snapshot-probability 0.5
```

current self-playはepisode開始時にdeep copyされ、`eval()`かつ全parameterが`requires_grad=False`の状態で固定されます。

`--pool-eval-freq 10000`の既定値では、rolloutを次の境界で短縮し、10,000、20,000、30,000…stepの丁度でpool評価を実行します。`num_envs`や`horizon`が境界の約数でなくても、最後のrolloutの環境数と長さを調整します。

## 保存名とデータ構造

他の手法と同じく、domain、対戦相手、timestampをディレクトリ名に使用します。

```text
results/case1/
└── Laptop_Boulware-Linear/
    └── 20261003-120000-TA/
        └── TransformerSelfPlay_Negotiator/
            ├── checkpoint.pt
            ├── config.json
            ├── training_log.tsv
            ├── tensorboard/
            ├── pool/
            │   ├── pool.json
            │   ├── stats.tsv
            │   └── snapshots/
            │       └── snapshot-10000.pt
            ├── evaluation/
            │   └── step-10000.tsv
            └── csv/
```

`checkpoint.pt`には累積step、モデル、optimizer、pool metadata、乱数状態、学習scheduleを保存します。snapshot pathはモデルディレクトリからの相対pathです。

## Resume

`--total-timesteps`は追加step数ではなく、resume後を含む累積到達stepです。

```bash
python train.py \
  --resume results/case1/Laptop_Boulware-Linear/<timestamp>-TA/TransformerSelfPlay_Negotiator \
  --total-timesteps 200000 \
  --device auto
```

domain、action mapping、Transformer構成、pool JSONがcheckpointと一致しない場合は読み込みを拒否します。

## 評価と結果出力

```bash
python test_negotiator.py \
  -m results/case1/Laptop_Boulware-Linear/<timestamp>-TA/TransformerSelfPlay_Negotiator \
  -a Boulware Linear \
  -i Laptop \
  --episodes 100 \
  --case case1 \
  --export-root results_selfplay-transformer-based
```

通常の評価ファイルは次に保存されます。

```text
<model_dir>/csv/<agent0>-<agent1>/<domain>/det=False_noise=False/*.tsv
```

`--export-root`を指定すると、集計用に次へコピーします。

```text
results_selfplay-transformer-based/
└── <expert|general>/<agent0>-<agent1>/<domain>/<case>/*.tsv
```

TSVの先頭列は次の既存契約を維持します。

```text
my_util, opp_util1, opp_util2, social, nash, agreement, step
```

自己対戦のopponent ID、source、score、seed、domain、caseは末尾列です。

`evaluation/step-*.tsv`は`Selfplay-mipn-based`と同じ次の16列順で出力します。

```text
my_util, opp_util1, opp_util2, social, nash, agreement, step,
style, opponent1_id, opponent2_id, opponent1_source, opponent2_source,
negotiation_score, seed, domain, case
```

複数の学習済みモデルから評価コマンドを生成する場合:

```bash
python command_generate.py results/case1 \
  --output run_command/run_test_case1.sh \
  --episodes 100 \
  --export-root results_selfplay-transformer-based
bash run_command/run_test_case1.sh
```

## 主要CLI

```text
--seed
--resume
--pool-path
--max-pool-size
--pfsp-alpha
--uniform-mix
--allow-duplicate-opponents / --no-allow-duplicate-opponents
--self-play-probability
--scripted-probability
--snapshot-probability
--snapshot-freq
--pool-eval-freq
--pool-eval-episodes
--dominance-tolerance
--utility-cases
```

## テスト

```bash
pytest -q
```

テストには行動契約、学習者手番境界、Observer reset、Accept mask、全issue/履歴pooling、PFSP、順序付きpair sampling、pool永続化・pruning、frozen current self-play、snapshot生成、checkpoint resume、TSV列整合性が含まれます。

## case1〜case3本実験

本実験はseed 0で、expert 210モデル（7既知ドメイン×10相手ペア×3 cases、各100,000 step）とgeneral 3モデル（3 cases、各300,000 step）を学習します。generalの学習対象は既知7ドメインだけで、未知5ドメインは事前の形状検証と最終評価だけに使用します。

```text
results/caseN/models/expert/<pair>/<domain>/Transformer_Negotiator/
results/caseN/models/general/Transformer_Negotiator/
results/caseN/evaluation/<expert|general>/<pair>/<domain>/caseN/
```

学習コマンド213件と評価条件570件は、それぞれ次で確認できます。

```bash
DRY_RUN=1 bash run_command/run_case_experiments.sh
DRY_RUN=1 bash run_command/evaluate_case_experiments.sh
```

RTX 5090用の実験イメージを構築し、GPUごとに独立したDockerコンテナを使う2 shardのジョブレベル並列実行は次で開始します。起動スクリプトは両GPUに既存compute processがある場合は安全のため開始しません。

```bash
bash run_command/launch_case_experiments_tmux.sh
tmux attach -t selfplay-transformer-cases
```

tmuxのウィンドウは`gpu0`と`gpu1`、ログは`results/experiment_logs/train-gpu0.log`と`train-gpu1.log`です。完了済みcheckpointはskipし、未完了checkpointは累積stepからresumeします。

全checkpointの学習後、最終評価570条件を2 GPUへ分割して実行します。

```bash
bash run_command/launch_case_evaluations_tmux.sh
tmux attach -t selfplay-transformer-evaluation
```

評価ログは`results/experiment_logs/evaluate-gpu0.log`と`evaluate-gpu1.log`、完了マーカーは`results/experiment_status/eval-shard-*-of-2.done`です。

## 論文・MiPN自己対戦版との差分

- 本実装はPPOであり、論文と`Selfplay-mipn-based`のDSAC/quantile criticではありません。
- opponent utilityとsocial welfareは事後ログだけに残し、報酬、PFSP、snapshot採用判定には使用しません。
- snapshot採用の第一キーは学習者utility、差がtolerance内の場合だけ交渉scoreを使います。
- current learnerをlive参照する相手は作りません。
- 第二段階のDSAC化はPPO自己対戦の比較実験後に独立した変更として行います。
