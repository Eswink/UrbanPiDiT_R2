# R7 GPU bring-up 简报（已迁移）

本文件原为 goal 模式的长文源。**内容已迁至 [`docs/goals/gpu-bringup-3090.md`](goals/gpu-bringup-3090.md)**，
以符合 `docs/rules/artifact-storage.md` R-034 的命名与存放约定（goal 长文放 `docs/goals/`）。

路径保留在此，是因为 R-034 把它登记为「早于约定、不迁移」的例外，
且 `docs/rules/artifact-storage.md`、`docs/goals/README.md`、`docs/plans/0001-*.md`
都按此路径引用它。若把它们一并改指新路径，应作为一次独立的小改动。

**不再在此维护第二份任务定义** —— 两份内容会漂移成互相矛盾的事实（旧版本曾记录
HEAD `c36e3f7`、745 passed、CI run `35899051997`，而已被 `50954c9`、796 passed、
run `35976102003` 取代）。任何 GPU bring-up 任务的唯一权威定义是
[`docs/goals/gpu-bringup-3090.md`](goals/gpu-bringup-3090.md)。
