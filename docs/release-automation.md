# MUSA 版本自动发布流程

`deepselect-musa` 的 MUSA wheel 依赖 MUSA 工具链与 GPU，不能在 GitHub 托管 runner 上构建。
本仓库使用 **self-hosted MUSA runner** 完成构建和测试，PyPI 发布走 **Trusted Publishing**（无需保存 PyPI token）。

## 总体链路

```
main 上修改 deep_select/__version__.py 并 push
        │  (paths 触发)
        ▼
release-musa-auto.yml   runs-on: [self-hosted, linux, x64, musa, musa52]
  1. 读版本号；若 v<version> tag 已存在 → 跳过（幂等）
  2. 用 MUSA 构建容器构建 plain-version wheel（PyPI 不允许 local version）
  3. 真机 GPU smoke + MUSA 回归测试
  4. 创建 tag v<version> + GitHub Release（wheel 附件）
  5. 调用 publish-pypi.yml
        │
        ▼
publish-pypi.yml   runs-on: ubuntu-latest, environment: pypi
  从 GitHub Release 下载 wheel → PyPI Trusted Publishing 上传
```

## 一次性配置（已完成的部分会标注）

1. **MUSA self-hosted runner**（每台构建机一次）
   - 在 MUSA 构建机上执行：
     ```bash
     cd <repo>
     sudo -v
     GITHUB_PAT=<对仓库有 admin 权限的 token> \
       ./scripts/ci/musa/setup_github_runner.sh
     ```
   - 脚本会注册 repo-scoped runner，labels 为 `musa,musa52,s5000,py310`，
     并以 systemd service 方式常驻。
   - 检查：`sudo /data/actions-runner-deepselect/svc.sh status`
     或 `gh api repos/MooreThreads/DeepSelect/actions/runners`。

2. **GitHub Environment `pypi`**（已完成）：Settings → Environments → `pypi`，无需 secret。

3. **PyPI Trusted Publisher**（在 PyPI 后台配置）：
   | 字段 | 值 |
   | --- | --- |
   | PyPI Project Name | `deepselect-musa` |
   | Owner | `MooreThreads` |
   | Repository name | `DeepSelect` |
   | Workflow name | `publish-pypi.yml` |
   | Environment name | `pypi` |

## 日常发版（推荐）

```bash
# 1. 改版本号（PyPI 版本不可复用，必须递增）
$EDITOR deep_select/__version__.py      # __version__ = "0.0.2"
$EDITOR CHANGELOG.md                    # 记录该版本变更

# 2. 提交并 push 到 main
git add deep_select/__version__.py CHANGELOG.md
git commit -m "release: deepselect-musa 0.0.2"
git push origin main
```

push 后自动执行：构建 → GPU 测试 → tag `v0.0.2` → GitHub Release → PyPI 上传。
约 10–20 分钟（取决于构建机缓存）。

验证：

```bash
pip install --upgrade deepselect-musa==0.0.2
python -m deep_select.compat
```

## 手动/预演

- 只构建和测试、不发布（dry run）：
  GitHub → Actions → `release-musa-auto` → Run workflow → `dry_run = true`
- 只补发 PyPI（GitHub Release 已存在但 PyPI 没发）：
  Actions → `publish-pypi` → Run workflow → 填入 `tag`（如 `v0.0.2`）
- 重跑失败任务：直接 Re-run；若 tag 已创建但 PyPI 未发布，用上面的 `publish-pypi` 手动补发。

## 版本与平台约定

- **PyPI 只接受 public version**（不能用 `+musa5.2.0torch2.11.0.post1` 这类 local version）。
  自动流程构建的是 `deepselect_musa-<X.Y.Z>-cp310-cp310-linux_x86_64.whl`。
- **一个 PyPI 版本对应一个已验证栈**：MUSA SDK 5.2 / `torch_musa` 2.11.0.post1 / Python 3.10 / MTT S5000。
  换 torch 或 SDK 时：改构建镜像/约束 → 升版本号发布；旧版本保留。
- 需要同时提供其它栈或 local-label wheel 时，把额外 wheel 作为 GitHub Release 附件附加
  （不要上传到 PyPI，同名版本会被拒绝）；或后续新增 `deepselect-musa-torch<ver>` 独立包名。
- 版本号一经 PyPI 发布不可覆盖/删除重传；修复只能升版本。

## 安全注意事项

- self-hosted runner 只应在 **main push / workflow_dispatch** 的发布流程中使用；
  不要给 `pull_request` 工作流配置该 runner（公开仓库的 fork PR 代码不应在构建机上执行）。
- runner 以 `mccxadmin` + docker 组运行，能访问 MUSA 设备；机器本身应按内网可信环境管理。
- PyPI Trusted Publishing 使用短时 OIDC token，仓库不需要也没有保存 PyPI token。
