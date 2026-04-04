# GitHub 协作与权限管理

这份文档说明如何把当前公开仓库开放给队友协作，以及推荐的权限和分支保护策略。

## 一、怎么让队友可以 push

最直接的做法不是共享你的账号，而是把他们加为仓库协作者。

在 GitHub 网页端操作：

1. 打开仓库 `Settings`
2. 进入 `Collaborators and teams`
3. 点击 `Add people`
4. 输入队友的 GitHub 用户名
5. 选择权限

推荐权限：

- `Write`：适合大多数算法同学，可以推自己的分支、提 PR
- `Maintain`：适合负责人或核心维护者

一般不建议随便给：

- `Admin`

## 二、最合适的团队协作模式

推荐模式是：

- 所有人都能推自己的分支
- 默认不要直接推 `main`
- 所有改动通过 PR 合并到 `main`

这样做的好处：

- 不会互相覆盖
- 每条研究线都能单独 review
- 便于追踪和回滚

## 三、main 分支怎么保护

推荐在 GitHub 仓库设置里给 `main` 开分支保护。

建议项：

- 禁止直接 push 到 `main`
- 要求通过 Pull Request 合并
- 至少 1 个 review 才能合并

如果团队刚起步，也可以采用轻量版：

- 普通成员走 PR
- owner 紧急情况可处理

## 四、队友本地应该怎么开发

每个人：

1. clone 仓库
2. 新建自己的分支
3. 在对应 workspace 和共享代码目录工作
4. 提交 commit
5. 推送分支
6. 发 PR

分支命名建议：

- `black-box/<topic>`
- `white-box/<topic>`
- `gray-box/<topic>`
- `implementation/<topic>`

## 五、Git 身份怎么设置才会显示 GitHub 头像

本地 git 提交邮箱必须和 GitHub 账号已验证邮箱一致，或者使用 GitHub 的 noreply 邮箱。

检查：

```powershell
git config user.name
git config user.email
```

设置当前仓库本地身份：

```powershell
git config user.name "你的GitHub名字"
git config user.email "你在GitHub已验证的邮箱"
```

如果历史提交邮箱不对，未来新提交会修正，但旧提交不会自动变化。旧提交想修复，只能改历史并 force-push，需要谨慎。

## 六、对当前仓库的推荐

当前最合适的策略是：

- 你作为仓库 owner
- 三位算法同学都加为 collaborator
- 给 `Write`
- `main` 开保护
- 统一通过 PR 合并
