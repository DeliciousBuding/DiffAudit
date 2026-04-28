# Download Naming Policy

这份文档只回答一个问题：

`Download/` 里的目录名，什么算规范，什么只是历史/来源痕迹？

---

## 1. 结论先行

当前 `Download/` 的结构方向是对的：

- `shared/ black-box/ gray-box/ white-box/ manifests/` 这一层是清晰的
- 原始数据、权重、supplementary、papers 分桶也是清晰的

真正还不够统一的地方，不在“大层级”，而在第三层资产名：

- 有的目录名偏 **项目消费语义**
- 有的目录名偏 **上游来源语义**
- 有的目录名偏 **压缩包/网页来源语义**

因此当前正确做法不是立刻乱重命名，而是先明确：

1. 哪些名字是 canonical consumer name
2. 哪些名字只是 source-trace name
3. 以后新资产应该怎么命名

---

## 2. Canonical Rule

`Download/` 的命名规则应该固定成三层：

```text
Download/
  <scope>/
    <bucket>/
      <asset-name>/
```

其中：

- `<scope>` 只能是：
  - `shared`
  - `black-box`
  - `gray-box`
  - `white-box`
- `<bucket>` 只能是：
  - `datasets`
  - `weights`
  - `supplementary`
  - `papers`
- `<asset-name>` 应优先使用 **项目消费语义**，而不是下载来源名字

推荐格式：

- 全小写
- kebab-case
- 尽量短
- 优先写“这是什么资产”，不是“它从哪来”

例如：

- `stable-diffusion-v1-5`
- `clip-vit-large-patch14`
- `google-ddpm-cifar10-32`
- `secmi-cifar-bundle`
- `recon-assets`

---

## 3. Source Trace Rule

如果一个目录只是为了保留来源痕迹，而不是项目主消费入口，那么它不应该冒充 canonical asset name。

这类东西应该满足下面规则：

- 最好放在 canonical asset 目录下面
- 或至少在文档里明确标成 source-trace / raw archive

推荐内部子层：

```text
<asset-name>/
  raw/
  contents/
  notes/
```

含义：

- `raw/`
  - 原始压缩包、网页下载文件、OneDrive zip、7z 分卷
- `contents/`
  - 解压后的上游原样内容
- `notes/`
  - 来源说明、许可证、手工下载备注

---

## 4. Current Canonical Names

当前这些名字可以视为 canonical，方向是合理的：

- `shared/weights/stable-diffusion-v1-5/`
- `shared/weights/clip-vit-large-patch14/`
- `shared/weights/blip-image-captioning-large/`
- `shared/weights/google-ddpm-cifar10-32/`
- `shared/datasets/celeba/`
- `black-box/supplementary/recon-assets/`
- `gray-box/weights/secmi-cifar-bundle/`

这些名字的问题不大，因为它们表达的是“资产本体是什么”。

---

## 5. Current Non-Canonical Or Mixed Names

下面这些名字不是“错”，但它们更像来源/历史痕迹，不应该被误读成最优 canonical style：

- `black-box/supplementary/clid-mia-supplementary/`
  - 混合了方法名和 supplementary 类型
  - 还能接受，但偏长
- `gray-box/supplementary/secmi-onedrive/`
  - 这是来源痕迹名，不是消费语义名
  - 它表达的是“从 OneDrive 来”，不是“资产是什么”
- `shared/supplementary/celeba-7z-parts/`
  - 这是压缩形态名，不是数据语义名

当前更合理的理解是：

- `secmi-cifar-bundle/` 才是 SecMI 的 canonical consumer asset
- `secmi-onedrive/` 只是 raw archive provenance stash
- `celeba/` 才是共享数据集入口
- `celeba-7z-parts/` 只是补充来源文件

---

## 6. Practical Guidance

以后新增资产，按这个判断：

1. 如果这是项目真正要消费的资产根：
   - 用 canonical name
2. 如果这只是下载来源压缩包：
   - 放到对应资产目录下的 `raw/`
3. 如果这是上游解压后的原样内容：
   - 放到 `contents/`
4. 如果这是网页/授权/镜像说明：
   - 放到 `notes/`

不要再把以下名字直接当 canonical 顶层资产名：

- `onedrive`
- `google-drive`
- `zip-parts`
- `manual-download`
- `hf-cache`

这些词只能描述来源，不应该描述资产本体。

---

## 7. Current Project Judgment

当前 `Download/` 结构已经比以前统一很多，足够用。

但如果要给新成员一个更稳定的长期印象，最好的口径是：

- `Download/` 已经是正确的 raw intake 层
- 现有少数 source-trace 目录允许暂时保留
- 新资产一律按 canonical naming policy 继续收口

换句话说：

- 结构已经基本对
- 命名还不是完美
- 但现在最合理的动作是“明确规则并停止继续变乱”，而不是立刻大搬家
