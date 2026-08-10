# `.scratch/` 纳入 git 版本控制（反转默认忽略）

本仓库 `issue-tracker.md` 约定 spec / issues / 章节大纲存放在 `.scratch/<slug>/`，而初始 `.gitignore` 忽略了 `.scratch/`——即 tracker 默认本地私有、不进历史。

为支持**远程 / AFK 开发**（作者需在远端 `git clone` 后凭仓库自足开工），我们**取消对 `.scratch/` 的忽略**，让 spec、章节 Markdown 大纲、tickets 随仓库同步。

**为什么记录**：这直接反转了 `.gitignore` 与"过程档案不进 git"的默认直觉，后续读者看到 `.scratch/` 在版本控制里可能困惑或"顺手改回忽略"。权衡：本地临时私有（原默认）vs 远程可协作可见（现选择）——因远程开发需求选后者。副作用：`.scratch/` 下所有内容都会入库，故其中只应放人可读的过程档案，不放机器产物/大文件。
