from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont


PALETTE = {
    "navy": "#16324F",
    "blue": "#2E6F9E",
    "cyan": "#51A7B8",
    "teal": "#3A8D7C",
    "orange": "#D9893D",
    "red": "#B5534E",
    "paper": "#F7F8FA",
    "ink": "#263238",
    "muted": "#68747D",
    "line": "#CBD3DA",
}


def configure_chinese_font() -> str:
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            prop = font_manager.FontProperties(fname=str(path))
            name = prop.get_name()
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return "DejaVu Sans"


def box(ax, xy, wh, title, lines, color, title_size=13, body_size=10):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.014,rounding_size=0.018",
        linewidth=1.3,
        edgecolor=color,
        facecolor="white",
        zorder=2,
    )
    ax.add_patch(patch)
    ax.add_patch(FancyBboxPatch(
        (x, y + h - 0.075), w, 0.075,
        boxstyle="round,pad=0.014,rounding_size=0.018",
        linewidth=0,
        facecolor=color,
        zorder=3,
    ))
    ax.text(x + 0.018, y + h - 0.037, title, color="white", fontsize=title_size,
            fontweight="bold", va="center", zorder=4)
    body_top = y + h - 0.108
    step = min(0.043, (h - 0.135) / max(1, len(lines) - 1))
    for i, line in enumerate(lines):
        ax.text(x + 0.022, body_top - i * step, "• " + line,
                color=PALETTE["ink"], fontsize=body_size, va="top", zorder=4)
    return patch


def arrow(ax, start, end, color=None, connectionstyle="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=14,
        linewidth=1.4, color=color or PALETTE["muted"],
        connectionstyle=connectionstyle, zorder=1,
    ))


def make_workflow(out_dir: Path):
    fig, ax = plt.subplots(figsize=(15.6, 8.8), dpi=180)
    fig.patch.set_facecolor(PALETTE["paper"])
    ax.set_facecolor(PALETTE["paper"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.03, 0.955, "GestureGraph Lab：从课程验收到开放研究的总体路线", fontsize=21,
            fontweight="bold", color=PALETTE["navy"], va="top")
    ax.text(0.03, 0.91, "同一数据协议 · 逐层消融 · 结果边界可追溯", fontsize=11,
            color=PALETTE["muted"], va="top")

    box(ax, (0.03, 0.60), (0.19, 0.23), "问题与输入",
        ["22 关节 × XYZ × 64 帧", "14 类正式划分", "空间拓扑 + 时间动态", "在线片段不完整"], PALETTE["navy"])
    box(ax, (0.28, 0.60), (0.20, 0.23), "课程基本要求",
        ["ST-GCN 与 MLP/LSTM 对照", "去图 / 单帧消融", "关节与手指遮挡", "正确 / 错误运动骨架"], PALETTE["blue"])
    box(ax, (0.54, 0.60), (0.19, 0.23), "结构开放路线",
        ["谱 SE 设计与注入边界", "物理 + 动态双 support", "融合方式比较", "AGCRN 节点专属 W"], PALETTE["teal"])
    box(ax, (0.78, 0.60), (0.19, 0.23), "最终 backbone",
        ["TCN 保留时间建模", "Aphysical + Adynamic", "节点自适应权重分解", "AGCRN 分支不重复加 SE"], PALETTE["orange"])

    arrow(ax, (0.22, 0.715), (0.28, 0.715))
    arrow(ax, (0.48, 0.715), (0.54, 0.715))
    arrow(ax, (0.73, 0.715), (0.78, 0.715))

    box(ax, (0.03, 0.22), (0.28, 0.25), "数据与时间讨论",
        ["Velocity 显式速度", "TemporalGate 聚焦有效帧", "Focal Loss 处理难混淆对", "随机裁剪 + 重采样"], PALETTE["cyan"])
    box(ax, (0.36, 0.22), (0.28, 0.25), "渐进分类讨论",
        ["前缀特征与观察比例", "类别后验一阶马尔可夫更新", "均匀先验软重置", "四步教师 → 少步部署"], PALETTE["teal"])
    box(ax, (0.69, 0.22), (0.28, 0.25), "质量与可靠性讨论",
        ["checkpoint 回归与微扰", "OOD / Unknown 拒识", "解释一致性与 ECE", "跨受试者偏差"], PALETTE["red"])

    arrow(ax, (0.385, 0.60), (0.17, 0.47), connectionstyle="arc3,rad=0.12")
    arrow(ax, (0.64, 0.60), (0.50, 0.47), connectionstyle="arc3,rad=0.05")
    arrow(ax, (0.875, 0.60), (0.83, 0.47), connectionstyle="arc3,rad=-0.08")
    arrow(ax, (0.31, 0.345), (0.36, 0.345))
    arrow(ax, (0.64, 0.345), (0.69, 0.345))

    ax.text(0.03, 0.11, "交付：基础要求闭环 ｜ 最佳结构 81.98%±1.69% ｜ 前缀 AUC 74.05%±0.17% ｜ 未闭环项显式标注",
            fontsize=13, fontweight="bold", color=PALETTE["navy"], va="center")
    ax.plot([0.03, 0.97], [0.15, 0.15], color=PALETTE["line"], linewidth=1)

    fig.savefig(out_dir / "overall_workflow.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(out_dir / "overall_workflow.svg", bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(out_dir / "overall_workflow.pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def make_results_overview(out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.2), dpi=180)
    fig.patch.set_facecolor("white")

    labels = ["完整\nST-GCN", "MLP", "去图", "单帧"]
    vals = [70.95, 63.81, 68.93, 32.02]
    colors = [PALETTE["navy"], PALETTE["blue"], PALETTE["cyan"], PALETTE["red"]]
    bars = axes[0].bar(labels, vals, color=colors, width=0.64)
    axes[0].set_title("A 课程基础消融", loc="left", fontweight="bold", color=PALETTE["navy"])
    axes[0].set_ylim(0, 90)
    axes[0].set_ylabel("准确率（%）")
    for b, v in zip(bars, vals):
        axes[0].text(b.get_x()+b.get_width()/2, v+1.4, f"{v:.2f}", ha="center", fontsize=9)

    labels2 = ["ST-GCN\n+SE", "QKV\n+SE", "GWN\n+SE", "AGCRN\n无SE"]
    vals2 = [73.77, 80.99, 74.72, 81.98]
    errs2 = [1.33, 0.68, 0.59, 1.69]
    colors2 = [PALETTE["blue"], PALETTE["cyan"], PALETTE["teal"], PALETTE["orange"]]
    bars2 = axes[1].bar(labels2, vals2, yerr=errs2, capsize=3, color=colors2, width=0.64)
    axes[1].set_title("B 候选 backbone（3 seeds）", loc="left", fontweight="bold", color=PALETTE["navy"])
    axes[1].set_ylim(65, 86)
    for b, v in zip(bars2, vals2):
        axes[1].text(b.get_x()+b.get_width()/2, v+2.0, f"{v:.2f}", ha="center", fontsize=9)

    ratios = [25, 50, 65, 80, 100]
    baseline = [34.05, 61.47, 71.71, 78.37, 80.16]
    diffusion = [57.90, 73.77, 78.17, 79.33, 79.37]
    axes[2].plot(ratios, baseline, marker="o", linewidth=2.3, color=PALETTE["muted"], label="普通前缀分类")
    axes[2].plot(ratios, diffusion, marker="o", linewidth=2.6, color=PALETTE["orange"], label="四步类别扩散")
    axes[2].fill_between(ratios, baseline, diffusion, where=[d >= b for b, d in zip(baseline, diffusion)],
                         color=PALETTE["orange"], alpha=0.12)
    axes[2].set_title("C 不同观察比例的准确率", loc="left", fontweight="bold", color=PALETTE["navy"])
    axes[2].set_xlabel("观察比例（%）")
    axes[2].set_ylabel("准确率（%）")
    axes[2].set_xticks(ratios)
    axes[2].set_ylim(25, 86)
    axes[2].legend(frameon=False, fontsize=9, loc="lower right")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=PALETTE["line"], linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(colors=PALETTE["ink"])
    fig.tight_layout(w_pad=2.2)
    fig.savefig(out_dir / "results_overview.png", bbox_inches="tight")
    fig.savefig(out_dir / "results_overview.svg", bbox_inches="tight")
    fig.savefig(out_dir / "results_overview.pdf", bbox_inches="tight")
    plt.close(fig)


def image_font(size: int, bold: bool = False):
    choices = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
    ]
    for p in choices:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def make_skeleton_examples(root: Path, out_dir: Path):
    items = [
        (root / "runs" / "shrec17_benchmark" / "correct_classification.gif", "正确分类示例", PALETTE["teal"]),
        (root / "runs" / "shrec17_benchmark" / "incorrect_classification.gif", "错误分类示例", PALETTE["red"]),
    ]
    cards = []
    for path, title, color in items:
        with Image.open(path) as im:
            im.seek(0)
            frame = im.convert("RGB")
        frame.thumbnail((720, 500), Image.Resampling.LANCZOS)
        card = Image.new("RGB", (760, 590), "white")
        draw = ImageDraw.Draw(card)
        draw.rounded_rectangle((4, 4, 756, 586), radius=16, outline=color, width=5)
        draw.rectangle((4, 4, 756, 70), fill=color)
        draw.text((28, 20), title, font=image_font(30, True), fill="white")
        x = (760 - frame.width) // 2
        y = 82 + (490 - frame.height) // 2
        card.paste(frame, (x, y))
        draw.text((28, 548), "完整动画见 runs/shrec17_benchmark/*.gif", font=image_font(18), fill=PALETTE["muted"])
        cards.append(card)
    canvas = Image.new("RGB", (1560, 620), PALETTE["paper"])
    canvas.paste(cards[0], (10, 15))
    canvas.paste(cards[1], (790, 15))
    canvas.save(out_dir / "skeleton_examples.png", dpi=(180, 180))


def copy_heatmap(root: Path, out_dir: Path):
    source = root / "runs" / "joint_aggregation_recheck_aggregate" / "agcrn_joint_seed43_adjacency.png"
    shutil.copy2(source, out_dir / "agcrn_learned_adjacency.png")


def make_equations(out_dir: Path):
    equations = [
        r"Y_l=\mathrm{ReLU}\!\left(\mathrm{TCN}\!\left(\mathrm{Spatial}_l(X_l)\right)+\mathrm{Residual}(X_l)\right)",
        r"L\,SE_k=\lambda_k SE_k",
        r"S=SE\odot\lambda",
        r"P_l=SW_{\mathrm{linear},l}+\eta_l\mathrm{MLP}_l(S),\qquad Z_l=X_l+\beta_lP_l",
        r"A_{\mathrm{attn}}^{(h)}=\mathrm{softmax}\!\left(\frac{Q^{(h)}K^{(h)\top}}{\sqrt{d_h}}+b_hA_{\mathrm{physical}}\right)",
        r"A_{\mathrm{dynamic}}=\mathrm{softmax}\!\left(\mathrm{ReLU}(E_1E_2^{\top})\right)",
        r"H=[X,\,A_pX,\,A_p^2X,\,A_dX,\,A_d^2X]",
        r"W_v=\sum_{k=1}^{d_e}E_1[v,k]W_{\mathrm{pool},k}",
        r"q_r^{(0)}=\gamma p_{r-1}+(1-\gamma)u,\qquad \gamma=0.5",
        r"p_r=D_\theta\!\left(q_r^{(0)},c_r\right)",
        r"\mathcal{L}=\mathcal{L}_{\mathrm{cls}}+0.5\mathcal{L}_{\mathrm{denoise}}+0.1\mathcal{L}_{\mathrm{KD}}",
    ]
    for idx, eq in enumerate(equations, start=1):
        fig = plt.figure(figsize=(8.0, 0.72), dpi=220)
        fig.patch.set_alpha(0)
        fig.text(0.5, 0.5, f"${eq}$", ha="center", va="center", fontsize=17, color=PALETTE["ink"])
        fig.savefig(out_dir / f"equation_{idx:02d}.png", transparent=True, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_chinese_font()
    make_workflow(out_dir)
    make_results_overview(out_dir)
    make_skeleton_examples(root, out_dir)
    copy_heatmap(root, out_dir)
    make_equations(out_dir)
    print(f"generated assets in {out_dir}")


if __name__ == "__main__":
    main()
