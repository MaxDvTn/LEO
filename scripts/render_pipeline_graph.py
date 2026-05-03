"""Render the LEO dataset generation pipeline as a styled PNG."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── colour palette ────────────────────────────────────────────────────────────
C = {
    "bg":        "#0f1117",
    "grid":      "#1a1d27",
    "source":    "#1e3a5f",   # input sources
    "model":     "#1a3d2b",   # model / backend
    "process":   "#2d2040",   # processing steps
    "storage":   "#3d2010",   # file storage
    "training":  "#3d1020",   # training / eval
    "decision":  "#2d3a10",   # decision / routing
    "accent1":   "#4a9eff",   # source border
    "accent2":   "#4aff8a",   # model border
    "accent3":   "#a855f7",   # process border
    "accent4":   "#ff8c42",   # storage border
    "accent5":   "#ff4466",   # training border
    "accent6":   "#c8ff44",   # decision border
    "text":      "#e8eaf0",
    "subtext":   "#9095a8",
    "arrow":     "#5a6080",
    "arrow_hl":  "#4a9eff",
}

fig, ax = plt.subplots(figsize=(22, 32), facecolor=C["bg"])
ax.set_facecolor(C["bg"])
ax.set_xlim(0, 22)
ax.set_ylim(0, 32)
ax.axis("off")

# ── helpers ───────────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, label, sublabel="", fill=C["process"], border=C["accent3"],
        fontsize=9, radius=0.25):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f"round,pad=0.05,rounding_size={radius}",
                          linewidth=1.5, edgecolor=border, facecolor=fill, zorder=3)
    ax.add_patch(rect)
    if sublabel:
        ax.text(x, y + 0.13, label, ha="center", va="center", fontsize=fontsize,
                color=C["text"], fontweight="bold", zorder=4)
        ax.text(x, y - 0.18, sublabel, ha="center", va="center", fontsize=fontsize - 1.5,
                color=C["subtext"], zorder=4, style="italic")
    else:
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
                color=C["text"], fontweight="bold", zorder=4)

def diamond(ax, x, y, w, h, label, fill=C["decision"], border=C["accent6"], fontsize=8.5):
    dx, dy = w/2, h/2
    xs = [x,      x+dx,  x,      x-dx,  x]
    ys = [y+dy,   y,     y-dy,   y,     y+dy]
    ax.fill(xs, ys, color=fill, zorder=3)
    ax.plot(xs, ys, color=border, linewidth=1.5, zorder=4)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
            color=C["text"], fontweight="bold", zorder=5)

def arrow(ax, x1, y1, x2, y2, label="", color=C["arrow"], lw=1.2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=12),
                zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.08, my, label, fontsize=7.5, color=C["subtext"],
                va="center", zorder=5)

def section_label(ax, x, y, text):
    ax.text(x, y, text, fontsize=7, color=C["subtext"], ha="left",
            fontfamily="monospace", zorder=6)

# ── title ─────────────────────────────────────────────────────────────────────
ax.text(11, 31.3, "LEO — Dataset Generation Pipeline",
        ha="center", va="center", fontsize=17, color=C["text"],
        fontweight="bold")
ax.text(11, 30.85, "Synthetic data · Multi-model · Multi-directional · Checkpointed",
        ha="center", va="center", fontsize=9.5, color=C["subtext"], style="italic")

# ── ROW 1: Input sources ──────────────────────────────────────────────────────
y_src = 29.7
box(ax, 5.5,  y_src, 3.2, 0.7, "Curated Glossary", "src/synthesis/glossary_data.py",
    fill=C["source"], border=C["accent1"])
box(ax, 11,   y_src, 3.2, 0.7, "Competitor Websites", "SpiderConfig.target_urls",
    fill=C["source"], border=C["accent1"])
box(ax, 16.5, y_src, 3.2, 0.7, "PDF Corpus", "data/raw/pdfs/  (99 files)",
    fill=C["source"], border=C["accent1"])

# source pre-processing
y_pre = 28.5
box(ax, 5.5,  y_pre, 3.2, 0.6, "get_terms_list()",
    fill=C["process"], border=C["accent3"])
box(ax, 11,   y_pre, 3.2, 0.6, "Filter noise", "cookie / privacy / generic",
    fill=C["process"], border=C["accent3"])
box(ax, 16.5, y_pre, 3.2, 0.6, "Extract · Dedup · LangDetect",
    fill=C["process"], border=C["accent3"])

arrow(ax, 5.5,  y_src-0.35, 5.5,  y_pre+0.3)
arrow(ax, 11,   y_src-0.35, 11,   y_pre+0.3)
arrow(ax, 16.5, y_src-0.35, 16.5, y_pre+0.3)

# ── ROW 2: Language split (PDF only) ─────────────────────────────────────────
y_split = 27.3
box(ax, 15.0, y_split, 2.2, 0.6, "IT / unknown", "~17 k sentences",
    fill=C["source"], border=C["accent1"], fontsize=8.5)
box(ax, 17.4, y_split, 1.9, 0.6, "EN native", "~6.2 k",
    fill=C["model"], border=C["accent2"], fontsize=8.5)
box(ax, 19.1, y_split, 1.9, 0.6, "FR / ES", "~9.7 k",
    fill=C["model"], border=C["accent2"], fontsize=8.5)

arrow(ax, 16.5, y_pre-0.3, 15.0, y_split+0.3, color=C["accent1"])
arrow(ax, 16.5, y_pre-0.3, 17.4, y_split+0.3, color=C["accent2"])
arrow(ax, 16.5, y_pre-0.3, 19.1, y_split+0.3, color=C["accent2"])

# converge glossary/web to model selection row
y_conv = 27.3
box(ax, 5.5,  y_conv, 3.0, 0.6, "46 terms × variants", fill=C["process"], border=C["accent3"], fontsize=8.5)
box(ax, 11,   y_conv, 3.0, 0.6, "Domain terms", fill=C["process"], border=C["accent3"], fontsize=8.5)
arrow(ax, 5.5,  y_pre-0.3, 5.5,  y_conv+0.3)
arrow(ax, 11,   y_pre-0.3, 11,   y_conv+0.3)

# ── ROW 3: Model selection ────────────────────────────────────────────────────
y_ms = 26.1
diamond(ax, 11, y_ms, 3.5, 0.75, "Select model\n& backend")
arrow(ax, 5.5,  y_conv-0.3, 9.25, y_ms, color=C["accent6"])
arrow(ax, 11,   y_conv-0.3, 11,   y_ms+0.375, color=C["accent6"])
arrow(ax, 15.0, y_split-0.3, 12.75, y_ms, color=C["accent6"])

# ── ROW 4: Backends ───────────────────────────────────────────────────────────
y_be = 24.7
backends = [
    (4.5,  "OllamaGenerator", "mistral-small3.2 · qwen2.5:32b\ngemma3:27b · aya-expanse:8b", C["accent2"]),
    (9.5,  "CloudGenerator", "Gemini 2.5 Flash\ngoogle/ prefix → litellm", C["accent1"]),
    (14.5, "CloudGenerator", "OpenAI / Anthropic\nDeepSeek via litellm", C["accent1"]),
]
for bx, bl, bs, bc in backends:
    box(ax, bx, y_be, 4.0, 0.85, bl, bs, fill=C["model"], border=bc, fontsize=8.5)
    arrow(ax, 11, y_ms - 0.375, bx, y_be + 0.425, color=C["accent6"], lw=1.0)

# ── ROW 5: BaseGenerator ─────────────────────────────────────────────────────
y_base = 23.3
box(ax, 9.5, y_base, 9.5, 0.65,
    "BaseGenerator  —  shared logic",
    "ThreadPoolExecutor · _chat_with_retry · parse_output · num_workers",
    fill=C["process"], border=C["accent3"], fontsize=8.5)
for bx, *_ in backends:
    arrow(ax, bx, y_be - 0.425, 9.5, y_base + 0.325, color=C["arrow"], lw=1.0)

# ── ROW 6: Prompt types ───────────────────────────────────────────────────────
y_pt = 22.05
box(ax, 5.5,  y_pt, 3.8, 0.72, "generate_dataset()", "IT sentence + EN/FR/ES\nJSON prompt  ·  term + context + doc_type",
    fill=C["process"], border=C["accent3"], fontsize=8)
box(ax, 11,   y_pt, 3.8, 0.72, "translate_text()", "IT → EN / FR / ES\ntranslation JSON prompt",
    fill=C["process"], border=C["accent3"], fontsize=8)
box(ax, 16.5, y_pt, 3.8, 0.72, "translate_to_italian()", "EN / FR / ES → IT\nnew prompt per source lang",
    fill=C["process"], border=C["accent3"], fontsize=8)
arrow(ax, 9.5, y_base-0.325, 5.5,  y_pt+0.36, color=C["accent3"], lw=1.0)
arrow(ax, 9.5, y_base-0.325, 11,   y_pt+0.36, color=C["accent3"], lw=1.0)
arrow(ax, 9.5, y_base-0.325, 16.5, y_pt+0.36, color=C["accent3"], lw=1.0)

# route native langs to translate_to_italian
arrow(ax, 17.4, y_split-0.3, 16.5, y_pt+0.36, color=C["accent2"], lw=1.0)
arrow(ax, 19.1, y_split-0.3, 16.5, y_pt+0.36, color=C["accent2"], lw=1.0)

# ── ROW 7: Output rows ────────────────────────────────────────────────────────
y_out = 20.7
box(ax, 5.5,  y_out, 3.8, 0.72, "Glossary / Web rows",
    "origin · model_id · term · context\n+ reverse pairs (add_reverse=True)",
    fill=C["process"], border=C["accent3"], fontsize=8)
box(ax, 11,   y_out, 3.8, 0.72, "IT→X  +  free X→IT",
    "6 rows per sentence\nprompt_version: translate_json_v1 / _rev",
    fill=C["process"], border=C["accent3"], fontsize=8)
box(ax, 16.5, y_out, 3.8, 0.72, "Native X→IT rows",
    "1 row per sentence\nprompt_version: translate_to_it_v1",
    fill=C["process"], border=C["accent3"], fontsize=8)
arrow(ax, 5.5,  y_pt-0.36, 5.5,  y_out+0.36)
arrow(ax, 11,   y_pt-0.36, 11,   y_out+0.36)
arrow(ax, 16.5, y_pt-0.36, 16.5, y_out+0.36)

# ── ROW 8: Checkpoint ─────────────────────────────────────────────────────────
y_ckpt = 19.5
box(ax, 11, y_ckpt, 10.5, 0.65,
    "Incremental Checkpoint  (every 300 rows)",
    "data/synthetic/checkpoints/rover_pdf_augmented__<model>.csv  —  resume-safe",
    fill=C["storage"], border=C["accent4"], fontsize=8.5)
arrow(ax, 5.5,  y_out-0.36, 11, y_ckpt+0.325, color=C["accent4"], lw=1.0)
arrow(ax, 11,   y_out-0.36, 11, y_ckpt+0.325, color=C["accent4"], lw=1.0)
arrow(ax, 16.5, y_out-0.36, 11, y_ckpt+0.325, color=C["accent4"], lw=1.0)

# ── ROW 9: Normalize & save ───────────────────────────────────────────────────
y_norm = 18.35
box(ax, 11, y_norm, 10.5, 0.65,
    "_normalize_synthetic_df()",
    "drop bad translations · deduplicate · length ratio filter",
    fill=C["process"], border=C["accent3"], fontsize=8.5)
arrow(ax, 11, y_ckpt-0.325, 11, y_norm+0.325)

y_save = 17.2
box(ax, 5.5,  y_save, 4.0, 0.65, "Canonical active file", "data/synthetic/*.csv",
    fill=C["storage"], border=C["accent4"], fontsize=8.5)
box(ax, 11,   y_save, 4.0, 0.65, "Archive", "data/synthetic/archive/",
    fill=C["storage"], border=C["accent4"], fontsize=8.5)
box(ax, 16.5, y_save, 4.0, 0.65, "Versioned run file", "data/synthetic/runs/",
    fill=C["storage"], border=C["accent4"], fontsize=8.5)
arrow(ax, 11, y_norm-0.325, 5.5,  y_save+0.325, color=C["accent4"])
arrow(ax, 11, y_norm-0.325, 11,   y_save+0.325, color=C["accent4"])
arrow(ax, 11, y_norm-0.325, 16.5, y_save+0.325, color=C["accent4"])

# more models loop
diamond(ax, 11, 16.1, 3.0, 0.65, "More models?")
arrow(ax, 11, y_save-0.325, 11, 16.1+0.325, color=C["accent6"])
# yes loop back
ax.annotate("", xy=(4.5, 26.1), xytext=(4.5, 16.1),
            arrowprops=dict(arrowstyle="-|>", color=C["accent6"], lw=1.2,
                            connectionstyle="arc3,rad=0.0"), zorder=2)
ax.plot([9.5, 4.5], [16.1, 16.1], color=C["accent6"], lw=1.2, zorder=2)
ax.text(3.2, 21, "yes → next model", fontsize=7.5, color=C["accent6"],
        va="center", rotation=90)

# ── ROW 10: Build Ensemble ────────────────────────────────────────────────────
y_ens = 14.85
box(ax, 11, y_ens, 14, 0.65,
    "build_ensemble()  —  merge all versioned run CSVs",
    "glossary_synthetic__*.csv  ·  rover_pdf_augmented__*.csv  ·  competitor_synthetic__*.csv",
    fill=C["process"], border=C["accent3"], fontsize=8.5)
arrow(ax, 12.5, 16.1-0.325, 11, y_ens+0.325, color=C["accent3"], label="no")

y_e1, y_e2, y_e3, y_e4 = 14.05, 13.25, 12.45, 11.65
steps = [
    (y_e1, "1. Model priority sort", "mistral-small3.2 > gemma3:27b > qwen2.5:32b > Gemini > aya > mistral-nemo > phi4"),
    (y_e2, "2. Exact deduplication", "drop_duplicates(source_text + target_text + target_lang)"),
    (y_e3, "3. Near-duplicate filter", "_dedup_near_duplicates()  ·  sentence-transformers cosine sim > 0.90"),
    (y_e4, "4. Quality filter", "_normalize_synthetic_df()  ·  bad-translation heuristics"),
]
for yy, lbl, sub in steps:
    box(ax, 11, yy, 13.5, 0.55, lbl, sub, fill=C["process"], border=C["accent3"], fontsize=8)
    arrow(ax, 11, yy+0.275+0.55, 11, yy+0.275, lw=1.0)

y_eset = 10.9
box(ax, 11, y_eset, 7, 0.65,
    "ensemble_training_set.csv",
    "data/synthetic/ensemble_training_set.csv",
    fill=C["storage"], border=C["accent4"], fontsize=9)
arrow(ax, 11, y_e4-0.275, 11, y_eset+0.325, color=C["accent4"])

# ── ROW 11: Training dataloader ───────────────────────────────────────────────
y_dl = 9.75
box(ax, 7.5,  y_dl, 3.5, 0.65, "ensemble_training_set.csv", fill=C["storage"], border=C["accent4"], fontsize=8)
box(ax, 11,   y_dl, 3.5, 0.65, "rover_synthetic_multilingual.csv", fill=C["storage"], border=C["accent4"], fontsize=7.5)
box(ax, 14.5, y_dl, 3.5, 0.65, "competitor_synthetic.csv", fill=C["storage"], border=C["accent4"], fontsize=8)
arrow(ax, 11, y_eset-0.325, 7.5,  y_dl+0.325, color=C["accent4"])
arrow(ax, 11, y_eset-0.325, 11,   y_dl+0.325, color=C["accent4"])
arrow(ax, 11, y_eset-0.325, 14.5, y_dl+0.325, color=C["accent4"])

y_loader = 8.75
box(ax, 11, y_loader, 10, 0.65,
    "SeamlessDataModule  —  Training Dataloader",
    "DataLeakage-safe split  ·  SEAMLESS_LANG_MAP normalisation  ·  test_set excluded",
    fill=C["process"], border=C["accent3"], fontsize=8.5)
for bx in [7.5, 11, 14.5]:
    arrow(ax, bx, y_dl-0.325, 11, y_loader+0.325, color=C["accent3"])

# ── ROW 12: Training ──────────────────────────────────────────────────────────
y_train = 7.6
box(ax, 8.5, y_train, 5.5, 0.72,
    "SeamlessM4T v2 Large  —  Fine-tuning",
    "QLoRA  ·  LoRA r=32 α=64  ·  out_proj target\nbf16-mixed  ·  Lightning + WandB",
    fill=C["training"], border=C["accent5"], fontsize=8.5)
box(ax, 14.5, y_train, 4.5, 0.72,
    "Metrics (val)",
    "BLEU · chrF · METEOR\nper lang-pair",
    fill=C["training"], border=C["accent5"], fontsize=8.5)
arrow(ax, 11, y_loader-0.325, 8.5, y_train+0.36, color=C["accent5"])
arrow(ax, 8.5+2.75, y_train, 14.5-2.25, y_train, color=C["accent5"])

# ── ROW 13: Benchmark & select ────────────────────────────────────────────────
y_bm = 6.35
box(ax, 8.5, y_bm, 5.5, 0.65,
    "Benchmark on gold test set",
    "BLEU · chrF · METEOR  vs  base model\nWandB heatmap · regression table",
    fill=C["training"], border=C["accent5"], fontsize=8.5)
arrow(ax, 8.5, y_train-0.36, 8.5, y_bm+0.325, color=C["accent5"])

y_sel = 5.2
box(ax, 8.5, y_sel, 5.5, 0.65,
    "Select best checkpoint",
    "Export LoRA adapter  ·  leo_hf_release/",
    fill=C["training"], border=C["accent5"], fontsize=8.5)
arrow(ax, 8.5, y_bm-0.325, 8.5, y_sel+0.325, color=C["accent5"])

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    (C["source"],   C["accent1"], "Input source"),
    (C["model"],    C["accent2"], "LLM backend"),
    (C["process"],  C["accent3"], "Processing step"),
    (C["storage"],  C["accent4"], "File / storage"),
    (C["training"], C["accent5"], "Training / eval"),
    (C["decision"], C["accent6"], "Decision / routing"),
]
lx, ly = 0.5, 4.2
ax.text(lx, ly + 0.3, "Legend", fontsize=8, color=C["subtext"], fontweight="bold")
for i, (fill, border, label) in enumerate(legend_items):
    rect = FancyBboxPatch((lx, ly - i*0.55 - 0.2), 0.55, 0.35,
                          boxstyle="round,pad=0.03,rounding_size=0.07",
                          linewidth=1.2, edgecolor=border, facecolor=fill, zorder=3)
    ax.add_patch(rect)
    ax.text(lx + 0.7, ly - i*0.55 - 0.02, label, fontsize=7.5, color=C["text"],
            va="center", zorder=4)

# stats box
sx, sy = 14.5, 4.5
ax.text(sx, sy + 0.2, "Dataset stats (per model, PDF kind)", fontsize=8,
        color=C["subtext"], fontweight="bold")
stats = [
    "IT/unknown sentences :  ~17 000",
    "× 6 rows (3 fwd + 3 rev) :  ~102 000",
    "Native EN/FR/ES        :  ~15 900",
    "Total rows / model     :  ~118 000",
    "Models in ensemble     :  5",
    "Checkpoint interval    :  300 rows",
]
for i, s in enumerate(stats):
    ax.text(sx, sy - 0.4 - i*0.4, s, fontsize=8, color=C["text"],
            fontfamily="monospace", va="top")

# footer
ax.text(11, 0.3, "LEO — Lingua Engineering Optimizer  ·  Roverplastik NMT fine-tuning",
        ha="center", fontsize=8, color=C["subtext"], style="italic")

plt.tight_layout(pad=0)
out = "/home/mbosetti/LEO/reports/dataset_generation_pipeline.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C["bg"])
print(f"Saved: {out}")
