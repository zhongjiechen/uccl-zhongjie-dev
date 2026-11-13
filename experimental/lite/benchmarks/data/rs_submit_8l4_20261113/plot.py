"""Plot the fixed-configuration confirmation; no GPU work or variant selection."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--preview', type=Path, help='optional PNG output path')
args = parser.parse_args()
rows = list(csv.DictReader((DATA / 'final_summary.csv').open()))
variants = [
    ('previous', 'Previous cap4 + batch', '#6b7280'),
    ('hybrid', 'Small-message GPU progress', '#2563eb'),
    ('hybrid_parallel', 'GPU progress + 8 launch lanes', '#0d9488'),
]
plt.rcParams.update({'font.size': 10, 'svg.hashsalt': 'rs-submit-8l4-20261113'})
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
for name, label, color in variants:
    selected = [r for r in rows if r['variant'] == name and int(r['bytes']) <= 1048576]
    x = [int(r['bytes']) / 1024 for r in selected]
    y = [float(r['median_us']) for r in selected]
    error = [[v - float(r['sweep_min_us']) for r, v in zip(selected, y)],
             [float(r['sweep_max_us']) - v for r, v in zip(selected, y)]]
    axes[0].errorbar(x, y, yerr=error, marker='o', markersize=3.5,
                     linewidth=1.6, capsize=2, label=label, color=color)
    if name != 'previous':
        axes[1].plot(x, [float(r['reduction_vs_previous_pct']) for r in selected],
                     marker='o', markersize=3.5, linewidth=1.6,
                     label=label, color=color)
selected = [r for r in rows if r['variant'] == 'hybrid_parallel' and int(r['bytes']) <= 1048576]
axes[0].plot([int(r['bytes']) / 1024 for r in selected],
             [float(r['paired_nccl_median_us']) for r in selected],
             '--', linewidth=1.6, color='#d97706', label='NCCL paired with combined')
for ax in axes:
    ax.set_xscale('log', base=2)
    ax.set_xticks([1, 4, 16, 64, 256, 1024], ['1K', '4K', '16K', '64K', '256K', '1M'])
    ax.set_xlabel('Complete BF16 input bytes per rank')
    ax.grid(True, alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('Slowest-rank eager latency (us)')
axes[0].set_ylim(bottom=65)
axes[0].legend(frameon=False, fontsize=8, loc='upper left')
axes[1].axhline(0, color='#6b7280', linewidth=.8)
axes[1].set_ylabel('Latency reduction vs previous (%)')
axes[1].set_ylim(-3, 29)
axes[1].legend(frameon=False, fontsize=8, loc='upper right')
fig.suptitle('8 x NVIDIA L4: BF16 ReduceScatter, dense eager calls', fontsize=13)
fig.text(.5, .012,
         'Fixed configurations; no best-per-size selection. Two reversed-order sweeps; bars show sweep medians range.\n'
         'Original event scope includes host-submission gaps. Parallel lanes spin up to 2 ms and can regress with sparse calls.',
         ha='center', fontsize=8, color='#4b5563')
fig.tight_layout(rect=(0, .08, 1, .95))
fig.savefig(DATA / 'summary.svg', metadata={'Date': None})
if args.preview:
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.preview, dpi=160)
