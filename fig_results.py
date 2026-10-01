"""Result figures in grayscale.  Legends sit outside the plotting area so that no legend,
label or annotation overlaps data."""
import json, glob, collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'Liberation Serif', 'mathtext.fontset': 'stix', 'font.size': 7.5,
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
RES = json.load(open('results.json'))
NAMES = dict(cose='CoSE', median='Median-LoRA', fedavg='FedAvg-LoRA', fedprox='FedProx-LoRA', ditto='Ditto',
             gossip='Gossip-LoRA', ffa='FFA-LoRA', ties='TIES exchange', local='Local')
STY = dict(cose=('0.0', '-', 'o', 1.6), median=('0.35', '--', 's', 1.1), fedavg=('0.5', '-.', '^', 1.1),
           fedprox=('0.62', ':', 'v', 1.2), ditto=('0.3', (0, (1, 1)), 'D', 1.0), gossip=('0.7', '-', 'x', 1.0),
           ffa=('0.55', (0, (5, 1, 1, 1)), 'P', 1.0), ties=('0.8', '--', '*', 1.0), local=('0.75', '-', '.', 1.0))
ORDER = ['cose', 'median', 'fedavg', 'fedprox', 'ditto', 'gossip', 'ffa', 'ties', 'local']


def runs(base):
    out = {}
    for f in glob.glob(f'runs/{base}_s*.json'):
        out[int(f.rsplit('_s', 1)[1][:-5])] = json.load(open(f))['res']
    return out


def fig_curves(path):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.75), sharey=True)
    for ax, pre, ttl in [(axes[0], 'E1', '(a) three unreliable nodes'), (axes[1], 'E1b', '(b) all nodes honest')]:
        for m in ORDER:
            r = RES.get(f'{pre}_{m}')
            if not r or 'curve' not in r:
                continue
            c, ls, mk, lw = STY[m]
            t = np.array(r['curve']['t']) + 1
            ax.fill_between(t, r['curve']['lo'], r['curve']['hi'], color=c, alpha=0.12, lw=0)
            ax.plot(t, r['curve']['mean'], color=c, ls=ls, marker=mk, ms=3.2, lw=lw, label=NAMES[m], markevery=2)
        for x in (12.5, 24.5):
            ax.axvline(x, color='0.6', lw=0.5, ls=':')
        ax.set_xlabel('Round'); ax.set_title(ttl, fontsize=7.5, pad=4)
        ax.set_xlim(1, 37); ax.set_ylim(0, 0.8); ax.grid(axis='y', color='0.9', lw=0.5)
    axes[0].set_ylabel('Mean accuracy over six domains')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.0), fontsize=6.8, handlelength=2.6, columnspacing=1.6)
    fig.tight_layout(rect=(0.01, 0, 0.99, 0.84))
    fig.savefig(path, dpi=400); plt.close(fig)


def fig_domains(path):
    doms = ['ag', 'r8', 'trec', 'mr', 'cr', 'tweet']
    dl = ['AG', 'R8', 'TREC', 'MR', 'CR', 'Msg.']
    ms = [m for m in ORDER if f'E1_{m}' in RES]
    A = np.array([[RES[f'E1_{m}']['dom'][d] for d in doms] for m in ms])
    fig = plt.figure(figsize=(3.5, 2.75))
    ax = fig.add_axes([0.25, 0.1, 0.56, 0.84])
    im = ax.imshow(A, cmap='Greys', vmin=0, vmax=1, aspect='auto')
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j, i, f'{A[i, j]:.2f}', ha='center', va='center', fontsize=6.3, color='white' if A[i, j] > 0.55 else 'black')
    ax.set_xticks(range(6)); ax.set_xticklabels(dl); ax.set_yticks(range(len(ms))); ax.set_yticklabels([NAMES[m] for m in ms])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cax = fig.add_axes([0.85, 0.1, 0.03, 0.84]); cb = fig.colorbar(im, cax=cax); cb.set_label('Final accuracy', fontsize=7); cb.outline.set_linewidth(0.4)
    fig.savefig(path, dpi=400); plt.close(fig)


def fig_gate(path):
    """Trust score by sender type and acceptance shares."""
    T = collections.defaultdict(list); K = collections.defaultdict(list)
    for r in runs('E1_cose').values():
        for kind, dec, t, kap, mi, gain in r.get('T_kind', []):
            if t is not None:
                T[kind].append(t)
            if kap is not None:
                K[kind].append(kap)
    kinds = ['honest', 'flip', 'noise', 'scale']
    lab = ['Honest', 'Label flip', 'Noise', 'Scaled ($-5\\times$)']
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.3))
    ax = axes[0]
    data = [T[k] for k in kinds]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color='black', lw=1.1))
    for p, g in zip(bp['boxes'], ['0.95', '0.75', '0.6', '0.45']):
        p.set_facecolor(g); p.set_linewidth(0.6)
    for i, d in enumerate(data):
        x = np.random.default_rng(i).normal(i + 1, 0.06, len(d))
        ax.scatter(x, d, s=1.2, color='0.15', alpha=0.25, lw=0)
    ax.axhline(0.5, color='black', lw=0.7, ls='--'); ax.text(4.45, 0.52, '$\\tau_{\\mathrm{acc}}$', fontsize=7, va='bottom', ha='right')
    ax.set_xticks(range(1, 5)); ax.set_xticklabels([l + '\n' for l in lab]); ax.set_ylabel('Trust score $T_c$'); ax.set_ylim(-0.02, 1.02)
    ax.set_title('(a) trust score of delivered capsules', fontsize=7.5, pad=4)
    ax = axes[1]
    bk = RES['E1_cose'].get('by_kind', {})
    acc = [bk.get(k, [0, 1])[0] / max(1, sum(bk.get(k, [0, 1]))) for k in kinds]
    data = [K[k] for k in kinds]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color='black', lw=1.1))
    for p, g in zip(bp['boxes'], ['0.95', '0.75', '0.6', '0.45']):
        p.set_facecolor(g); p.set_linewidth(0.6)
    ax.axhline(0.7, color='black', lw=0.7, ls='--'); ax.text(4.45, 0.72, '$\\kappa_0$', fontsize=7, va='bottom', ha='right')
    ax.set_xticks(range(1, 5)); ax.set_xticklabels([f'{l}\nadmitted {100 * a:.0f}%' for l, a in zip(lab, acc)])
    ax.set_ylabel('Peer consensus $\\kappa_c$'); ax.set_ylim(-0.02, 1.02)
    ax.set_title('(b) consensus and admission share', fontsize=7.5, pad=4)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(path, dpi=400); plt.close(fig)


def fig_robust(path):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.35))
    ax = axes[0]
    xs = [0, 1, 2, 3, 4]
    for m in ['cose', 'median', 'fedavg', 'fedprox']:
        ys, lo, hi = [], [], []
        for a in ['a0', 'a1', 'a2', None, 'a4']:
            key = f'E4_{a}_{m}' if a else f'E1_{m}'
            r = RES.get(key)
            if a is None and r:
                vals = [v for s, v in zip(r['seeds'], r['acc']['vals']) if s in (10, 11, 12)]
                ys.append(np.mean(vals)); lo.append(np.min(vals)); hi.append(np.max(vals)); continue
            if r:
                ys.append(r['acc']['mean']); lo.append(min(r['acc']['vals'])); hi.append(max(r['acc']['vals']))
            else:
                ys.append(np.nan); lo.append(np.nan); hi.append(np.nan)
        c, ls, mk, lw = STY[m]
        ax.errorbar(xs, ys, yerr=[np.array(ys) - np.array(lo), np.array(hi) - np.array(ys)], color=c, ls=ls, marker=mk,
                    ms=3.5, lw=lw, capsize=2, elinewidth=0.6, label=NAMES[m])
    ax.set_xticks(xs); ax.set_xticklabels(['0', '1', '2', '3', '4']); ax.set_xlabel('Unreliable nodes out of 12')
    ax.set_ylabel('Final mean accuracy'); ax.set_ylim(0, 0.75); ax.grid(axis='y', color='0.9', lw=0.5)
    ax.set_title('(a) attacker share', fontsize=7.5, pad=4)
    ax = axes[1]
    for m in ['cose', 'median', 'fedavg']:
        ys, lo, hi = [], [], []
        for p in ['4', '6', None, '0']:
            key = f'E4p_p{p}_{m}' if p else f'E1_{m}'
            r = RES.get(key)
            if p is None and r:
                vals = [v for s, v in zip(r['seeds'], r['acc']['vals']) if s in (10, 11, 12)]
                ys.append(np.mean(vals)); lo.append(np.min(vals)); hi.append(np.max(vals)); continue
            if r:
                ys.append(r['acc']['mean']); lo.append(min(r['acc']['vals'])); hi.append(max(r['acc']['vals']))
            else:
                ys.append(np.nan); lo.append(np.nan); hi.append(np.nan)
        c, ls, mk, lw = STY[m]
        ax.errorbar([0.4, 0.6, 0.8, 1.0], ys, yerr=[np.array(ys) - np.array(lo), np.array(hi) - np.array(ys)], color=c, ls=ls,
                    marker=mk, ms=3.5, lw=lw, capsize=2, elinewidth=0.6, label=NAMES[m])
    ax.set_xticks([0.4, 0.6, 0.8, 1.0]); ax.set_xlabel('Participation probability $p_{\\mathrm{act}}$')
    ax.set_ylim(0, 0.75); ax.grid(axis='y', color='0.9', lw=0.5)
    ax.set_title('(b) participation', fontsize=7.5, pad=4)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize=6.8)
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=2.0)
    fig.savefig(path, dpi=400); plt.close(fig)


def fig_sens(path):
    base = RES.get('E1_cose')
    bvals = [v for s, v in zip(base['seeds'], base['acc']['vals']) if s in (10, 11, 12)]
    panels = [('$\\tau_{\\mathrm{acc}}$', [('tacc35', 0.35), (None, 0.5), ('tacc65', 0.65)]),
              ('capsule rank $r_c$', [('rk2', 2), ('rk4', 4), (None, 8)]),
              ('scope threshold $\\tau_r$', [('tr10', 0.10), (None, 0.15), ('tr20', 0.20)]),
              ('emission period (rounds)', [('em2', 2), (None, 3), ('em6', 6)])]
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.9), sharey=True)
    for ax, (lab, pts) in zip(axes, panels):
        xs, ys, e = [], [], []
        for k, x in pts:
            vals = bvals if k is None else (RES.get(f'E3_{k}', {}).get('acc', {}).get('vals', []))
            if not vals:
                continue
            xs.append(x); ys.append(np.mean(vals)); e.append(np.std(vals))
        ax.errorbar(xs, ys, yerr=e, color='black', marker='o', ms=3.5, lw=1.0, capsize=2, elinewidth=0.6)
        ax.set_xlabel(lab); ax.set_xticks([x for _, x in pts]); ax.grid(axis='y', color='0.9', lw=0.5)
        ax.set_ylim(0.45, 0.68)
    axes[0].set_ylabel('Final mean accuracy')
    fig.tight_layout(w_pad=1.2)
    fig.savefig(path, dpi=400); plt.close(fig)


def fig_rollback(path):
    modes = ['none', 'direct', 'lineage', 'repair', 'reset']
    lab = ['No action', 'Direct\nremoval', 'Lineage\ndeletion', 'Weighted\nrepair', 'Federation\nreset']
    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    w = 0.26
    for j, (key, g, h, nm) in enumerate([('rb_pre', '0.85', '', 'before audit (round 27)'),
                                           ('rb_post', '0.55', '////', 'immediately after'),
                                           ('final', '0.2', '', 'final (round 36)')]):
        ys = []
        for m in modes:
            r = RES.get(f'E10_off_{m}', {})
            ys.append(r['acc']['mean'] if key == 'final' and 'acc' in r else r.get(key, np.nan))
        ax.bar(np.arange(5) + (j - 1) * w, ys, w * 0.92, color=g, edgecolor='black', lw=0.5, hatch=h, label=nm)
    ax.set_xticks(range(5)); ax.set_xticklabels(lab, fontsize=6.6); ax.set_ylabel('Mean accuracy (honest nodes)')
    ax.set_ylim(0.0, 0.75); ax.grid(axis='y', color='0.9', lw=0.5)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, fontsize=6.3, handlelength=1.6, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=400); plt.close(fig)


def fig_reliability(path):
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.plot([0, 1], [0, 1], color='0.6', lw=0.6, ls=':')
    for m in ['cose', 'median', 'fedavg', 'local']:
        rel = np.array(RES[f'E1_{m}']['rel'])
        n = rel[:, 2]; ok = n > 50
        conf = rel[ok, 0] / n[ok]; acc = rel[ok, 1] / n[ok]
        c, ls, mk, lw = STY[m]
        ax.plot(conf, acc, color=c, ls=ls, marker=mk, ms=3.5, lw=lw,
                label=f"{NAMES[m]} (ECE {RES[f'E1_{m}']['ece']['mean']:.3f})")
    ax.set_xlabel('Mean confidence in bin'); ax.set_ylabel('Accuracy in bin'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(color='0.92', lw=0.5)
    ax.legend(loc='upper left', fontsize=6.4, handlelength=2.2, bbox_to_anchor=(0.0, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=400); plt.close(fig)


if __name__ == '__main__':
    import sys
    which = sys.argv[1:] or ['curves', 'domains', 'gate', 'robust', 'sens', 'rollback', 'reliability']
    for w in which:
        globals()['fig_' + w](f'figs/fig_{w}.png')
        print('done', w)
