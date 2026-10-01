"""Architecture diagrams (grayscale).  Every arrow is drawn between explicit anchor points
so its direction and endpoints can be audited against the method description."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({'font.family': 'Liberation Serif', 'mathtext.fontset': 'stix', 'font.size': 7})
ARROWS = []   # (figure, name, start, end, style) log for the arrow audit


class Canvas:
    def __init__(self, name, w, h, xmax, ymax):
        self.name = name
        self.fig = plt.figure(figsize=(w, h))
        self.ax = self.fig.add_axes([0, 0, 1, 1]); self.ax.set_xlim(0, xmax); self.ax.set_ylim(0, ymax)
        self.ax.axis('off'); self.b = {}

    def box(self, key, x, y, w, h, text, fill='white', lw=0.8, fs=7, bold=False, ls='-', round_=0.6, va='center', ha='center'):
        p = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={round_}', fc=fill, ec='black', lw=lw, ls=ls)
        self.ax.add_patch(p)
        tx = x + w / 2 if ha == 'center' else x + 0.8
        ty = y + h / 2 if va == 'center' else y + h - 1.0
        self.ax.text(tx, ty, text, ha=ha, va=va if va == 'center' else 'top', fontsize=fs,
                     fontweight='bold' if bold else 'normal', linespacing=1.15)
        self.b[key] = (x, y, w, h)

    def pt(self, key, side, frac=0.5):
        x, y, w, h = self.b[key]
        return {'l': (x, y + h * frac), 'r': (x + w, y + h * frac), 't': (x + w * frac, y + h), 'b': (x + w * frac, y)}[side]

    def arrow(self, name, p0, p1, dashed=False, conn='arc3', label=None, lpos=None, lw=0.9, fs=6.3):
        a = FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=8, lw=lw, color='black',
                            ls=(0, (4, 2.5)) if dashed else '-', connectionstyle=conn, shrinkA=0, shrinkB=0)
        self.ax.add_patch(a)
        ARROWS.append((self.name, name, tuple(round(v, 1) for v in p0), tuple(round(v, 1) for v in p1), 'dashed' if dashed else 'solid'))
        if label:
            lx, ly = lpos if lpos else ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
            self.ax.text(lx, ly, label, fontsize=fs, ha='center', va='center',
                         bbox=dict(fc='white', ec='none', pad=0.6))

    def poly(self, name, pts, dashed=False, lw=0.9):
        """Orthogonal connector through pts, arrow head on the last segment."""
        for a, b in zip(pts[:-2], pts[1:-1]):
            self.ax.plot([a[0], b[0]], [a[1], b[1]], color='black', lw=lw, ls=(0, (4, 2.5)) if dashed else '-')
        a = FancyArrowPatch(pts[-2], pts[-1], arrowstyle='-|>', mutation_scale=8, lw=lw, color='black',
                            ls=(0, (4, 2.5)) if dashed else '-', shrinkA=0, shrinkB=0)
        self.ax.add_patch(a)
        ARROWS.append((self.name, name, tuple(round(v, 1) for v in pts[0]), tuple(round(v, 1) for v in pts[-1]), 'dashed' if dashed else 'solid'))

    def save(self, path):
        self.fig.savefig(path, dpi=400); plt.close(self.fig)


# ============================================================== Fig. 1: overall framework
def fig_overview(path):
    c = Canvas('overview', 7.16, 4.05, 100, 56.6)
    G1, G2, G3 = '0.93', '0.82', '0.70'
    # ---- sender node
    c.box('S', 1, 1.5, 30, 53, '', fill=G1, lw=0.9)
    c.ax.text(16, 52.2, 'Edge node $i$ (sender)', ha='center', va='center', fontsize=7.6, fontweight='bold')
    c.box('stream', 4, 43, 24, 6.5, 'Non-stationary stream $\\mathcal{S}_i$\n(abrupt domain switches)')
    c.box('bb', 4, 33, 24, 6.5, 'Frozen GPT-2 backbone $f_{\\theta_0}$\n(blocks 0–11, verbaliser $\\mathbf{E}$)', fill=G2)
    c.box('wa', 4, 23, 24, 6.5, 'Scope-routed working adapter\n$\\Delta\\mathbf{W}_i$, gate $g_{\\mathrm{w}}(\\mathbf{x})$')
    c.box('dm', 4, 13, 24, 6.5, 'Drift monitor\nPage–Hinkley on $\\ell_t$  |  key shift')
    c.box('enc', 4, 3.5, 24, 6.5, 'Capsule encoder\nrank-$r_c$ SVD of effective $\\Delta\\mathbf{W}$', fill=G2)
    c.arrow('stream->bb', c.pt('stream', 'b'), c.pt('bb', 't'))
    c.arrow('bb->wa', c.pt('bb', 'b'), c.pt('wa', 't'), label='$\\mathbf{h},\\mathbf{o},\\mathbf{z}$', lpos=(19.6, 31.4))
    c.arrow('wa->dm', c.pt('wa', 'b'), c.pt('dm', 't'), label='prequential loss $\\ell_t$', lpos=(16, 21.4))
    c.arrow('dm->enc', c.pt('dm', 'b'), c.pt('enc', 't'), label='seal / emit', lpos=(16, 11.4))
    # ---- capsule and relay (middle)
    c.box('cap', 35, 27, 19, 20, 'Evolution capsule $\\mathcal{C}$\n\nfactors $\\{\\mathbf{P}_m,\\mathbf{Q}_m\\}$\nscope key $\\boldsymbol{\\kappa}$\nmixing weights $\\{\\bar g_p\\}$\nparents, round, digest',
          fill='white', lw=1.0)
    c.box('relay', 35, 7, 19, 9, 'Publish/subscribe relay\n(transport only,\ndelay $\\leq 2$ rounds)', fill=G1, ls=(0, (3, 2)))
    c.poly('enc->cap', [c.pt('enc', 'r'), (32, 6.75), (32, 37), c.pt('cap', 'l')])
    c.arrow('cap->relay', c.pt('cap', 'b'), c.pt('relay', 't'), label='broadcast', lpos=(44.5, 21.5))
    # ---- receiver node
    c.box('R', 59, 1.5, 40, 53, '', fill=G1, lw=0.9)
    c.ax.text(79, 52.2, 'Edge node $j$ (receiver)', ha='center', va='center', fontsize=7.6, fontweight='bold')
    c.box('gate', 62, 33, 34, 16.2, '', fill='white')
    c.ax.text(79, 47.4, 'Trust gate', ha='center', va='center', fontsize=7.2, fontweight='bold')
    w, y0 = 7.4, 39.0
    for k, (key, t) in enumerate([('ev1', 'buffer\ngain $\\hat g$'), ('ev2', 'peer\nconsensus $\\kappa$'),
                                   ('ev3', 'epistemic\nMI $u$'), ('ev4', 'source\nreputation $\\rho$')]):
        c.box(key, 63 + k * (w + 0.8), y0, w, 6.2, t, fill=G2, fs=6.3)
    c.box('T', 68, 34, 22, 3.6, 'trust score $T$ → accept / defer / reject', fs=6.4)
    for k in range(4):
        c.arrow(f'ev{k+1}->T', c.pt(f'ev{k+1}', 'b'), (69 + k * 6.6, 37.6))
    c.box('spec', 62, 23, 34, 6.5, 'Conflict-aware spectral composition\nkeep $\\gamma_k\\geq-\\tau_\\gamma$ (compatible, complementary)', fill='white')
    c.box('lib', 62, 13, 34, 6.5, 'Routed capsule library $\\mathcal{L}_j$\n(soft scope routing $g_c(\\mathbf{x})$)', fill=G2)
    c.box('prov', 62, 3.5, 15.5, 6.5, 'Temporal\nprovenance graph', fill='white', fs=6.6)
    c.box('rep', 80.5, 3.5, 15.5, 6.5, 'Revocation →\nlineage repair', fill='white', fs=6.6)
    c.poly('relay->gate', [c.pt('relay', 'r'), (57, 11.5), (57, 41), (59.6, 41), (62, 41)])
    c.arrow('gate->spec', c.pt('gate', 'b'), c.pt('spec', 't'), label='accepted', lpos=(85.5, 31.2))
    c.arrow('spec->lib', c.pt('spec', 'b'), c.pt('lib', 't'), label='surviving components', lpos=(79, 21.2))
    c.arrow('lib->prov', c.pt('lib', 'b', 0.229), c.pt('prov', 't'), label='lineage', lpos=(69.8, 11.4))
    c.arrow('prov->rep', c.pt('prov', 'r'), c.pt('rep', 'l'))
    c.poly('rep->lib', [c.pt('rep', 't', 0.5), (88.25, 13)], dashed=True)
    c.ax.text(91.2, 11.4, 'repair', fontsize=6.3, ha='center', va='center', bbox=dict(fc=G1, ec='none', pad=0.4))
    c.save(path)


# ============================================================== Fig. 2: node computation graph
def fig_node(path):
    c = Canvas('node', 7.16, 3.45, 100, 48.2)
    G2 = '0.82'; G1 = '0.93'
    c.box('x', 1, 34, 13, 8, 'Prompted text\n$\\mathbf{x}$ + “Label:”')
    c.box('frozen', 18, 34, 17, 8, 'Frozen GPT-2\nblocks 0–10\n(cached once)', fill=G2)
    c.box('router', 18, 20, 17, 8, 'Scope router\n$s_c=\\mathbf{z}^{\\top}\\boldsymbol{\\kappa}_c$', fill='white')
    c.box('blk', 40, 26, 24, 16, '', fill=G1)
    c.ax.text(52, 40.2, 'Read-out block 11', ha='center', va='center', fontsize=7, fontweight='bold')
    c.box('W0', 41.5, 31.5, 9.5, 6.5, 'frozen\n$\\mathbf{W}_m$', fill=G2, fs=6.5)
    c.box('dW', 53, 31.5, 9.5, 6.5, '$g_{\\mathrm{w}}\\Delta\\mathbf{W}_m$\n(working)', fs=6.5)
    c.box('lib', 41.5, 27, 21, 3.8, '$\\sum_c g_c\\,\\mathbf{P}_{c,m}\\mathbf{Q}_{c,m}$ (library)', fs=6.5)
    c.box('verb', 69, 34, 13, 8, 'Verbaliser\nlogits $\\mathbf{E}\\mathbf{u}$')
    c.box('loss', 86, 34, 13, 8, 'Loss $\\ell$\n(policy labels)', fill='white')
    c.box('dm', 69, 18, 13, 8, 'Drift\nmonitor', fill='white')
    c.box('enc', 86, 18, 13, 8, 'Capsule\nencoder', fill=G2)
    c.box('probe', 40, 4, 24, 9, 'Receiver probe $\\mathbf{D}_m=\\mathbf{0}$\n$\\gamma_k=-\\mathbf{p}_k^{\\top}\\nabla_{\\mathbf{D}_m}\\mathcal{L}\\,\\mathbf{q}_k$', fill='white')
    c.box('buf', 69, 4, 30, 9, 'Labelled buffer $\\mathcal{B}_j$ (in-scope)\nfirst-order transfer gain per component', fill=G1)
    # forward (solid)
    c.arrow('x->frozen', c.pt('x', 'r'), c.pt('frozen', 'l'))
    c.arrow('frozen->blk', c.pt('frozen', 'r'), (40, 38), label='$\\mathbf{h},\\mathbf{o}$', lpos=(37.5, 39.6))
    c.arrow('frozen->router', c.pt('frozen', 'b'), c.pt('router', 't'), label='key $\\mathbf{z}$', lpos=(30.5, 31))
    c.poly('router->blk', [c.pt('router', 'r'), (38, 24), (38, 28.9), (41.5, 28.9)])
    c.ax.text(38.2, 21.9, '$g_c,\\ g_{\\mathrm{w}}$', fontsize=6.3, ha='center', va='center')
    c.arrow('blk->verb', (64, 38), c.pt('verb', 'l'), label='$\\mathbf{u}$', lpos=(66.5, 39.6))
    c.arrow('verb->loss', c.pt('verb', 'r'), c.pt('loss', 'l'))
    c.poly('loss->dm', [(89, 34), (89, 30), (75.5, 30), (75.5, 26)])
    c.ax.text(82.2, 31.3, 'prequential $\\ell_t$', fontsize=6.3, ha='center', va='center')
    c.arrow('dm->enc', c.pt('dm', 'r'), c.pt('enc', 'l'), label='seal', lpos=(84, 23.6))
    c.poly('dW->enc', [(62.5, 33.2), (66.5, 33.2), (66.5, 16), (95.5, 16), (95.5, 18)])
    c.ax.text(81, 14.8, 'effective $\\Delta\\mathbf{W}$ (working + routed parents)', fontsize=6.3, ha='center', va='center')
    c.arrow('buf->probe', c.pt('buf', 'l'), c.pt('probe', 'r'))
    c.arrow('probe->blk (insert D)', (47, 13), (47, 26), label='insert $\\mathbf{D}_m$', lpos=(47, 19.4))
    # reverse-mode gradients (dashed)
    c.poly('grad loss->dW', [c.pt('loss', 't'), (92.5, 45.2), (57.75, 45.2), (57.75, 38)], dashed=True)
    c.ax.text(75, 46.3, 'reverse-mode gradient $\\nabla_{\\mathbf{A},\\mathbf{B}}\\ell$ (training)', fontsize=6.3, ha='center', va='center')
    c.arrow('grad blk->probe', (57, 26), (57, 13), dashed=True, label='$\\nabla_{\\mathbf{D}_m}\\mathcal{L}$', lpos=(57, 19.4))
    c.save(path)


if __name__ == '__main__':
    import json, sys
    fig_overview('figs/fig1_overview.png')
    fig_node('figs/fig2_node.png')
    json.dump(ARROWS, open('figs/arrow_audit.json', 'w'), indent=0)
    for a in ARROWS:
        print(a)
