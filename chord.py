"""
Gradient-colored chord diagram for the kinetic Ising model.

Visualizes directed couplings as a chord diagram with one arc per neuron pair.
Each arc shows both coupling directions simultaneously via a color gradient:
at node i the color encodes J[i,j] (influence of j on i), at node j
the color encodes J[j,i] (influence of i on j), with linear interpolation
along the curve.  Double-headed arrows indicate the bidirectional nature.

- Nodes on a circle, colored by bias (field) parameter h_i = theta_s[t,i,0]
- Arcs colored by gradient interpolation of directed couplings (seismic cmap)
- Arc width proportional to max(|J[i,j]|, |J[j,i]|)
- Arrowheads at both ends, each colored by its endpoint coupling value

Usage:
    from ssll_kinetic.chord import show_chord, show_chord_snapshots

    show_chord(emd, dt=0.02, t=25)
    show_chord_snapshots(emd, dt=0.02, times=[0.1, 0.3, 0.5])
"""

import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


def _ensure_save_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def get_net_flow_matrix(emd, t):
    """
    Extract the directed coupling matrix and compute net flow.

    :param emd: container.EMData
        Fitted model container with theta_s of shape (T, N, N+1).
    :param t: int
        Time bin index.
    :returns: (net, J)
        net: (N, N) antisymmetric matrix, net[i,j] = J[i,j] - J[j,i]
        J: (N, N) directed coupling matrix, J[i,j] = theta_s[t, i, j+1]
    """
    # theta_s[t, i, :] = [h_i, J[i,0], J[i,1], ..., J[i,N-1]]
    J = emd.theta_s[t, :, 1:]  # (N, N) — J[i,j] = influence of j on i
    net = J - J.T               # net[i,j] = J[i,j] - J[j,i]
    return net, J


def _draw_gradient_arc(ax, p_i, p_j, val_i, val_j, edge_cm, edge_norm,
                       linewidth, curvature=0.3, alpha=0.7, n_seg=50):
    """
    Draw a gradient-colored arc between p_i and p_j with arrowheads at both ends.

    The color at the p_i end encodes val_i, the color at p_j encodes val_j,
    with linear interpolation along the quadratic Bezier curve.

    :param p_i, p_j: array-like (2,)
        Endpoint positions.
    :param val_i, val_j: float
        Coupling values at each endpoint (for colormap lookup).
    :param edge_cm: colormap
    :param edge_norm: Normalize
    :param linewidth: float
    :param curvature: float
        0 = straight line, 1 = control point at origin.
    :param alpha: float
    :param n_seg: int
        Number of line segments for the gradient body.
    """
    p_i = np.asarray(p_i, dtype=float)
    p_j = np.asarray(p_j, dtype=float)

    # Quadratic Bezier: B(t) = (1-t)^2 * p_i + 2(1-t)t * ctrl + t^2 * p_j
    midpoint = 0.5 * (p_i + p_j)
    origin = np.array([0.0, 0.0])
    ctrl = midpoint + curvature * (origin - midpoint)

    ts = np.linspace(0, 1, n_seg + 1)
    pts = ((1 - ts)**2)[:, None] * p_i + (2 * (1 - ts) * ts)[:, None] * ctrl + (ts**2)[:, None] * p_j

    # Gradient values along curve
    vals = (1 - ts) * val_i + ts * val_j
    colors = edge_cm(edge_norm(vals))
    colors[:, 3] = alpha

    # Draw body as LineCollection
    segments = np.stack([pts[:-1], pts[1:]], axis=1)  # (n_seg, 2, 2)
    seg_colors = colors[:-1]  # color each segment by its start value
    lc = LineCollection(segments, colors=seg_colors, linewidths=linewidth,
                        zorder=10)
    ax.add_collection(lc)

    # Draw arrowheads at both ends — use points ~8% along the curve for a
    # visible tangent direction (index 4 out of 50 segments).
    k = min(4, n_seg // 4)
    ms = 15 + 4 * linewidth
    # Arrowhead at p_i (pointing inward from curve toward p_i)
    arrow_i = mpatches.FancyArrowPatch(
        posA=tuple(pts[k]), posB=tuple(pts[0]),
        arrowstyle='-|>', color=edge_cm(edge_norm(val_i)),
        linewidth=0, alpha=alpha, mutation_scale=ms, zorder=11)
    ax.add_patch(arrow_i)
    # Arrowhead at p_j (pointing inward from curve toward p_j)
    arrow_j = mpatches.FancyArrowPatch(
        posA=tuple(pts[-1 - k]), posB=tuple(pts[-1]),
        arrowstyle='-|>', color=edge_cm(edge_norm(val_j)),
        linewidth=0, alpha=alpha, mutation_scale=ms, zorder=11)
    ax.add_patch(arrow_j)


def show_chord(emd, dt, t, ax=None,
               node_cmap='hot', edge_cmap='seismic',
               bias_lim=None, limit_ij=None, threshold=0,
               curvature=0.3,
               node_size=300, fp_chord=None, dpi=200, figsize=None):
    """
    Display a net-flow chord diagram for a single time step.

    For each neuron pair (i, j) with i < j, a single directed arc is drawn.
    The arrow points toward the neuron receiving stronger net influence:
    if J[i,j] > J[j,i] (j influences i more), the arrow goes j -> i.

    :param emd: container.EMData
        Fitted model container.
    :param dt: float
        Bin size in seconds.
    :param t: int
        Time bin index.
    :param threshold: float
        Percentile threshold (0-100) on |net flow|. Edges below this
        percentile are hidden. Default 0 (show all).
    :param curvature: float
        Arc curvature. 0 = straight line, 1 = bows through origin.
        Default 0.3.
    :param ax: matplotlib.axes.Axes or None
        If given, draw on this axes. Otherwise create a new figure.
    :param node_cmap: str
        Colormap for nodes (bias parameter). Default: 'hot'.
    :param edge_cmap: str
        Colormap for edges (net flow). Default: 'seismic'.
    :param bias_lim: float or None
        Symmetric limit for node colormap (bias). If None, computed from data.
    :param limit_ij: float or None
        Symmetric limit for edge colormap. If None, computed from data.
    :param node_size: int
        Size of node markers.
    :param fp_chord: str or None
        If given, save figure to this path.
    :param dpi: int
        Resolution for saved figure.
    :param figsize: tuple or None
        Figure size (width, height) in inches. Default: (7, 7).
    """
    N = emd.N
    net, J = get_net_flow_matrix(emd, t)
    bias = emd.theta_s[t, :, 0]  # (N,) bias/field parameter

    # Compute threshold cutoff from upper triangle max(|J[i,j]|, |J[j,i]|)
    triu_idx = np.triu_indices(N, k=1)
    abs_max_couplings = np.maximum(np.abs(J[triu_idx]), np.abs(J.T[triu_idx]))
    if threshold > 0 and len(abs_max_couplings) > 0:
        cutoff = np.percentile(abs_max_couplings, threshold)
    else:
        cutoff = 0.0

    # Node positions on unit circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pos = np.column_stack([np.cos(angles), np.sin(angles)])

    # Auto-scale node size and font for large N so nodes nearly touch.
    # Calibrated: at N=20, s=300 looks good; at N=50, s=250 is nearly
    # adjacent. Scale linearly in gap ratio (not quadratic) to keep
    # markers large enough.
    fontsize = 8
    if N > 20:
        gap = 2 * np.sin(np.pi / N)
        gap_ref = 2 * np.sin(np.pi / 20)  # 0.3129
        node_size = max(15, min(node_size, int(375 * np.sqrt(gap / gap_ref))))
        fontsize = max(4, min(8, 8 * np.sqrt(node_size / 300)))

    # Color limits
    if bias_lim is None:
        bias_lim = np.max(np.abs(emd.theta_s[:, :, 0]))
    if limit_ij is None:
        limit_ij = np.max(np.abs(emd.theta_s[:, :, 1:]))
    if limit_ij == 0:
        limit_ij = 1.0

    # Create figure/axes
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=figsize or (7, 7))

    node_cm = plt.get_cmap(node_cmap)
    edge_cm = plt.get_cmap(edge_cmap)
    edge_norm = Normalize(vmin=-limit_ij, vmax=limit_ij)
    node_norm = Normalize(vmin=-bias_lim, vmax=bias_lim)

    # Draw gradient-colored arcs for each pair
    for i in range(N):
        for j in range(i + 1, N):
            val_i = J[i, j]   # influence of j on i (color at node i end)
            val_j = J[j, i]   # influence of i on j (color at node j end)
            if threshold and max(abs(val_i), abs(val_j)) < cutoff:
                continue
            linewidth = np.clip(3.0 * max(abs(val_i), abs(val_j)) / limit_ij,
                                0.5, 3.0)
            _draw_gradient_arc(ax, pos[i], pos[j], val_i, val_j,
                               edge_cm, edge_norm, linewidth,
                               curvature=curvature)

    # Draw nodes colored by bias
    node_colors = [node_cm(node_norm(bias[k])) for k in range(N)]
    ax.scatter(pos[:, 0], pos[:, 1], s=node_size, c=node_colors,
               edgecolors='black', linewidths=0.5, zorder=5)

    # Node labels
    for k in range(N):
        ax.text(pos[k, 0], pos[k, 1], str(k), ha='center', va='center',
                fontsize=fontsize, color='#D3D3D3', zorder=6)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('t = %.2f s' % (t * dt), fontsize=14)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)

    # Colorbars (only for own figure)
    if own_fig:
        fig = ax.get_figure()
        sm_node = plt.cm.ScalarMappable(cmap=node_cm, norm=node_norm)
        sm_node.set_array([])
        cbar_n = fig.colorbar(sm_node, ax=ax, fraction=0.03, pad=0.04,
                              location='left')
        cbar_n.set_label('$h_i$', fontsize=12)

        sm_edge = plt.cm.ScalarMappable(cmap=edge_cm, norm=edge_norm)
        sm_edge.set_array([])
        cbar_e = fig.colorbar(sm_edge, ax=ax, fraction=0.03, pad=0.02)
        cbar_e.set_label('$J_{ij},\\, J_{ji}$', fontsize=12)

        if fp_chord is not None:
            _ensure_save_dir(fp_chord)
            fig.savefig(fp_chord, dpi=dpi, bbox_inches='tight')

        plt.show()


def show_chord_snapshots(emd, dt, times, threshold=0, curvature=0.3,
                         fp_chord=None, dpi=200, figsize=None,
                         node_cmap='hot', edge_cmap='seismic'):
    """
    Display net-flow chord diagrams at multiple time points.

    :param emd: container.EMData
        Fitted model container.
    :param dt: float
        Bin size in seconds.
    :param times: list of float
        Times in seconds at which to show the chord diagram.
    :param threshold: float
        Percentile threshold (0-100) on |net flow|. Default 0.
    :param curvature: float
        Arc curvature. 0 = straight, 1 = through origin. Default 0.3.
    :param fp_chord: str or None
        Path to save the combined figure.
    :param dpi: int
        Resolution for saved figure.
    :param figsize: tuple or None
        Figure size. Default: auto-computed.
    :param node_cmap: str
        Colormap for nodes.
    :param edge_cmap: str
        Colormap for edges.
    """
    N = emd.N
    n_panels = len(times)
    ncols = min(n_panels, 3)
    nrows = int(np.ceil(n_panels / ncols))

    if figsize is None:
        figsize = (6 * ncols, 6 * nrows)

    # Global color limits for consistency across panels
    bias_lim = np.max(np.abs(emd.theta_s[:, :, 0]))
    limit_ij = np.max(np.abs(emd.theta_s[:, :, 1:]))
    if limit_ij == 0:
        limit_ij = 1.0

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_panels == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, time_s in enumerate(times):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        t = int(round(time_s / dt))
        show_chord(emd, dt, t, ax=ax,
                   node_cmap=node_cmap, edge_cmap=edge_cmap,
                   bias_lim=bias_lim, limit_ij=limit_ij,
                   threshold=threshold, curvature=curvature)

    # Hide unused axes
    for idx in range(n_panels, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis('off')

    # Shared colorbars
    node_cm = plt.get_cmap(node_cmap)
    edge_cm = plt.get_cmap(edge_cmap)
    node_norm = Normalize(vmin=-bias_lim, vmax=bias_lim)
    edge_norm = Normalize(vmin=-limit_ij, vmax=limit_ij)

    sm_node = plt.cm.ScalarMappable(cmap=node_cm, norm=node_norm)
    sm_node.set_array([])
    sm_edge = plt.cm.ScalarMappable(cmap=edge_cm, norm=edge_norm)
    sm_edge.set_array([])

    try:
        fig.tight_layout(rect=[0.11, 0.02, 0.89, 0.98])
    except Exception:
        fig.subplots_adjust(left=0.11, right=0.89, top=0.98, bottom=0.02)

    cax_bias = fig.add_axes([0.02, 0.18, 0.018, 0.64])
    cbar_n = fig.colorbar(sm_node, cax=cax_bias)
    cbar_n.set_label('$h_i$', fontsize=12)

    cax_net = fig.add_axes([0.965, 0.18, 0.018, 0.64])
    cbar_e = fig.colorbar(sm_edge, cax=cax_net)
    cbar_e.set_label('$J_{ij},\\, J_{ji}$', fontsize=12)

    if fp_chord is not None:
        _ensure_save_dir(fp_chord)
        fig.savefig(fp_chord, dpi=dpi, bbox_inches='tight')

    plt.show()


def render(N, T=30, R=400, seed=42, max_iter=50, dt=0.02,
           t_single=15, times=None, threshold=0, mu=0.2,
           fig_dir='fig'):
    """
    End-to-end: generate synthetic data, run EM, save chord diagrams.

    :param N: int — number of neurons.
    :param T: int — number of time bins.
    :param R: int — number of trials.
    :param seed: int — random seed.
    :param max_iter: int — max EM iterations.
    :param dt: float — bin size in seconds.
    :param t_single: int — time bin for single-frame diagram.
    :param times: list of float — times for snapshot panels (default [0.1, 0.3, 0.5]).
    :param threshold: float — percentile threshold (0-100).
    :param mu: float — mean coupling strength for synthesis.
    :param fig_dir: str — output directory for PNGs.
    """
    import matplotlib
    matplotlib.use('Agg')
    from . import synthesis, run

    if times is None:
        times = [0.1, 0.3, 0.5]

    np.random.seed(seed)
    thetas = synthesis.generate_thetas(T, N, mu=mu)
    spikes = synthesis.generate_spikes(T, R, N, thetas)
    emd = run(spikes, max_iter=max_iter, mstep=True, state_cov=0.5)

    tag = 'N%d' % N
    _ensure_save_dir(os.path.join(fig_dir, 'dummy'))
    fp_single = os.path.join(fig_dir, 'chord_%s.png' % tag)
    fp_snap = os.path.join(fig_dir, 'chord_%s_snapshots.png' % tag)

    show_chord(emd, dt=dt, t=t_single, threshold=threshold,
               fp_chord=fp_single)
    print('%s single saved: %s' % (tag, fp_single))

    show_chord_snapshots(emd, dt=dt, times=times, threshold=threshold,
                         fp_chord=fp_snap)
    print('%s snapshots saved: %s' % (tag, fp_snap))
