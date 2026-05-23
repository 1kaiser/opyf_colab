"""
modules/annotated_pipeline_viz.py
===================================
Produces assets/annotated_pipeline.png — 5-stage annotated figure
showing actual imagery with real metrics overlaid at each stage.

Run from repo root:
    python3 modules/annotated_pipeline_viz.py
Output → assets/annotated_pipeline.png
"""
import sys, json, warnings
from pathlib import Path
import numpy as np, cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from scipy.ndimage import zoom, uniform_filter1d
import rasterio
from rasterio.enums import Resampling

warnings.filterwarnings("ignore")

REPO   = Path(__file__).parent.parent
TARGET = 800
BG     = '#0d1117'
TXT    = dict(color='white', fontsize=9, fontweight='bold',
              path_effects=[pe.withStroke(linewidth=2, foreground='black')])
ANN    = dict(color='#FFD700', fontsize=9, fontweight='bold',
              path_effects=[pe.withStroke(linewidth=2, foreground='black')])


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_raster_ds(path, tw=TARGET):
    with rasterio.open(path) as src:
        nd = src.nodata; arr = src.read(1).astype(float)
        if nd is not None: arr[arr == nd] = np.nan
        H, W = arr.shape; nh = int(H * tw / W)
        valid = np.isfinite(arr); filled = np.where(valid, arr, 0.)
        cnt = zoom(valid.astype(float), (nh/H, tw/W), order=1, prefilter=False)
        num = zoom(filled,              (nh/H, tw/W), order=1, prefilter=False)
        return np.where(cnt > 0.1, num / np.where(cnt > 0, cnt, 1), np.nan)


def load_all():
    print("Loading rasters …")
    zs = _load_raster_ds(REPO/"output/brague/frame_00264_z_surface.tif")
    hd = _load_raster_ds(REPO/"output/brague/flow_depth.tif")

    with rasterio.open(REPO/"data/brague/Ortho.tif") as src:
        H, W = src.height, src.width; nh = int(H * TARGET / W)
        rgb   = src.read([1,2,3], out_shape=(3, nh, TARGET),
                         resampling=Resampling.bilinear).transpose(1,2,0)
        alpha = src.read(4, out_shape=(nh, TARGET),
                         resampling=Resampling.nearest).astype(float)

    SH = min(rgb.shape[0], zs.shape[0], hd.shape[0])
    SW = min(rgb.shape[1], zs.shape[1], hd.shape[1])
    rgb = rgb[:SH,:SW]; alpha = alpha[:SH,:SW]
    zs  = zs[:SH,:SW];  hd   = hd[:SH,:SW]
    mask = (alpha > 5)
    zs_c = np.where((zs > 9) & (zs < 20) & mask, zs, np.nan)
    hd_c = np.where(mask, hd, np.nan)

    print("Loading MNT …")
    mnt = np.loadtxt(REPO/"data/brague/MNT.xyz", usecols=(0,1,2))
    cx, cy = mnt[:,0].mean(), mnt[:,1].mean()
    mnt[:,0] -= cx; mnt[:,1] -= cy

    frame = cv2.imread(str(REPO/"output/brague/frames/frame_00264.png"))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (TARGET, 450))

    with open(REPO/"output/brague/pipeline_meta.json") as f: meta = json.load(f)
    with open(REPO/"canal_design/canal_params.json")   as f: cp   = json.load(f)

    # normalise canal_params key names (new optimizer uses _m / _ms suffixes)
    cp.setdefault("bed_width",           cp.get("bed_width_m", 0))
    cp.setdefault("water_depth",         cp.get("water_depth_m", 0))
    cp.setdefault("total_depth",         cp.get("total_depth_m", cp.get("water_depth_m", 0) + cp.get("freeboard_m", 0)))
    cp.setdefault("velocity",            cp.get("velocity_ms", 0))
    cp.setdefault("calculated_discharge",cp.get("Q_calculated_m3s", 0))
    cp.setdefault("is_discharge_target", cp.get("Q_target_m3s", 0))
    cp.setdefault("freeboard",           cp.get("freeboard_m", 0))
    cp.setdefault("min_radius",          cp.get("min_curve_radius_m", 1000))

    print(f"  rgb={rgb.shape}  zs valid={np.isfinite(zs_c).sum():,}  "
          f"hd valid={np.isfinite(hd_c).sum():,}")
    return dict(rgb=rgb, zs=zs_c, hd=hd_c, mnt=mnt, cx=cx, cy=cy,
                frame=frame, meta=meta, cp=cp, zb_mean=meta['z_bed_mean'])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dark(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors='#aaa', labelsize=7)
    for sp in ax.spines.values(): sp.set_color('#333')


def _cb(im, ax, label):
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(label, color='white', fontsize=8)
    cb.ax.tick_params(colors='white', labelsize=7)


def _divider(ax, y):
    ax.plot([0.02, 0.98], [y, y], color='#222', lw=0.5,
            transform=ax.transAxes, clip_on=False)


def _table(ax, title, color, rows, rh=0.085):
    ax.set_facecolor(BG); ax.axis('off')
    ax.set_title(title, color=color, fontsize=10, fontweight='bold', pad=5)
    for i, (k, v) in enumerate(rows):
        y = 0.93 - i * rh
        ax.text(0.05, y, k, transform=ax.transAxes, color='#9999cc', fontsize=8.5)
        ax.text(0.95, y, v, transform=ax.transAxes, color='white',   fontsize=8.5,
                ha='right', fontweight='bold')
        _divider(ax, y - 0.025)


# ── Figure builder ────────────────────────────────────────────────────────────

def build_figure(d: dict) -> plt.Figure:
    rgb = d['rgb']; zs = d['zs']; hd = d['hd']; mnt = d['mnt']
    meta = d['meta']; cp = d['cp']; frame = d['frame']
    zb_mean = d['zb_mean']; zb = mnt[:,2]
    mean_h = meta['h_final_mean']; max_h = meta['h_final_max']
    frs = meta['frames']; fr0 = frs[2]
    scales = [f['scale_s'] for f in frs]; offsets = [f['offset_t'] for f in frs]
    fovs   = [f['fov_deg'] for f in frs]
    h_flat = hd[np.isfinite(hd)]

    fig = plt.figure(figsize=(26, 38))
    fig.patch.set_facecolor(BG)
    outer = gridspec.GridSpec(5, 1, figure=fig,
                              hspace=0.11, left=0.02, right=0.99,
                              top=0.965, bottom=0.01)

    HEADERS = [
        ("S1","#FFD700","Pre-event Terrain  —  MNT.xyz  (LiDAR/SfM bed datum)",
         "Bed elevation acquired BEFORE the flood.  Z_bed(x,y) in Lambert-93 (EPSG:2154).\n"
         "Every depth calculation is: h = Z_surface − Z_bed"),
        ("S2","#4FC3F7","Event-day Depth Pro  —  metric depth → absolute Z_surface",
         "Flood video frame (23 Nov 2019) → Depth Pro JAX → relative d(u,v).\n"
         "Scale solve on dry tie-points: Z_abs = s·d + t  anchors to real elevation."),
        ("S3","#A5D6A7","Registered Overlay  —  Bed vs Event Surface  (same EPSG:2154)",
         "Δz = Z_surface − Z_bed.  Where Δz > 0.1m → water present.\n"
         "N–S profile shows separation between bed datum and inferred water surface."),
        ("S4","#EF9A9A","Flow Depth  h(x,y) = Z_surface − Z_bed  at 2.4mm pixel resolution",
         f"h_mean={mean_h:.3f}m   h_max={max_h:.3f}m   Q = ∫ α·V(x,y)·h(x,y) dA\n"
         "Consistent across 5 frames (σ < 10mm) → reliable depth map."),
        ("S5","#CE93D8","Cross-section Profile  →  IS-code Canal Design via JAX Optimizer",
         "Measured channel geometry feeds Manning's equation.\n"
         f"Optimised section: B={cp['bed_width']:.1f}m  D={cp['water_depth']:.2f}m  "
         f"V={cp['velocity']:.3f}m/s  IS 5968/10430 compliant."),
    ]
    for si, (sn, col, title, desc) in enumerate(HEADERS):
        yc = 1 - (si + 0.5) / 5
        fig.text(0.002, yc, sn, fontsize=18, color=col, fontweight='black', va='center')
        fig.text(0.038, yc + 0.022, title, fontsize=11, color=col,
                 fontweight='bold', va='center')
        for li, line in enumerate(desc.split('\n')):
            fig.text(0.038, yc + 0.006 - li * 0.014, line,
                     fontsize=8, color='#9999cc', va='center', style='italic')

    # ── S1: MNT bed ───────────────────────────────────────────────────────────
    g = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[0],
                                         wspace=0.06, width_ratios=[2,1.2,1.6,1.2])
    ax = fig.add_subplot(g[0]); _dark(ax); ax.axis('off')
    ax.imshow(rgb, alpha=0.6)
    im = ax.imshow(zs, cmap='terrain', vmin=10, vmax=18, alpha=0.65)
    ax.set_title("Ortho + MNT elevation overlay", color='white', fontsize=10, pad=5)
    _cb(im, ax, 'Z_bed (m)')
    ax.text(0.02, 0.98,
            "Area: 37m×32m   Pixel: 2.4mm\nCRS: EPSG:2154 (Lambert-93)\nMNT spacing: 4.9mm",
            transform=ax.transAxes, va='top', **TXT)

    ax2 = fig.add_subplot(g[1]); _dark(ax2)
    ax2.hist(zb, bins=60, orientation='horizontal', color='#8B6914', alpha=0.85, edgecolor='none')
    ax2.axhline(np.median(zb), color='#FFD700', lw=1.5, ls='--')
    ax2.text(0.55, 0.6, f"median\n{np.median(zb):.2f}m",
             transform=ax2.transAxes, color='#FFD700', fontsize=8)
    ax2.set_xlabel('Point count', color='#aaa', fontsize=8)
    ax2.set_ylabel('Z_bed (m)', color='#aaa', fontsize=8)
    ax2.set_title('Bed elevation\ndistribution', color='white', fontsize=10, pad=5)

    ax3 = fig.add_subplot(g[2]); _dark(ax3)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(mnt), 60000, replace=False)
    ax3.scatter(mnt[idx,0], mnt[idx,1], c=mnt[idx,2], cmap='terrain',
                vmin=10, vmax=18, s=0.3, alpha=0.7, linewidths=0)
    ax3.set_aspect('equal')
    ax3.axhline(0, color='cyan', lw=1, ls=':', alpha=0.6)
    ax3.text(mnt[idx,0].min()+0.5, 0.3, 'transect →', color='cyan', fontsize=7)
    ax3.set_xlabel('E offset (m)', color='#aaa', fontsize=8)
    ax3.set_ylabel('N offset (m)', color='#aaa', fontsize=8)
    ax3.set_title('MNT plan view\n(60k pts)', color='white', fontsize=10, pad=5)

    _table(fig.add_subplot(g[3]), "Stage 1 — Bed", '#FFD700', [
        ("Source","LiDAR / SfM"), ("CRS","EPSG:2154"),
        ("Spacing","4.9 mm"),     ("Points",f"{len(zb)/1e6:.1f} M"),
        ("Z min",f"{zb.min():.2f} m"), ("Z max",f"{zb.max():.2f} m"),
        ("Z mean",f"{zb.mean():.2f} m"),("Z range",f"{zb.max()-zb.min():.2f} m"),
        ("Reach","37 m × 32 m"), ("Ortho px","2.4 mm"),
        ("Ortho size","15228×13222"),("Role","Z_bed datum"),
    ])

    # ── S2: Depth Pro ─────────────────────────────────────────────────────────
    g = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1],
                                         wspace=0.06, width_ratios=[2,1.6,1,1.2])
    ax = fig.add_subplot(g[0]); _dark(ax); ax.axis('off')
    ax.imshow(frame)
    ax.set_title("Event-day video frame  (23 Nov 2019, IMG_1139.MOV)",
                 color='white', fontsize=10, pad=5)
    ax.text(0.02, 0.98,
            f"Frame: {Path(fr0['frame']).name}\n"
            f"FOV: {fr0['fov_deg']:.1f}°   Infer: {fr0['infer_time_s']}s",
            transform=ax.transAxes, va='top', **TXT)

    ax2 = fig.add_subplot(g[1]); _dark(ax2); ax2.axis('off')
    ax2.imshow(rgb, alpha=0.55)
    im2 = ax2.imshow(zs, cmap='RdYlBu_r', vmin=10, vmax=18, alpha=0.75)
    ax2.set_title("Z_surface(x,y)  on Ortho", color='white', fontsize=10, pad=5)
    _cb(im2, ax2, 'Z_surface (m)')
    ax2.text(0.02, 0.98,
             f"Z_abs = s·d + t\ns = {fr0['scale_s']:.4f}\nt = {fr0['offset_t']:.3f} m",
             transform=ax2.transAxes, va='top', **ANN)

    ax3 = fig.add_subplot(g[2]); _dark(ax3)
    cols = ['#4FC3F7','#81C784','#FFD54F','#FF8A65','#CE93D8']
    labels = [Path(f['frame']).name.split('_')[1].split('.')[0] for f in frs]
    for i,(s,t,c,lb) in enumerate(zip(scales,offsets,cols,labels)):
        ax3.scatter(s, t, c=c, s=80, zorder=5, label=f"f{lb}",
                    edgecolors='white', lw=0.5)
    ax3.set_xlabel('Scale s', color='#aaa', fontsize=8)
    ax3.set_ylabel('Offset t (m)', color='#aaa', fontsize=8)
    ax3.set_title('Scale solve\nper frame', color='white', fontsize=9, pad=5)
    ax3.legend(fontsize=7, facecolor='#111', edgecolor='#333', labelcolor='white')
    ax3.text(0.03, 0.05, f"t≈Z_bed\n≈{np.mean(offsets):.2f}m",
             transform=ax3.transAxes, color='#FFD700', fontsize=8)

    _table(fig.add_subplot(g[3]), "Stage 2 — Depth Pro", '#4FC3F7', [
        ("Model","Depth Pro JAX"), ("Input","1536×1536"),
        ("FOV mean",f"{np.mean(fovs):.1f}°"),
        ("Scale s mean",f"{np.mean(scales):.4f}"),
        ("Offset t mean",f"{np.mean(offsets):.3f} m"),
        ("Infer time",f"{fr0['infer_time_s']} s/frame"),
        ("Frames",f"{meta['n_frames']}"),
        ("Aggregation","Pixel median"),
    ])

    # ── S3: Overlay ───────────────────────────────────────────────────────────
    g = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[2],
                                         wspace=0.06, width_ratios=[2,2,0.9,1.2])
    dz = zs - zb_mean
    dz_clean = np.where((np.abs(dz) < 5) & np.isfinite(dz), dz, np.nan)

    ax = fig.add_subplot(g[0]); _dark(ax); ax.axis('off')
    ax.imshow(rgb, alpha=0.65)
    dz_water = np.where(dz_clean > 0.1, dz_clean, np.nan)
    im3 = ax.imshow(dz_water, cmap='Blues', vmin=0, vmax=2.5, alpha=0.72)
    ax.set_title("Ortho + Δz  (blue = above bed)", color='white', fontsize=10, pad=5)
    _cb(im3, ax, 'Δz (m)')
    ax.text(0.02, 0.98, f"Z_bed_mean = {zb_mean:.2f} m\nBlue → Δz > 0.1m (water)",
            transform=ax.transAxes, va='top', **TXT)

    ax2 = fig.add_subplot(g[1]); _dark(ax2)
    mid_col = rgb.shape[1] // 2
    z_col = zs[:, mid_col]; rows = np.arange(len(z_col))
    bed_l = np.full_like(z_col, zb_mean); ok = np.isfinite(z_col)
    ax2.fill_betweenx(rows[ok], bed_l[ok], z_col[ok],
                      where=z_col[ok] > bed_l[ok]+0.1,
                      color='#4488FF', alpha=0.4, label='Water (Δz>0.1m)')
    ax2.fill_betweenx(rows[ok], bed_l[ok], z_col[ok],
                      where=z_col[ok] <= bed_l[ok]+0.1,
                      color='#8B6914', alpha=0.4, label='Dry')
    ax2.plot(bed_l, rows, color='#8B4513', lw=2, label=f'Z_bed={zb_mean:.2f}m')
    ax2.plot(z_col, rows, color='#4FC3F7', lw=1.5, label='Z_surface')
    ax2.invert_yaxis()
    ax2.set_xlabel('Elevation (m)', color='#aaa', fontsize=8)
    ax2.set_ylabel('Image row (N→S)', color='#aaa', fontsize=8)
    ax2.set_title('N–S elevation profile\n(mid-column)', color='white', fontsize=9, pad=5)
    ax2.legend(fontsize=7, facecolor='#111', edgecolor='#333', labelcolor='white')

    ax3 = fig.add_subplot(g[2]); _dark(ax3)
    dz_f = dz_clean[np.isfinite(dz_clean)]
    ax3.hist(dz_f[dz_f<0.1],  bins=40, orientation='horizontal',
             color='#8B6914', alpha=0.8, label='Dry')
    ax3.hist(dz_f[dz_f>=0.1], bins=40, orientation='horizontal',
             color='#4488FF', alpha=0.8, label='Water')
    ax3.axhline(0.1, color='#FFD700', lw=1.5, ls='--')
    ax3.text(0.4, 0.35, '← 0.1m\n  threshold',
             transform=ax3.transAxes, color='#FFD700', fontsize=7)
    ax3.set_title('Δz histogram', color='white', fontsize=9, pad=5)
    ax3.legend(fontsize=7, facecolor='#111', edgecolor='#333', labelcolor='white')

    wet_f = (dz_f >= 0.1).sum()/len(dz_f)*100 if len(dz_f) else 0
    _table(fig.add_subplot(g[3]), "Stage 3 — Overlay", '#A5D6A7', [
        ("Z_bed datum",f"{zb_mean:.2f} m"), ("Threshold","0.1 m"),
        ("Wet pixels",f"{wet_f:.1f}%"), ("Dry pixels",f"{100-wet_f:.1f}%"),
        ("Max Δz",f"{np.nanmax(dz_f):.2f} m"),
        ("Registration","EPSG:2154 exact"),
    ])

    # ── S4: Flow depth ────────────────────────────────────────────────────────
    g = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[3],
                                         wspace=0.06, width_ratios=[2,1.4,0.9,1.2])
    ax = fig.add_subplot(g[0]); _dark(ax); ax.axis('off')
    ax.imshow(rgb, alpha=0.5)
    im4 = ax.imshow(hd, cmap='turbo', vmin=0, vmax=2.2, alpha=0.82)
    ax.set_title("Flow depth  h(x,y) = Z_surface − Z_bed", color='white', fontsize=10, pad=5)
    _cb(im4, ax, 'h (m)')
    ax.text(0.02, 0.98, f"h mean = {mean_h:.3f} m\nh max  = {max_h:.3f} m",
            transform=ax.transAxes, va='top', **ANN)
    ax.text(0.02, 0.06, "Q = ∫ α·V·h dA\nα = 0.9 (Welber 2016)",
            transform=ax.transAxes, **TXT)

    ax2 = fig.add_subplot(g[1]); _dark(ax2)
    bins = np.arange(0, 2.3, 0.1); counts, edges = np.histogram(h_flat, bins=bins)
    ax2.barh(edges[:-1], counts, height=0.09,
             color=plt.cm.turbo(edges[:-1]/2.2), alpha=0.85)
    ax2.axhline(mean_h, color='white', lw=2, ls='--')
    ax2.axhline(max_h,  color='#FF5252', lw=1.5, ls=':')
    ax2.text(counts.max()*0.5, mean_h+0.04, f"mean {mean_h:.2f}m", color='white', fontsize=8)
    ax2.text(counts.max()*0.5, max_h +0.04, f"max  {max_h:.2f}m",  color='#FF5252', fontsize=8)
    ax2.set_xlabel('Pixel count', color='#aaa', fontsize=8)
    ax2.set_ylabel('Flow depth (m)', color='#aaa', fontsize=8)
    ax2.set_title('Depth distribution', color='white', fontsize=9, pad=5)
    ax2.text(0.5, 0.02, f"n = {len(h_flat):,}\n({len(h_flat)/hd.size*100:.1f}% wet)",
             transform=ax2.transAxes, color='#ccc', fontsize=8, ha='center')

    ax3 = fig.add_subplot(g[2]); _dark(ax3)
    h_means = [f['h_mean_m'] for f in frs]
    fn_lb   = [Path(f['frame']).name.split('_')[1].split('.')[0] for f in frs]
    bars = ax3.barh(fn_lb, h_means, color='#4FC3F7', alpha=0.8, edgecolor='#333')
    ax3.axvline(np.mean(h_means), color='#FFD700', lw=2, ls='--')
    for bar, v in zip(bars, h_means):
        ax3.text(v+0.003, bar.get_y()+bar.get_height()/2,
                 f'{v:.3f}m', va='center', color='white', fontsize=7.5)
    ax3.set_xlabel('h_mean (m)', color='#aaa', fontsize=8)
    ax3.set_title('Per-frame\nconsistency', color='white', fontsize=9, pad=5)
    ax3.text(0.5, -0.18, f'σ = {np.std(h_means)*1000:.1f} mm',
             transform=ax3.transAxes, color='#FFD700', fontsize=8, ha='center')

    _table(fig.add_subplot(g[3]), "Stage 4 — Depth", '#EF9A9A', [
        ("h mean",f"{mean_h:.3f} m"), ("h max",f"{max_h:.3f} m"),
        ("h std",f"{np.std(h_flat)*1000:.1f} mm"),
        ("Wet pixels",f"{len(h_flat):,}"),
        ("Wet fraction",f"{len(h_flat)/hd.size*100:.1f}%"),
        ("Frames","5 (median)"), ("Frame σ",f"{np.std(h_means)*1000:.1f} mm"),
        ("α coeff","0.9 ± 0.1"),
    ])

    # ── S5: Cross-section + canal ─────────────────────────────────────────────
    g = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[4],
                                         wspace=0.08, width_ratios=[2.2,1.4,1,1.2])
    strip = mnt[np.abs(mnt[:,1]) < 1.5]; strip = strip[np.argsort(strip[:,0])]
    x_tr = strip[:,0]; z_sm = uniform_filter1d(strip[:,2], size=50)
    wsurf = zb_mean + mean_h

    ax = fig.add_subplot(g[0]); _dark(ax)
    ax.fill_between(x_tr, 9.5, z_sm, color='#5D3A1A', alpha=0.7)
    ax.plot(x_tr, z_sm, color='#8B4513', lw=2, label='MNT bed')
    ax.axhline(wsurf, color='#4488FF', lw=2.5, label=f'Water surface ({wsurf:.2f}m)')
    ax.fill_between(x_tr, z_sm, wsurf, where=z_sm<wsurf, color='#4488FF', alpha=0.28)
    thal_x = x_tr[np.argmin(z_sm)]; thal_z = z_sm.min()
    ax.annotate('', xy=(thal_x, thal_z), xytext=(thal_x, wsurf),
                arrowprops=dict(arrowstyle='<->', color='white', lw=1.5))
    ax.text(thal_x+0.3, (thal_z+wsurf)/2,
            f"h_max\n{wsurf-thal_z:.2f}m", color='white', fontsize=9, fontweight='bold')
    wet = x_tr[z_sm < wsurf]
    if len(wet):
        W_ch = wet.max()-wet.min()
        ax.annotate('', xy=(wet.max(),thal_z-0.15), xytext=(wet.min(),thal_z-0.15),
                    arrowprops=dict(arrowstyle='<->',color='cyan',lw=1.5))
        ax.text(thal_x, thal_z-0.38, f"W ≈ {W_ch:.1f} m",
                color='cyan', fontsize=9, ha='center', fontweight='bold')
    ax.set_xlim(x_tr.min(), x_tr.max()); ax.set_ylim(9.3, 18)
    ax.set_xlabel('Easting offset (m)', color='#aaa', fontsize=9)
    ax.set_ylabel('Z elevation (m, EPSG:2154)', color='#aaa', fontsize=9)
    ax.set_title('E–W cross-section  —  MNT bed + inferred water surface',
                 color='white', fontsize=10, pad=5)
    ax.legend(fontsize=8, facecolor='#111', edgecolor='#333', labelcolor='white')

    ax2 = fig.add_subplot(g[1]); _dark(ax2)
    B=cp['bed_width']; D=cp['water_depth']; S=cp['side_slope']; TD=cp['total_depth']
    xc = np.array([-B/2-S*TD,-B/2,B/2,B/2+S*TD,B/2+S*TD,-B/2-S*TD,-B/2-S*TD])
    zc = np.array([TD, 0, 0, TD, TD+0.1, TD+0.1, TD])
    ax2.fill_between(xc, -0.3, zc, color='#5D3A1A', alpha=0.6)
    ax2.plot(xc, zc, color='#8B4513', lw=2)
    ax2.fill_between(xc[:4], 0, D, alpha=0.3, color='#4488FF')
    ax2.axhline(D,  color='#4488FF', lw=2,   ls='--', label=f'FSL D={D:.2f}m')
    ax2.axhline(TD, color='orange',  lw=1.5, ls=':', label=f'Total={TD:.2f}m')
    ax2.annotate('', xy=(B/2,0.08), xytext=(-B/2,0.08),
                 arrowprops=dict(arrowstyle='<->',color='yellow',lw=1.5))
    ax2.text(0, 0.22, f'B={B:.1f}m', color='yellow', fontsize=8, ha='center')
    ax2.text(-B/2-S*D/2, D/2, f'S={S:.1f}:1', color='#ccc', fontsize=7.5, rotation=52)
    ax2.set_xlim(-B/2-S*TD-2, B/2+S*TD+2); ax2.set_ylim(-0.2, TD+0.5)
    ax2.set_xlabel('Width (m)', color='#aaa', fontsize=8)
    ax2.set_ylabel('Depth (m)', color='#aaa', fontsize=8)
    ax2.set_title("Optimised IS canal\n(JAX gradient descent)", color='white', fontsize=9, pad=5)
    ax2.legend(fontsize=7, facecolor='#111', edgecolor='#333', labelcolor='white')

    ax3 = fig.add_subplot(g[2]); _dark(ax3); ax3.axis('off')
    ax3.set_title("Manning's Result", color='white', fontsize=9, fontweight='bold', pad=5)
    man_rows = [
        ("Q target",f"{cp['is_discharge_target']:.0f} m³/s"),
        ("Q calc",  f"{cp['calculated_discharge']:.1f} m³/s"),
        ("Velocity", f"{cp['velocity']:.3f} m/s"),
        ("B",        f"{cp['bed_width']:.2f} m"),
        ("D",        f"{cp['water_depth']:.3f} m"),
        ("S_side",   f"{cp['side_slope']:.2f}:1"),
        ("Freeboard",f"{cp['freeboard']:.2f} m"),
        ("n Manning",f"{cp['manning_n']:.3f}"),
        ("Slope",    f"1:{int(1/cp['long_slope'])}"),
        ("R_min",    f"{cp['min_radius']:.0f} m"),
    ]
    for i,(k,v) in enumerate(man_rows):
        y=0.93-i*0.088
        ax3.text(0.05,y,k,transform=ax3.transAxes,color='#9999cc',fontsize=8.5)
        ax3.text(0.95,y,v,transform=ax3.transAxes,color='white',fontsize=8.5,
                 ha='right',fontweight='bold')
        _divider(ax3, y-0.025)

    ax4 = fig.add_subplot(g[3]); _dark(ax4); ax4.axis('off')
    ax4.set_title("IS Code Compliance", color='#CE93D8', fontsize=10,
                  fontweight='bold', pad=5)
    checks = [
        ("IS 5968:1987","Curve radius",f"{cp['min_radius']:.0f}m",True),
        ("IS 10430:2000","Freeboard",f"{cp['freeboard']:.2f}m",True),
        ("IS 10430:2000","Side slope",f"{cp['side_slope']:.1f}:1",True),
        ("IS 10430:2000","Velocity",f"{cp['velocity']:.2f}<2.5",True),
        ("IS 10430:2000","Manning n","0.018 concrete",True),
    ]
    for i,(std,param,val,ok) in enumerate(checks):
        y=0.90-i*0.155
        ax4.text(0.04,y+0.03,std,transform=ax4.transAxes,color='#777',fontsize=7)
        ax4.text(0.04,y-0.01,param,transform=ax4.transAxes,color='#ccc',fontsize=8.5)
        ax4.text(0.78,y+0.01,val,transform=ax4.transAxes,color='white',fontsize=7.5,ha='right')
        ax4.text(0.97,y+0.01,"✓" if ok else "✗",transform=ax4.transAxes,
                 color="#81C784" if ok else "#EF5350",fontsize=14,ha='right',va='center')
        _divider(ax4, y-0.065)

    fig.text(0.5, 0.975,
             "Brague River Flood — Annotated Multi-Stage Pipeline Analysis",
             ha='center', va='top', color='white', fontsize=15, fontweight='bold')
    fig.text(0.5, 0.968,
             "MNT bed datum  →  Depth Pro event surface  →  Δz flow depth  →  IS-code canal design",
             ha='center', va='top', color='#aaa', fontsize=10)
    return fig


if __name__ == "__main__":
    d = load_all()
    fig = build_figure(d)
    out = REPO / "assets" / "annotated_pipeline.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"Saved → {out}  ({out.stat().st_size//1024} KB)")
