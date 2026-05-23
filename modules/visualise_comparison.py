import numpy as np, rasterio, warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
warnings.filterwarnings("ignore")

REPO  = "/home/kaiser/projects/opyf_colab"
N_PTS = 25_000
rng   = np.random.default_rng(42)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading MNT ...")
mnt = np.loadtxt(f"{REPO}/data/brague/MNT.xyz", usecols=(0,1,2))
cx, cy = mnt[:,0].mean(), mnt[:,1].mean()
mnt[:,0] -= cx; mnt[:,1] -= cy

with rasterio.open(f"{REPO}/output/brague/frame_00264_z_surface.tif") as src:
    zs_arr = src.read(1).astype(float); zs_arr[zs_arr==src.nodata]=np.nan
    H,W = src.height, src.width; T = src.transform

print("Building grids ...")
cols=np.arange(W); rows=np.arange(H)
cg,rg = np.meshgrid(cols,rows)
Xm,Ym = rasterio.transform.xy(T, rg.ravel(), cg.ravel())
Xm=np.array(Xm).reshape(H,W)-cx; Ym=np.array(Ym).reshape(H,W)-cy

with rasterio.open(f"{REPO}/output/brague/flow_depth.tif") as src:
    h_arr = src.read(1).astype(float); h_arr[h_arr==src.nodata]=np.nan

# Sub-sampled clouds
ib  = rng.choice(len(mnt), N_PTS, replace=False)
bed = mnt[ib]

vs  = np.isfinite(zs_arr)
sp  = np.column_stack([Xm[vs],Ym[vs],zs_arr[vs]])
is_ = rng.choice(len(sp), N_PTS, replace=False)
surf = sp[is_]

vw  = np.isfinite(h_arr)
wp  = np.column_stack([Xm[vw],Ym[vw],
                        zs_arr[vw]-h_arr[vw],   # z_bed under water
                        zs_arr[vw],              # z_surface
                        h_arr[vw]])              # depth
iw  = rng.choice(len(wp), N_PTS, replace=False)
wat = wp[iw]

# Global Z range
zlo = min(float(bed[:,2].min()), float(surf[:,2].min()))
zhi = max(float(bed[:,2].max()), float(surf[:,2].max()))
hhi = float(np.nanmax(h_arr))

VIEWS = [
    ("Top",       88, -88),
    ("Front",      8, -88),
    ("Side",       8,   2),
    ("Isometric", 32, -55),
]
CMAPS = {"bed":"YlOrBr","surf":"Blues_r","depth":"turbo"}

BG = '#0d1117'
STAGES = [
    ("1", "Pre-event Bed Surface  —  MNT.xyz",
     "LiDAR/SfM terrain model captured BEFORE the flood event.\n"
     "X,Y = Lambert-93 (EPSG:2154)   Z = absolute elevation above sea level (m)\n"
     "Resolution: 4.9 mm point spacing — this is the DATUM for all depth calculations."),

    ("2", "Event-day Surface  —  Depth Pro JAX inference",
     "Depth Pro infers metric depth from each flood video frame.\n"
     "Relative depths are registered to Lambert-93 using dry tie-points from the MNT.\n"
     "The fitted transform Z_abs = s·d + t  (s≈0.031, t≈11.9 m) anchors to real elevation."),

    ("3", "Registered Overlay  —  Bed (brown) + Event surface (blue)",
     "Both clouds share the same coordinate system.  Vertical GAP = Δz.\n"
     "Where Depth Pro surface EXCEEDS the MNT bed → water is present.\n"
     "Where they align → exposed dry ground (bank, road, bridge)."),

    ("4", "Water Extent  —  Non-aligning pixels  h(x,y) = Z_surface − Z_bed",
     "Pixels where Z_surface > Z_bed are declared WATER.  Colour = flow depth.\n"
     f"Mean depth: {np.nanmean(h_arr):.2f} m    Max depth: {np.nanmax(h_arr):.2f} m\n"
     "Discharge: Q = ∫ α · V_surface(x,y) · h(x,y) dA   [α=0.9, V from LSPIV]"),

    ("5", "Cross-section Profile  —  E–W transect at reach mid-point",
     "Bed profile (brown line) vs water surface (blue line) across channel width.\n"
     "Blue shading = flow depth.  Profile drives Manning's equation for Q calculation.\n"
     "Optimised canal dimensions (IS 5968/10430) are fitted to this measured geometry."),
]

fig = plt.figure(figsize=(26, 36))
fig.patch.set_facecolor(BG)
outer = gridspec.GridSpec(5, 1, figure=fig, hspace=0.14,
                          left=0.01, right=0.91, top=0.97, bottom=0.02)

def make_3d(fig, gs_cell, bg=BG):
    ax = fig.add_subplot(gs_cell, projection='3d')
    ax.set_facecolor(bg)
    for p in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        p.fill = False; p.set_edgecolor('#2a2a3a')
    ax.tick_params(colors='#555', labelsize=7, pad=1)
    for label in (ax.get_xticklabels()+ax.get_yticklabels()+ax.get_zticklabels()):
        label.set_color('#555')
    ax.xaxis.label.set_color('#777'); ax.yaxis.label.set_color('#777')
    ax.zaxis.label.set_color('#777')
    ax.set_xlabel("E (m)", fontsize=7, labelpad=0)
    ax.set_ylabel("N (m)", fontsize=7, labelpad=0)
    ax.set_zlabel("Z (m)", fontsize=7, labelpad=0)
    return ax

for st_i, (snum, stitle, sdesc) in enumerate(STAGES):
    inner = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer[st_i], wspace=0.05)

    is_profile = (st_i == 4)

    for v_i, (vname, elev, azim) in enumerate(VIEWS):
        ax = make_3d(fig, inner[v_i])
        ax.view_init(elev=elev, azim=azim)

        # View label top of first stage only, then just name
        col_title = vname if st_i == 0 else ""
        ax.set_title(col_title, color='#ddd', fontsize=11,
                     fontweight='bold', pad=4)

        s = 0.4; alpha = 0.7

        if st_i == 0:   # bed
            sc = ax.scatter(bed[:,0],bed[:,1],bed[:,2],
                c=bed[:,2],cmap=CMAPS['bed'],vmin=zlo,vmax=zhi,
                s=s, alpha=alpha, linewidths=0)

        elif st_i == 1: # surface
            ax.scatter(surf[:,0],surf[:,1],surf[:,2],
                c=surf[:,2],cmap=CMAPS['surf'],vmin=zlo,vmax=zhi,
                s=s, alpha=alpha, linewidths=0)

        elif st_i == 2: # overlay
            ax.scatter(bed[:,0],bed[:,1],bed[:,2],
                c=bed[:,2],cmap=CMAPS['bed'],vmin=zlo,vmax=zhi,
                s=s*0.6, alpha=0.45, linewidths=0)
            ax.scatter(surf[:,0],surf[:,1],surf[:,2],
                c=surf[:,2],cmap=CMAPS['surf'],vmin=zlo,vmax=zhi,
                s=s*0.6, alpha=0.45, linewidths=0)

        elif st_i == 3: # water depth
            ax.scatter(wat[:,0],wat[:,1],wat[:,3],
                c=wat[:,4],cmap=CMAPS['depth'],vmin=0,vmax=hhi,
                s=s, alpha=alpha, linewidths=0)

        else:  # cross-section profile
            mid_r = H // 2
            x_tr  = Xm[mid_r, :]
            zb_tr = zs_arr[mid_r, :] - h_arr[mid_r, :]
            zs_tr = zs_arr[mid_r, :]
            h_tr  = h_arr[mid_r, :]
            ok    = np.isfinite(zb_tr) & np.isfinite(zs_tr)

            xt = x_tr[ok]; zb = zb_tr[ok]; zs2 = zs_tr[ok]; ht = h_tr[ok]
            yn = np.zeros_like(xt)

            # Fill polygon between bed and surface
            verts = (list(zip(xt, yn, zb)) +
                     list(zip(xt[::-1], yn[::-1], zs2[::-1])))
            poly = Poly3DCollection([verts], alpha=0.25, facecolor='#4488FF',
                                    edgecolor='none')
            ax.add_collection3d(poly)
            ax.plot(xt, yn, zb,  color='#8B4513', lw=2.0, label='Bed')
            ax.plot(xt, yn, zs2, color='#4488FF', lw=2.0, label='Surface')
            # Annotate max depth
            imax = np.argmax(ht)
            ax.plot([xt[imax],xt[imax]],[0,0],[zb[imax],zs2[imax]],
                    color='white', lw=1.5, linestyle='--')
            ax.text(xt[imax], 0, (zb[imax]+zs2[imax])/2,
                    f"  h_max\n  {ht[imax]:.2f}m",
                    color='white', fontsize=7)

            # Channel width annotation
            wet_x = xt[ht > 0.1]
            if len(wet_x):
                ax.plot([wet_x[0],wet_x[-1]],[0,0],[zb.min()-0.2]*2,
                        color='cyan', lw=1.5)
                ax.text((wet_x[0]+wet_x[-1])/2, 0, zb.min()-0.5,
                        f"W≈{wet_x[-1]-wet_x[0]:.1f}m",
                        color='cyan', fontsize=7, ha='center')

    # ── Stage header bar (left text) ─────────────────────────────────────────
    yc = 1 - (st_i + 0.5) / 5
    fig.text(0.002, yc,
             f"S{snum}", fontsize=16, color='#FFD700',
             fontweight='bold', va='center', ha='left')
    fig.text(0.025, yc + 0.036, stitle,
             fontsize=11, color='#FFD700', fontweight='bold', va='center')
    for li, line in enumerate(sdesc.split('\n')):
        fig.text(0.025, yc + 0.018 - li*0.015, line,
                 fontsize=8, color='#9999bb', va='center', style='italic')

# ── Colorbars ─────────────────────────────────────────────────────────────────
def add_cb(fig, cmap, vmin, vmax, label, ypos, height=0.14):
    cax = fig.add_axes([0.917, ypos, 0.012, height])
    cb  = plt.colorbar(ScalarMappable(norm=Normalize(vmin,vmax),
                                      cmap=plt.get_cmap(cmap)),
                       cax=cax, label=label)
    cb.ax.yaxis.label.set_color('white')
    cb.ax.tick_params(colors='white', labelsize=8)

add_cb(fig, CMAPS['bed'],   zlo, zhi, 'Z bed (m)',     0.815)
add_cb(fig, CMAPS['surf'],  zlo, zhi, 'Z surface (m)', 0.617)
add_cb(fig, CMAPS['depth'],   0, hhi, 'Depth h (m)',   0.225)

# ── Main title ────────────────────────────────────────────────────────────────
fig.text(0.46, 0.989,
    "Brague River Flood — Multi-Stage Point Cloud Analysis",
    ha='center', va='top', color='white', fontsize=15, fontweight='bold')
fig.text(0.46, 0.983,
    "MNT pre-event bed  ↔  Depth Pro event surface  →  flow depth h(x,y)  →  discharge Q",
    ha='center', va='top', color='#aaa', fontsize=10)

OUT = f"{REPO}/output/brague/multiview_comparison.png"
plt.savefig(OUT, dpi=130, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"Saved {OUT}")
