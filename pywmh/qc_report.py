"""
pywmh.qc_report - Automated Visual Quality Control and Analytics Report Generator

Produces:
1. Self-contained HTML report (embedded Base64 figures, metrics tables, and sanity badges)
2. Standalone summary image (qc_summary.png) for quick thumbnail review
"""

import os
import io
import base64
import time
from typing import Dict, Any, List, Optional
import numpy as np
import nibabel as nib

# Headless matplotlib configuration
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap


def _img_to_base64(fig: plt.Figure) -> str:
    """Converts a matplotlib figure to a base64 encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor="#1e1e24")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def select_informative_slices(lesion_data: np.ndarray, num_slices: int = 12) -> List[int]:
    """
    Selects axial slices with significant lesion volume and anatomical coverage.
    """
    z_sums = np.sum(lesion_data > 0.05, axis=(0, 1))
    active_slices = np.where(z_sums > 0)[0]

    if len(active_slices) < num_slices:
        # Fallback to middle range if very few or no lesions
        total_z = lesion_data.shape[2]
        mid = total_z // 2
        span = min(total_z, 40)
        return sorted(list(np.linspace(mid - span // 2, mid + span // 2, num_slices, dtype=int)))

    # Sample evenly across the 5th to 95th percentile of lesion-containing slices
    min_z, max_z = active_slices[0], active_slices[-1]
    step = (max_z - min_z) / max(1, (num_slices - 1))
    selected = [int(round(min_z + i * step)) for i in range(num_slices)]
    return sorted(list(set(selected)))


def generate_multi_slice_figure(
    flair_data: np.ndarray,
    lesion_data: np.ndarray,
    pv_data: np.ndarray,
    dw_data: np.ndarray,
    slices: List[int]
) -> plt.Figure:
    """
    Generates a high-contrast multi-slice grid with PVWMH (cyan) and DWMH (coral) overlays.
    """
    num_slices = len(slices)
    cols = min(6, num_slices)
    rows = int(np.ceil(num_slices / cols))

    fig = plt.figure(figsize=(cols * 2.8, rows * 3.2), facecolor="#18181b")
    gs = gridspec.GridSpec(rows, cols, wspace=0.03, hspace=0.06, left=0.02, right=0.98, top=0.92, bottom=0.02)

    # Normalize FLAIR background contrast
    flair_masked = flair_data[flair_data > 0]
    p1 = np.percentile(flair_masked, 2) if len(flair_masked) > 0 else 0
    p99 = np.percentile(flair_masked, 99.5) if len(flair_masked) > 0 else 1
    norm_flair = np.clip((flair_data - p1) / max(1e-5, p99 - p1), 0, 1)

    # Colormaps: Cyan for PVWMH, Bright Coral/Red for DWMH
    cmap_pv = LinearSegmentedColormap.from_list("pv", [(0, 1, 0.9, 0), (0, 0.9, 0.8, 0.85)])
    cmap_dw = LinearSegmentedColormap.from_list("dw", [(1, 0.2, 0.3, 0), (1, 0.1, 0.2, 0.85)])

    for i, z in enumerate(slices):
        ax = fig.add_subplot(gs[i])
        ax.axis("off")

        flair_slice = np.rot90(norm_flair[:, :, z])
        ax.imshow(flair_slice, cmap="gray", origin="upper", aspect="equal")

        # Overlay PVWMH
        pv_slice = np.rot90((pv_data[:, :, z] == 1) & (lesion_data[:, :, z] > 0.05))
        if np.any(pv_slice):
            ax.imshow(pv_slice, cmap=cmap_pv, alpha=0.85, origin="upper")

        # Overlay DWMH
        dw_slice = np.rot90((dw_data[:, :, z] == 1) & (lesion_data[:, :, z] > 0.05))
        if np.any(dw_slice):
            ax.imshow(dw_slice, cmap=cmap_dw, alpha=0.85, origin="upper")

        ax.text(
            0.05, 0.92, f"Z={z}",
            transform=ax.transAxes, color="#f4f4f5", fontsize=10,
            weight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="#09090b", alpha=0.7, edgecolor="none")
        )

    fig.suptitle("Axial WMH Parcellation Overlays (Cyan: PVWMH | Coral Red: DWMH)", color="#e4e4e7", fontsize=14, weight="bold", y=0.97)
    return fig


def generate_coreg_check_figure(
    flair_data: np.ndarray,
    t1_data: np.ndarray,
    slice_z: int
) -> plt.Figure:
    """
    Generates edge overlay contour of T1 brain aligned with FLAIR.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), facecolor="#18181b")
    for ax in axes:
        ax.axis("off")

    flair_sl = np.rot90(flair_data[:, :, slice_z])
    t1_sl = np.rot90(t1_data[:, :, slice_z])

    # Normalize
    f_p99 = max(1e-5, np.percentile(flair_sl[flair_sl > 0], 99) if np.any(flair_sl > 0) else 1)
    t_p99 = max(1e-5, np.percentile(t1_sl[t1_sl > 0], 99) if np.any(t1_sl > 0) else 1)
    flair_norm = np.clip(flair_sl / f_p99, 0, 1)
    t1_norm = np.clip(t1_sl / t_p99, 0, 1)

    axes[0].imshow(t1_norm, cmap="gray", origin="upper")
    axes[0].set_title("T1w Structural", color="#d4d4d8", fontsize=11)

    axes[1].imshow(flair_norm, cmap="gray", origin="upper")
    axes[1].set_title("Coregistered FLAIR", color="#d4d4d8", fontsize=11)

    # Edge overlay
    axes[2].imshow(flair_norm, cmap="gray", origin="upper")
    try:
        from scipy.ndimage import gaussian_gradient_magnitude
        t1_edges = gaussian_gradient_magnitude(t1_norm, sigma=1.0)
        edge_mask = t1_edges > np.percentile(t1_edges[t1_edges > 0], 85)
        axes[2].contour(edge_mask, levels=[0.5], colors=["#22c55e"], linewidths=0.7)
    except Exception:
        pass
    axes[2].set_title("T1 Edges (Green) on FLAIR", color="#22c55e", fontsize=11, weight="bold")

    fig.tight_layout()
    return fig


def generate_qc_report(
    flair_path: str,
    t1_path: str,
    lesion_prob_path: str,
    pv_mask_path: str,
    dw_mask_path: str,
    results: Dict[str, Any],
    regional_results: Optional[Dict[str, Any]] = None,
    out_html: str = "WMH_Age_QC_Report.html",
    out_png: str = "qc_summary.png"
) -> str:
    """
    Main orchestrator for automated visual QC report generation.
    """
    flair_img = nib.load(flair_path)
    t1_img = nib.load(t1_path)
    lesion_img = nib.load(lesion_prob_path)
    pv_img = nib.load(pv_mask_path)
    dw_img = nib.load(dw_mask_path)

    flair_data = np.nan_to_num(flair_img.get_fdata())
    t1_data = np.nan_to_num(t1_img.get_fdata())
    lesion_data = np.nan_to_num(lesion_img.get_fdata())
    pv_data = np.nan_to_num(pv_img.get_fdata())
    dw_data = np.nan_to_num(dw_img.get_fdata())

    # 1. Select slices and generate multi-slice overlay figure
    slices = select_informative_slices(lesion_data, num_slices=12)
    fig_multislice = generate_multi_slice_figure(flair_data, lesion_data, pv_data, dw_data, slices)

    # Save standalone summary PNG
    fig_multislice.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="#18181b")
    b64_multislice = _img_to_base64(fig_multislice)

    # 2. Registration check slice
    mid_slice = slices[len(slices) // 2]
    fig_coreg = generate_coreg_check_figure(flair_data, t1_data, mid_slice)
    b64_coreg = _img_to_base64(fig_coreg)

    # 3. Compile Analytics Data
    subj_id = results.get("SubjectID", "Subject")
    pred_age = results.get("Predicted_Age", 0.0)
    total_wmh = results.get("Total_WMH_cc", 0.0)
    pv_wmh = results.get("PVWMH_cc", 0.0)
    dw_wmh = results.get("DWMH_cc", 0.0)
    c_age = results.get("Chronological_Age", None)
    bag = results.get("Brain_Age_Gap", None)
    tiv_cc = results.get("TIV_cc", None)
    wmh_pct_tiv = results.get("Total_WMH_pct_TIV", None)

    # Status badges
    vol_status = "Normal" if total_wmh < 25.0 else ("Moderate" if total_wmh < 40.0 else "Severe")
    vol_color = "#22c55e" if vol_status == "Normal" else ("#f59e0b" if vol_status == "Moderate" else "#ef4444")

    bag_html = ""
    if bag is not None:
        bag_color = "#ef4444" if bag > 5.0 else ("#f59e0b" if bag > 2.0 else "#22c55e")
        bag_sign = f"+{bag:.2f}" if bag > 0 else f"{bag:.2f}"
        bag_html = f"""
        <div class="metric-card">
            <div class="metric-title">Brain Age Gap (BAG)</div>
            <div class="metric-value" style="color: {bag_color};">{bag_sign} <span class="metric-unit">yrs</span></div>
            <div class="metric-sub">Chronological Age: {c_age:.1f} yrs</div>
        </div>
        """

    tiv_html = ""
    if tiv_cc is not None:
        tiv_html = f"""
        <div class="metric-card">
            <div class="metric-title">Normalized Burden (% TIV)</div>
            <div class="metric-value">{wmh_pct_tiv:.3f} <span class="metric-unit">%</span></div>
            <div class="metric-sub">Estimated TIV: {tiv_cc:.1f} cc</div>
        </div>
        """

    # Lobar & Vascular Tables HTML
    lobar_rows = ""
    arterial_rows = ""
    if regional_results:
        for lobe in ["Frontal", "Parietal", "Temporal", "Occipital", "Subcortical"]:
            tot_cc = regional_results.get(f"Lobar_{lobe}_Total_cc", 0.0)
            tot_pct = regional_results.get(f"Lobar_{lobe}_Total_pct", 0.0)
            l_cc = regional_results.get(f"Lobar_{lobe}_Left_cc", 0.0)
            r_cc = regional_results.get(f"Lobar_{lobe}_Right_cc", 0.0)
            lobar_rows += f"""
            <tr>
                <td style="font-weight: 600;">{lobe} Lobe</td>
                <td>{tot_cc:.3f} cc</td>
                <td>
                    <div class="bar-container">
                        <div class="bar-fill" style="width: {min(100, tot_pct)}%;"></div>
                        <span class="bar-label">{tot_pct:.1f}%</span>
                    </div>
                </td>
                <td>{l_cc:.3f} cc</td>
                <td>{r_cc:.3f} cc</td>
            </tr>
            """

        for terr in ["MCA", "ACA", "PCA", "VB"]:
            tot_cc = regional_results.get(f"Arterial_{terr}_Total_cc", 0.0)
            tot_pct = regional_results.get(f"Arterial_{terr}_Total_pct", 0.0)
            l_cc = regional_results.get(f"Arterial_{terr}_Left_cc", 0.0)
            r_cc = regional_results.get(f"Arterial_{terr}_Right_cc", 0.0)
            arterial_rows += f"""
            <tr>
                <td style="font-weight: 600;">{terr} Territory</td>
                <td>{tot_cc:.3f} cc</td>
                <td>
                    <div class="bar-container">
                        <div class="bar-fill" style="width: {min(100, tot_pct)}%; background: #06b6d4;"></div>
                        <span class="bar-label">{tot_pct:.1f}%</span>
                    </div>
                </td>
                <td>{l_cc:.3f} cc</td>
                <td>{r_cc:.3f} cc</td>
            </tr>
            """

    # HTML Template
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>WMH-Age QC Report - {subj_id}</title>
    <style>
        :root {{
            --bg: #09090b;
            --card-bg: #18181b;
            --border: #27272a;
            --text: #f4f4f5;
            --text-muted: #a1a1aa;
            --accent: #3b82f6;
            --cyan: #06b6d4;
            --coral: #f43f5e;
            --green: #22c55e;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            padding: 30px 20px;
            line-height: 1.5;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 20px;
            margin-bottom: 25px;
        }}
        h1 {{ font-size: 24px; font-weight: 700; color: #fff; }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
            background: #27272a;
            color: #d4d4d8;
        }}
        .grid-metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 25px;
        }}
        .metric-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 18px;
        }}
        .metric-title {{ font-size: 13px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }}
        .metric-value {{ font-size: 28px; font-weight: 700; color: #fff; }}
        .metric-unit {{ font-size: 16px; font-weight: 400; color: var(--text-muted); }}
        .metric-sub {{ font-size: 12px; color: var(--text-muted); margin-top: 4px; }}
        
        .section-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 25px;
        }}
        .section-title {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .img-wrapper {{
            text-align: center;
            background: #09090b;
            border-radius: 8px;
            padding: 12px;
            border: 1px solid var(--border);
        }}
        .img-wrapper img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
            text-align: left;
        }}
        th, td {{ padding: 10px 14px; border-bottom: 1px solid var(--border); }}
        th {{ color: var(--text-muted); font-weight: 600; background: #202024; }}
        tr:last-child td {{ border-bottom: none; }}
        
        .bar-container {{
            background: #27272a;
            border-radius: 4px;
            height: 18px;
            width: 160px;
            position: relative;
            overflow: hidden;
            display: inline-flex;
            align-items: center;
        }}
        .bar-fill {{
            background: var(--accent);
            height: 100%;
            border-radius: 4px;
        }}
        .bar-label {{
            position: absolute;
            left: 6px;
            font-size: 11px;
            font-weight: 600;
            color: #fff;
        }}
        footer {{
            text-align: center;
            color: var(--text-muted);
            font-size: 12px;
            margin-top: 40px;
            border-top: 1px solid var(--border);
            padding-top: 15px;
        }}
    </style>
</head>
<body>
<div class="container">
    <header>
        <div>
            <h1>🧠 WMH-Age Estimation & Quality Control Report</h1>
            <div style="font-size: 13px; color: var(--text-muted); margin-top: 4px;">
                Subject: <strong style="color: #fff;">{subj_id}</strong> | Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
        <div>
            <span class="badge" style="background: {vol_color}22; color: {vol_color}; border: 1px solid {vol_color}66;">
                Burden: {vol_status}
            </span>
        </div>
    </header>

    <!-- Key Metrics Grid -->
    <div class="grid-metrics">
        <div class="metric-card">
            <div class="metric-title">Predicted WMH-Age</div>
            <div class="metric-value" style="color: #60a5fa;">{pred_age:.2f} <span class="metric-unit">yrs</span></div>
            <div class="metric-sub">Neurovascular biological age</div>
        </div>
        {bag_html}
        <div class="metric-card">
            <div class="metric-title">Total WMH Volume</div>
            <div class="metric-value">{total_wmh:.3f} <span class="metric-unit">cc</span></div>
            <div class="metric-sub">PVWMH: {pv_wmh:.2f} cc | DWMH: {dw_wmh:.2f} cc</div>
        </div>
        {tiv_html}
    </div>

    <!-- Axial Multi-Slice Segmentation View -->
    <div class="section-card">
        <div class="section-title">
            <span>🖼️ Lesion Parcellation Overlays</span>
            <span style="font-size: 12px; font-weight: normal; color: var(--text-muted); margin-left: auto;">
                <span style="color: #06b6d4;">■</span> PVWMH (Periventricular) &nbsp;|&nbsp; 
                <span style="color: #f43f5e;">■</span> DWMH (Deep White Matter)
            </span>
        </div>
        <div class="img-wrapper">
            <img src="data:image/png;base64,{b64_multislice}" alt="Axial Lesion Multi-slice Overlay">
        </div>
    </div>

    <!-- Regional Breakdown Tables -->
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 25px;">
        <div class="section-card" style="margin-bottom: 0;">
            <div class="section-title">🏛️ Lobar WMH Distribution</div>
            <table>
                <thead>
                    <tr>
                        <th>Lobe</th>
                        <th>Volume</th>
                        <th>Share</th>
                        <th>Left</th>
                        <th>Right</th>
                    </tr>
                </thead>
                <tbody>
                    {lobar_rows if lobar_rows else "<tr><td colspan='5'>Lobar parcellation not computed.</td></tr>"}
                </tbody>
            </table>
        </div>

        <div class="section-card" style="margin-bottom: 0;">
            <div class="section-title">🩸 Arterial Vascular Distribution</div>
            <table>
                <thead>
                    <tr>
                        <th>Territory</th>
                        <th>Volume</th>
                        <th>Share</th>
                        <th>Left</th>
                        <th>Right</th>
                    </tr>
                </thead>
                <tbody>
                    {arterial_rows if arterial_rows else "<tr><td colspan='5'>Arterial parcellation not computed.</td></tr>"}
                </tbody>
            </table>
        </div>
    </div>

    <!-- Coregistration Check -->
    <div class="section-card">
        <div class="section-title">📐 FLAIR to T1 Alignment & Brain Extraction Check</div>
        <div class="img-wrapper">
            <img src="data:image/png;base64,{b64_coreg}" alt="Coregistration Edge Overlay">
        </div>
        <div style="font-size: 12px; color: var(--text-muted); margin-top: 8px; text-align: center;">
            Green lines depict T1 anatomical brain boundaries mapped onto the FLAIR volume. Precise edge correspondence indicates accurate FLIRT coregistration.
        </div>
    </div>

    <footer>
        WMH-Age Estimation Pipeline &bull; Reference: Huang et al., Age and Ageing (2022) &bull; Algorithmic Engine: PyWMH (FSL + Pure Python LST-LGA)
    </footer>
</div>
</body>
</html>
"""

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Generated visual QC report: {out_html}")
    print(f"Saved summary snapshot    : {out_png}")
    return out_html
