#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PRISMA L2D quick-look generator.

Outputs:
- NDVI
- IOI2
- BD1000
- BD2200
- BD2330
- BD_MgOH_2335
- PUR
- SCORE
- RAW ratios and stretched view products
- RGB VRT composites
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Compression
from scipy.ndimage import uniform_filter

DEFAULT_OUT = "quicklook_output"

DEFAULT_BBLUE = 13
DEFAULT_BGREEN = 22
DEFAULT_BRED = 32

NODATA_VAL = -9999.0


def parse_envi_wavelengths(hdr_path: Path | None):
    if not hdr_path or not hdr_path.exists():
        return None
    txt = hdr_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"wavelength\s*=\s*\{([^}]*)\}", txt, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None

    vals = [v.strip() for v in match.group(1).replace("\n", " ").split(",") if v.strip()]
    wls = []
    for v in vals:
        try:
            wls.append(float(v))
        except Exception:
            continue

    wls = np.array(wls, dtype=float)
    if len(wls) == 0:
        return None

    if np.nanmedian(wls) > 3.0:
        wls = wls / 1000.0

    return wls


def resolve_envi_pair(cube_arg: Path):
    if cube_arg.suffix.lower() == ".hdr":
        hdr_path = cube_arg
        txt = hdr_path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"data\s*file\s*=\s*([^\r\n]+)", txt, flags=re.IGNORECASE)
        if match:
            data_file = match.group(1).strip().strip('{}"\' ')
            candidate = (hdr_path.parent / data_file).resolve()
            if candidate.exists():
                return candidate, hdr_path

        base = hdr_path.with_suffix("")
        if base.exists():
            return base, hdr_path

        for ext in (".dat", ".img", ".bin", ".bsq", ".bil", ".bip"):
            candidate = base.with_suffix(ext)
            if candidate.exists():
                return candidate, hdr_path

        raise FileNotFoundError(f"Cannot find ENVI data file for: {hdr_path}")

    if cube_arg.suffix.lower() in (".tif", ".tiff"):
        data_path = cube_arg
        hdr_path = cube_arg.with_suffix(".hdr")
        return data_path, hdr_path if hdr_path.exists() else None

    data_path = cube_arg
    hdr_path = cube_arg.with_suffix(".hdr")
    return data_path, hdr_path if hdr_path.exists() else None


def nearest_band(wls: np.ndarray, target_um: float) -> int:
    return int(np.abs(wls - target_um).argmin()) + 1


def clamp01(arr):
    return np.clip(arr, 0.0, 1.0)


def stretch01_percentiles(arr, p_lo=2, p_hi=98):
    lo = np.nanpercentile(arr, p_lo)
    hi = np.nanpercentile(arr, p_hi)
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    return clamp01(out).astype(np.float32)


def mean_window(ds, idx_center, w=1):
    idxs = [i for i in range(idx_center - w, idx_center + w + 1) if 1 <= i <= ds.count]
    arrs = [ds.read(i).astype(np.float32) for i in idxs]
    if len(arrs) == 1:
        return arrs[0]
    return np.mean(arrs, axis=0)


def continuum_reflectance(r_left, wl_left, r_right, wl_right, wl_center):
    denom = max(float(wl_right - wl_left), 1e-6)
    t = (wl_center - wl_left) / denom
    return r_left * (1.0 - t) + r_right * t


def band_depth_from_arrays(r_center, r_left, r_right, wl_center, wl_left, wl_right):
    r_cont = continuum_reflectance(r_left, wl_left, r_right, wl_right, wl_center)
    return clamp01(1.0 - (r_center / (r_cont + 1e-6))).astype(np.float32)


def autoscale_reflectance_if_needed(arrs):
    sample = np.concatenate([a.ravel()[::100000] for a in arrs if a.size > 0])
    if sample.size == 0:
        return 1.0

    p99 = np.nanpercentile(sample, 99.5)
    if p99 > 2.0:
        for i in range(len(arrs)):
            arrs[i] = arrs[i] / 10000.0
        return 1.0 / 10000.0

    return 1.0


def safe_band_depth(r_center, r_left, r_right, wl_center, wl_left, wl_right, nodata=NODATA_VAL):
    bad = (
        (r_center <= 0)
        | (r_left <= 0)
        | (r_right <= 0)
        | ~np.isfinite(r_center)
        | ~np.isfinite(r_left)
        | ~np.isfinite(r_right)
    )

    r_cont = continuum_reflectance(r_left, wl_left, r_right, wl_right, wl_center)
    bad |= (~np.isfinite(r_cont)) | (r_cont <= 1e-4)

    bd = np.full_like(r_center, nodata, dtype=np.float32)
    good = ~bad
    bd[good] = clamp01(1.0 - (r_center[good] / (r_cont[good] + 1e-6))).astype(np.float32)
    return bd


def local_variance_nansafe(img, k=3):
    valid = np.isfinite(img).astype(np.float32)
    img0 = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    w = uniform_filter(valid, size=k, mode="nearest")
    m = uniform_filter(img0, size=k, mode="nearest")
    m2 = uniform_filter(img0 * img0, size=k, mode="nearest")

    m = np.where(w > 0, m / np.maximum(w, 1e-6), np.nan)
    m2 = np.where(w > 0, m2 / np.maximum(w, 1e-6), np.nan)

    var = m2 - m * m
    var[var < 0] = 0.0
    return var


def purity_from_variance(var):
    p2 = np.nanpercentile(var, 2)
    p98 = np.nanpercentile(var, 98)

    if not np.isfinite(p2):
        p2 = 0.0
    if not np.isfinite(p98) or p98 <= p2:
        p98 = p2 + 1e-6

    var_norm = clamp01((var - p2) / (p98 - p2))
    return (1.0 - var_norm).astype(np.float32)


def save_tif_like(ref_path, out_path, arr, nodata=NODATA_VAL, tags=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(ref_path) as ref:
        profile = ref.profile.copy()

    profile.update(
        count=1,
        dtype=rasterio.float32,
        compress=Compression.deflate.value,
        predictor=2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        bigtiff="IF_SAFER",
        nodata=nodata,
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)
        if tags:
            dst.update_tags(**tags)

    print(f"[OK] Saved: {out_path}")
    return out_path


def apply_keep_as_nodata(arr, keep_mask, nodata=NODATA_VAL):
    out = arr.astype(np.float32).copy()
    out[~keep_mask] = nodata
    return out


def stretch01_view(arr, nodata=NODATA_VAL, p_lo=2, p_hi=98):
    valid = np.isfinite(arr) & (arr != nodata)
    out = np.full_like(arr, 0, dtype=np.float32)

    if not valid.any():
        return out

    lo = np.nanpercentile(arr[valid], p_lo)
    hi = np.nanpercentile(arr[valid], p_hi)

    if hi > lo:
        out[valid] = clamp01((arr[valid] - lo) / (hi - lo)).astype(np.float32)

    return out


def write_rgb_vrt(ref_tif: Path, out_vrt: Path, r_path: Path, g_path: Path, b_path: Path):
    out_vrt = Path(out_vrt)

    with rasterio.open(ref_tif) as src:
        width, height = src.width, src.height
        transform = src.transform
        crs = src.crs.to_wkt() if src.crs else ""

    r_rel = Path(r_path).name
    g_rel = Path(g_path).name
    b_rel = Path(b_path).name

    vrt = f"""<VRTDataset rasterXSize="{width}" rasterYSize="{height}">
  <SRS>{crs}</SRS>
  <GeoTransform>{",".join(map(str, [transform.c, transform.a, transform.b, transform.f, transform.d, transform.e]))}</GeoTransform>
  <VRTRasterBand dataType="Float32" band="1"><ColorInterp>Red</ColorInterp><SimpleSource><SourceFilename relativeToVRT="1">{r_rel}</SourceFilename><SourceBand>1</SourceBand></SimpleSource></VRTRasterBand>
  <VRTRasterBand dataType="Float32" band="2"><ColorInterp>Green</ColorInterp><SimpleSource><SourceFilename relativeToVRT="1">{g_rel}</SourceFilename><SourceBand>1</SourceBand></SimpleSource></VRTRasterBand>
  <VRTRasterBand dataType="Float32" band="3"><ColorInterp>Blue</ColorInterp><SimpleSource><SourceFilename relativeToVRT="1">{b_rel}</SourceFilename><SourceBand>1</SourceBand></SimpleSource></VRTRasterBand>
</VRTDataset>
"""
    out_vrt.write_text(vrt, encoding="utf-8")
    print(f"[OK] VRT created: {out_vrt}")


def add_formula_tags(base_tags, extra):
    tags = dict(base_tags)
    tags.update(extra)
    return tags


def main():
    parser = argparse.ArgumentParser(description="Generate PRISMA L2D quick-look products.")
    parser.add_argument("--cube", required=True, help="Path to ENVI data file, ENVI .hdr, or GeoTIFF")
    parser.add_argument("--outdir", default=DEFAULT_OUT, help="Output directory")
    parser.add_argument("--ndvi_thr", type=float, default=0.25, help="Keep pixels where NDVI < threshold")
    parser.add_argument("--bblue", type=int, default=DEFAULT_BBLUE)
    parser.add_argument("--bgreen", type=int, default=DEFAULT_BGREEN)
    parser.add_argument("--bred", type=int, default=DEFAULT_BRED)
    parser.add_argument("--bnir", type=int, default=None)
    parser.add_argument("--b210", type=int, default=None)
    parser.add_argument("--b220", type=int, default=None)
    parser.add_argument("--b225", type=int, default=None)
    parser.add_argument("--b233", type=int, default=None)
    parser.add_argument("--b228", type=int, default=None)
    parser.add_argument("--b234", type=int, default=None)
    parser.add_argument("--b240", type=int, default=None)
    args = parser.parse_args()

    cube_arg = Path(args.cube)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cube_suffix = ""
    match = re.search(r"_(\d{3})(?:[/\\]|$)", str(cube_arg))
    if match:
        cube_suffix = f"_{match.group(1)}"

    data_path, hdr_path = resolve_envi_pair(cube_arg)
    print(f"[INFO] DATA: {data_path}")
    print(f"[INFO] HDR: {hdr_path if hdr_path else 'not available'}")

    wls = parse_envi_wavelengths(hdr_path) if hdr_path else None

    with rasterio.open(data_path) as ds:
        if wls is None:
            meta = ds.tags()
            if "wavelength" in meta:
                raw = meta["wavelength"].replace("{", "").replace("}", "")
                try:
                    arr = np.array([float(x) for x in raw.split(",")], dtype=float)
                    if np.nanmedian(arr) > 3.0:
                        arr = arr / 1000.0
                    wls = arr
                except Exception:
                    wls = None

        if wls is not None and len(wls) >= ds.count:
            b_blue = args.bblue
            b_green = args.bgreen
            b_red = args.bred
            b_nir = args.bnir or nearest_band(wls, 0.86)

            b210 = args.b210 or nearest_band(wls, 2.10)
            b220 = args.b220 or nearest_band(wls, 2.20)
            b230 = nearest_band(wls, 2.30)

            b225 = args.b225 or nearest_band(wls, 2.25)
            b233 = args.b233 or nearest_band(wls, 2.33)
            b239 = nearest_band(wls, 2.38)

            b228 = args.b228 or nearest_band(wls, 2.28)
            b234 = args.b234 or nearest_band(wls, 2.34)
            b240 = args.b240 or nearest_band(wls, 2.40)

            picked = {
                "bBlue": b_blue,
                "bGreen": b_green,
                "bRed": b_red,
                "bNIR": b_nir,
                "b210": b210,
                "b220": b220,
                "b230": b230,
                "b225": b225,
                "b233": b233,
                "b239": b239,
                "b088": nearest_band(wls, 0.88),
                "b100": nearest_band(wls, 1.00),
                "b105": nearest_band(wls, 1.05),
                "b228": b228,
                "b234": b234,
                "b240": b240,
            }
            wl_info = {k: float(wls[v - 1]) for k, v in picked.items()}
        else:
            required = ["bblue", "bgreen", "bred", "bnir", "b210", "b220", "b225", "b233"]
            got = {k: getattr(args, k) for k in required}
            missing = [k for k, v in got.items() if v is None]
            if missing:
                raise RuntimeError(f"Wavelengths not available. Please provide these band indices: {missing}")

            picked = {
                "bBlue": args.bblue,
                "bGreen": args.bgreen,
                "bRed": args.bred,
                "bNIR": args.bnir,
                "b210": args.b210,
                "b220": args.b220,
                "b230": args.b220 + 4 if args.b220 else None,
                "b225": args.b225,
                "b233": args.b233,
                "b239": args.b233 + 5 if args.b233 else None,
                "b088": args.bnir - 5 if args.bnir else None,
                "b100": args.bnir + 3 if args.bnir else None,
                "b105": args.bnir + 6 if args.bnir else None,
                "b228": args.b233 - 5 if args.b233 else None,
                "b234": args.b233 + 1 if args.b233 else None,
                "b240": args.b233 + 7 if args.b233 else None,
            }
            wl_info = None

        print(f"[INFO] Bands (1-based): {picked}")
        if wl_info is not None:
            print("[INFO] Wavelengths (µm):")
            print(json.dumps(wl_info, indent=2))

        blue = mean_window(ds, picked["bBlue"], w=1)
        green = mean_window(ds, picked["bGreen"], w=1)
        red = mean_window(ds, picked["bRed"], w=1)
        nir = mean_window(ds, picked["bNIR"], w=1)

        l210 = mean_window(ds, picked["b210"], w=2)
        c220 = mean_window(ds, picked["b220"], w=2)
        r230 = mean_window(ds, picked["b230"], w=2)

        l225 = mean_window(ds, picked["b225"], w=3)
        c233 = mean_window(ds, picked["b233"], w=3)
        r239 = mean_window(ds, picked["b239"], w=3)

        l088 = mean_window(ds, picked["b088"], w=1)
        c100 = mean_window(ds, picked["b100"], w=1)
        r105 = mean_window(ds, picked["b105"], w=1)

        l228 = mean_window(ds, picked["b228"], w=2)
        c234 = mean_window(ds, picked["b234"], w=2)
        r240 = mean_window(ds, picked["b240"], w=2)

    scale_factor = autoscale_reflectance_if_needed(
        [
            blue, green, red, nir,
            l210, c220, r230,
            l225, c233, r239,
            l088, c100, r105,
            l228, c234, r240,
        ]
    )
    if scale_factor != 1.0:
        print(f"[INFO] Reflectance autoscale applied (x{scale_factor:.5f})")

    ndvi = (nir - red) / (nir + red + 1e-6)
    keep_nonveg = ndvi < args.ndvi_thr
    keep_pct = float(np.sum(keep_nonveg)) / keep_nonveg.size * 100.0
    print(f"[INFO] Keep pixels (non-vegetated): {keep_pct:.2f}%")

    ioi2 = (red * red) / (blue * green + 1e-6)

    if wl_info is not None:
        bd1000 = band_depth_from_arrays(
            c100, l088, r105,
            wl_info["b100"], wl_info["b088"], wl_info["b105"]
        )
        bd2200 = band_depth_from_arrays(
            c220, l210, r230,
            wl_info["b220"], wl_info["b210"], wl_info["b230"]
        )
        bd2330 = safe_band_depth(
            c233, l225, r239,
            wl_info["b233"], wl_info["b225"], wl_info["b239"], nodata=NODATA_VAL
        )
        bd_mgoh = safe_band_depth(
            c234, l228, r240,
            wl_info["b234"], wl_info["b228"], wl_info["b240"], nodata=NODATA_VAL
        )
    else:
        bd1000 = clamp01(1.0 - (c100 / (0.5 * (l088 + r105) + 1e-6)))
        bd2200 = clamp01(1.0 - (c220 / (0.5 * (l210 + r230) + 1e-6)))

        rcont_carb = 0.5 * (l225 + r239)
        bad_carb = (c233 <= 0) | (rcont_carb <= 1e-4) | ~np.isfinite(c233) | ~np.isfinite(rcont_carb)
        bd2330 = np.full_like(c233, NODATA_VAL, dtype=np.float32)
        good_carb = ~bad_carb
        bd2330[good_carb] = clamp01(1.0 - (c233[good_carb] / (rcont_carb[good_carb] + 1e-6)))

        rcont_mg = 0.5 * (l228 + r240)
        bad_mg = (c234 <= 0) | (rcont_mg <= 1e-4) | ~np.isfinite(c234) | ~np.isfinite(rcont_mg)
        bd_mgoh = np.full_like(c234, NODATA_VAL, dtype=np.float32)
        good_mg = ~bad_mg
        bd_mgoh[good_mg] = clamp01(1.0 - (c234[good_mg] / (rcont_mg[good_mg] + 1e-6)))

    vnir_mean = (blue + green + red) / 3.0
    var = local_variance_nansafe(vnir_mean, k=3)
    pur = purity_from_variance(var)

    eps = 1e-6
    fe2_raw = (nir / (red + eps)).astype(np.float32)
    ioi_raw = (red / (blue + eps)).astype(np.float32)
    clay_raw = (c220 / (l210 + eps)).astype(np.float32)

    for arr in (bd2330, bd2200, bd_mgoh, ioi2, pur, fe2_raw, ioi_raw, clay_raw):
        arr[~np.isfinite(arr)] = NODATA_VAL

    score = (
        clamp01(0.75 * stretch01_percentiles(bd2200) + 0.25 * stretch01_percentiles(ioi2))
        * stretch01_percentiles(pur)
    )

    meta_common = {
        "source": "PRISMA L2D reflectance (BOA)",
        "ndvi_thr": f"{args.ndvi_thr}",
        "mask": "NoData applied where NDVI >= ndvi_thr",
        "windows": "Clay ±2; Carbonates ±3; Mg-OH ±2; VNIR/Fe ±1",
        "autoscale_reflectance": f"{scale_factor}",
        "keep_pct": f"{keep_pct:.2f}",
    }

    formula_ioi2 = "IOI2 = Red^2 / (Blue * Green)"
    formula_bd1000 = "BD1000 = 1 - R(1.00µm)/Rcont(0.88–1.05µm)"
    formula_bd2200 = "BD2200 = 1 - R(2.20µm)/Rcont(2.10–2.30µm)"
    formula_bd2330 = "BD2330 = 1 - R(2.33µm)/Rcont(2.25–2.38µm)"
    formula_bd_mgoh = "BD_MgOH_2335 = 1 - R(2.34µm)/Rcont(2.28–2.40µm)"
    formula_pur = "PUR = 1 - norm(VAR_3x3, p2–p98)"
    formula_fe_raw = "FE2_raw = NIR / Red"
    formula_ioi_raw = "IOI_raw = Red / Blue"
    formula_clay_raw = "CLAY_raw = R(2.20µm) / R(2.10µm)"

    out_ndvi = out_dir / f"NDVI{cube_suffix}.tif"
    out_mask = out_dir / f"MASK_nonveg_1_keep{cube_suffix}.tif"
    out_i = out_dir / f"IOI2_red2_over_blue_green{cube_suffix}.tif"
    out_bd1000 = out_dir / f"BD1000_fe2{cube_suffix}.tif"
    out_bd2200 = out_dir / f"BD2200_clay{cube_suffix}.tif"
    out_bd2330 = out_dir / f"BD2330_carb{cube_suffix}.tif"
    out_bdmg = out_dir / f"BD_MgOH_2335{cube_suffix}.tif"
    out_pur = out_dir / f"PUR_var3x3{cube_suffix}.tif"
    out_score = out_dir / f"SCORE_quicklook{cube_suffix}.tif"

    out_i_view = out_dir / f"IOI2_red2_over_blue_green_view{cube_suffix}.tif"
    out_bd1000_view = out_dir / f"BD1000_fe2_view{cube_suffix}.tif"
    out_bd2200_view = out_dir / f"BD2200_clay_view{cube_suffix}.tif"
    out_bd2330_view = out_dir / f"BD2330_carb_view{cube_suffix}.tif"
    out_bdmg_view = out_dir / f"BD_MgOH_2335_view{cube_suffix}.tif"
    out_pur_view = out_dir / f"PUR_var3x3_view{cube_suffix}.tif"
    out_score_view = out_dir / f"SCORE_quicklook_view{cube_suffix}.tif"

    out_fe_raw = out_dir / f"FE2_ratio_nir_red{cube_suffix}.tif"
    out_ioi_raw = out_dir / f"IOI_ratio_red_blue{cube_suffix}.tif"
    out_clay_raw = out_dir / f"CLAY_ratio_220_210{cube_suffix}.tif"
    out_fe_raw_view = out_dir / f"FE2_ratio_nir_red_view{cube_suffix}.tif"
    out_ioi_raw_view = out_dir / f"IOI_ratio_red_blue_view{cube_suffix}.tif"
    out_clay_raw_view = out_dir / f"CLAY_ratio_220_210_view{cube_suffix}.tif"

    ndvi_q = ndvi.astype(np.float32)
    mask_u8 = np.where(keep_nonveg, 1, 0).astype(np.uint8)

    ioi2_q = apply_keep_as_nodata(ioi2, keep_nonveg)
    bd1000_q = apply_keep_as_nodata(bd1000, keep_nonveg)
    bd2200_q = apply_keep_as_nodata(bd2200, keep_nonveg)
    bd2330_q = apply_keep_as_nodata(bd2330, keep_nonveg)
    bd_mgoh_q = apply_keep_as_nodata(bd_mgoh, keep_nonveg)
    pur_q = apply_keep_as_nodata(pur, keep_nonveg)
    score_q = apply_keep_as_nodata(score, keep_nonveg)

    fe2_raw_q = apply_keep_as_nodata(fe2_raw, keep_nonveg)
    ioi_raw_q = apply_keep_as_nodata(ioi_raw, keep_nonveg)
    clay_raw_q = apply_keep_as_nodata(clay_raw, keep_nonveg)

    tags_i = add_formula_tags(meta_common, {"index": "IOI2", "formula": formula_ioi2})
    tags_f = add_formula_tags(meta_common, {"index": "BD1000_Fe2", "formula": formula_bd1000})
    tags_c = add_formula_tags(meta_common, {"index": "BD2200_Clay", "formula": formula_bd2200})
    tags_cb = add_formula_tags(meta_common, {"index": "BD2330_Carbonates", "formula": formula_bd2330})
    tags_mg = add_formula_tags(meta_common, {"index": "BD_MgOH_2335", "formula": formula_bd_mgoh})
    tags_p = add_formula_tags(meta_common, {"index": "PUR", "formula": formula_pur})
    tags_s = add_formula_tags(
        meta_common,
        {"index": "SCORE", "formula": "0.75*stretch(BD2200)+0.25*stretch(IOI2) * stretch(PUR)"},
    )
    tags_fr = add_formula_tags(meta_common, {"index": "FE2_raw", "formula": formula_fe_raw})
    tags_ir = add_formula_tags(meta_common, {"index": "IOI_raw", "formula": formula_ioi_raw})
    tags_cr = add_formula_tags(meta_common, {"index": "CLAY_raw", "formula": formula_clay_raw})

    if wl_info is not None:
        tags_i.update({
            "lambda_blue": f'{wl_info["bBlue"]:.6f}',
            "lambda_green": f'{wl_info["bGreen"]:.6f}',
            "lambda_red": f'{wl_info["bRed"]:.6f}',
        })
        tags_f.update({
            "lambda_L": f'{wl_info["b088"]:.6f}',
            "lambda_C": f'{wl_info["b100"]:.6f}',
            "lambda_R": f'{wl_info["b105"]:.6f}',
        })
        tags_c.update({
            "lambda_L": f'{wl_info["b210"]:.6f}',
            "lambda_C": f'{wl_info["b220"]:.6f}',
            "lambda_R": f'{wl_info["b230"]:.6f}',
        })
        tags_cb.update({
            "lambda_L": f'{wl_info["b225"]:.6f}',
            "lambda_C": f'{wl_info["b233"]:.6f}',
            "lambda_R": f'{wl_info["b239"]:.6f}',
        })
        tags_mg.update({
            "lambda_L": f'{wl_info["b228"]:.6f}',
            "lambda_C": f'{wl_info["b234"]:.6f}',
            "lambda_R": f'{wl_info["b240"]:.6f}',
        })

    save_tif_like(
        data_path,
        out_ndvi,
        ndvi_q,
        nodata=NODATA_VAL,
        tags=add_formula_tags(meta_common, {"index": "NDVI", "formula": "(NIR-Red)/(NIR+Red)"}),
    )
    save_tif_like(
        data_path,
        out_mask,
        mask_u8.astype(np.float32),
        nodata=NODATA_VAL,
        tags=add_formula_tags(meta_common, {"mask_desc": "1=keep (non-veg), 0=masked"}),
    )

    save_tif_like(data_path, out_i, ioi2_q, nodata=NODATA_VAL, tags=tags_i)
    save_tif_like(data_path, out_bd1000, bd1000_q, nodata=NODATA_VAL, tags=tags_f)
    save_tif_like(data_path, out_bd2200, bd2200_q, nodata=NODATA_VAL, tags=tags_c)
    save_tif_like(data_path, out_bd2330, bd2330_q, nodata=NODATA_VAL, tags=tags_cb)
    save_tif_like(data_path, out_bdmg, bd_mgoh_q, nodata=NODATA_VAL, tags=tags_mg)
    save_tif_like(data_path, out_pur, pur_q, nodata=NODATA_VAL, tags=tags_p)
    save_tif_like(data_path, out_score, score_q, nodata=NODATA_VAL, tags=tags_s)

    save_tif_like(data_path, out_i_view, apply_keep_as_nodata(stretch01_view(ioi2), keep_nonveg), tags=tags_i)
    save_tif_like(data_path, out_bd1000_view, apply_keep_as_nodata(stretch01_view(bd1000), keep_nonveg), tags=tags_f)
    save_tif_like(data_path, out_bd2200_view, apply_keep_as_nodata(stretch01_view(bd2200), keep_nonveg), tags=tags_c)
    save_tif_like(data_path, out_bd2330_view, apply_keep_as_nodata(stretch01_view(bd2330), keep_nonveg), tags=tags_cb)
    save_tif_like(data_path, out_bdmg_view, apply_keep_as_nodata(stretch01_view(bd_mgoh), keep_nonveg), tags=tags_mg)
    save_tif_like(data_path, out_pur_view, apply_keep_as_nodata(stretch01_view(pur), keep_nonveg), tags=tags_p)
    save_tif_like(data_path, out_score_view, apply_keep_as_nodata(stretch01_view(score), keep_nonveg), tags=tags_s)

    save_tif_like(data_path, out_fe_raw, fe2_raw_q, nodata=NODATA_VAL, tags=tags_fr)
    save_tif_like(data_path, out_ioi_raw, ioi_raw_q, nodata=NODATA_VAL, tags=tags_ir)
    save_tif_like(data_path, out_clay_raw, clay_raw_q, nodata=NODATA_VAL, tags=tags_cr)

    save_tif_like(data_path, out_fe_raw_view, apply_keep_as_nodata(stretch01_view(fe2_raw), keep_nonveg), tags=tags_fr)
    save_tif_like(data_path, out_ioi_raw_view, apply_keep_as_nodata(stretch01_view(ioi_raw), keep_nonveg), tags=tags_ir)
    save_tif_like(data_path, out_clay_raw_view, apply_keep_as_nodata(stretch01_view(clay_raw), keep_nonveg), tags=tags_cr)

    vrt_raw = out_dir / f"explore_raw_RGB{cube_suffix}.vrt"
    write_rgb_vrt(out_fe_raw_view, vrt_raw, r_path=out_fe_raw_view, g_path=out_ioi_raw_view, b_path=out_clay_raw_view)

    vrt_mg = out_dir / f"explore_mgoh_RGB{cube_suffix}.vrt"
    write_rgb_vrt(out_bdmg_view, vrt_mg, r_path=out_bdmg_view, g_path=out_i_view, b_path=out_pur_view)

    print(f"\n[DONE] Output directory: {out_dir}")


if __name__ == "__main__":
    main()
