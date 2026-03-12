#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, logging
import numpy as np
import rasterio

INPUT_ENVI = "PRS_L2D_STD_20220609065_219/PRS_L2D_STD_20220609065219_20220609065223_0001/VNIR_SWIR_latlon_219"
LIB_MINERALS_DIR = "Minerals"
OUTPUT_DIR = "outputs"
USGS_ASD_WAVELENGTHS_TXT = "splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt"

WL_MIN = 0.45
WL_MAX = 2.45
DROP_WINDOWS = [(1.35, 1.45), (1.80, 1.95)]

VALID_MEAN_MIN = 0.02
VALID_RNG_MIN = 0.01

SMA_ADD_DARKFLAT = True
DARKFLAT_LEVEL = 0.5
SMA_MAX_ENDMEMBERS = 30

EXPORT_DOMINANT_FRAC = False
SMA_PURITY_THR = 0.75

EXPORT_DOMINANT_SCORE = True
SCORE_P50 = 50
SCORE_P90 = 90
SCORE_PURITY_THR = 0.75

EXPORT_PER_MINERAL_TIFS = True
EXPORT_FRACS_ALL_MULTIBAND = False

USGS_NODATA = -1.23e34
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "processing_log.txt"), encoding="utf-8"),
        logging.StreamHandler()
    ]
)

def _safe_name(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z_\-]+", "_", (s or "").strip())[:120]

def read_wavelengths_from_hdr(hdr_path):
    try:
        txt = open(hdr_path, 'r', encoding='utf-8', errors='ignore').read()
        m = re.search(r"wavelength\s*=\s*\{([^}]*)\}", txt, flags=re.I | re.S)
        if not m:
            return None
        vals = [v.strip() for v in m.group(1).replace("\n", " ").split(",")]
        wls = []
        for v in vals:
            try:
                wls.append(float(v))
            except:
                pass
        wls = np.array(wls, dtype=float)
        if np.nanmedian(wls) > 100:
            wls = wls / 1000.0
        return wls
    except Exception as e:
        logging.warning(f"WL HDR: {e}")
        return None

def mask_good_bands_prisma(wl_um):
    wl = np.array(wl_um, dtype=float)
    good = (wl >= WL_MIN) & (wl <= WL_MAX)
    for a, b in DROP_WINDOWS:
        good &= ~((wl >= a) & (wl <= b))
    return good

def open_prisma(envi_base):
    data_file = envi_base
    if not os.path.exists(data_file):
        if os.path.exists(envi_base + ".dat"):
            data_file = envi_base + ".dat"
        else:
            raise FileNotFoundError(f"File dati non trovato: {envi_base} (o {envi_base}.dat)")
    with rasterio.open(data_file) as ds:
        cube = ds.read()
        transform, crs = ds.transform, ds.crs
    return cube, transform, crs

def save_tif(path, arr, transform, crs, dtype=None):
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError("save_tif expects 2D array")
    if dtype is None:
        dtype = arr.dtype
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=arr.shape[0], width=arr.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        compress="DEFLATE"
    ) as dst:
        dst.write(arr.astype(dtype), 1)

def save_tif_multiband(path, arr3d, transform, crs, dtype=None):
    arr3d = np.asarray(arr3d)
    if arr3d.ndim != 3:
        raise ValueError("save_tif_multiband expects 3D array (bands, rows, cols)")
    if dtype is None:
        dtype = arr3d.dtype
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=arr3d.shape[1], width=arr3d.shape[2],
        count=arr3d.shape[0],
        dtype=dtype,
        crs=crs,
        transform=transform,
        compress="DEFLATE"
    ) as dst:
        dst.write(arr3d.astype(dtype))

def load_usgs_asd_wavelengths(path):
    vals = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                vals.append(float(s))
            except:
                pass
    wl = np.array(vals, dtype=float)
    if wl.size < 1000:
        raise RuntimeError(f"Wavelength ASD troppo corto ({wl.size}). File errato? {path}")
    return wl

def read_usgs_1col_spectrum(txt_path):
    ys = []
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith(("splib", "Version", "Description", "#", ";")):
                continue
            try:
                v = float(s.split()[0])
                ys.append(v)
            except:
                continue
    y = np.array(ys, dtype=np.float64)
    y[np.isclose(y, USGS_NODATA)] = np.nan
    y[(y < 0) & np.isfinite(y)] = 0
    return y

def _scene_basename(envi_base):
    base = os.path.basename(envi_base)
    base = re.sub(r"\.dat$", "", base, flags=re.I)
    base = re.sub(r"\.hdr$", "", base, flags=re.I)
    return base or "prisma_scene"

def _assert_paths():
    if not (os.path.exists(INPUT_ENVI) or os.path.exists(INPUT_ENVI + ".dat")):
        raise FileNotFoundError(f"INPUT_ENVI non trovato: {INPUT_ENVI} (o {INPUT_ENVI}.dat)")
    if not os.path.exists(USGS_ASD_WAVELENGTHS_TXT):
        raise FileNotFoundError(f"USGS_ASD_WAVELENGTHS_TXT non trovato: {USGS_ASD_WAVELENGTHS_TXT}")
    if not os.path.isdir(LIB_MINERALS_DIR):
        raise FileNotFoundError(f"LIB_MINERALS_DIR non trovato: {LIB_MINERALS_DIR}")

def _list_txt_files_flat_or_subfolders(root_dir):
    flat = [
        os.path.join(root_dir, f)
        for f in sorted(os.listdir(root_dir))
        if os.path.isfile(os.path.join(root_dir, f)) and f.lower().endswith(".txt")
    ]
    if flat:
        return [("Minerals", p) for p in flat]

    out = []
    for name in sorted(os.listdir(root_dir)):
        sub = os.path.join(root_dir, name)
        if not os.path.isdir(sub):
            continue
        files = [
            os.path.join(sub, f)
            for f in sorted(os.listdir(sub))
            if os.path.isfile(os.path.join(sub, f)) and f.lower().endswith(".txt")
        ]
        for p in files:
            out.append((name, p))
    return out

def build_library_auto(lib_dir, wl_prisma_good, wl_usgs_asd):
    pairs = _list_txt_files_flat_or_subfolders(lib_dir)
    if not pairs:
        raise RuntimeError(f"Nessun .txt trovato in {lib_dir} (né in sottocartelle di 1° livello).")

    mineral_names = []
    spectra_raw_list = []
    target_wl = wl_prisma_good

    for group_name, p in pairs:
        fn = os.path.basename(p)
        y = read_usgs_1col_spectrum(p)
        if y.size != wl_usgs_asd.size:
            continue
        m = np.isfinite(y)
        if m.sum() < 50:
            continue
        interp = np.interp(target_wl, wl_usgs_asd[m], y[m], left=np.nan, right=np.nan)
        if not np.all(np.isfinite(interp)):
            continue

        interp_raw = interp.astype(np.float32, copy=False)

        name = os.path.splitext(fn)[0]
        name = re.sub(r"^splib07a_", "", name, flags=re.I)
        if group_name != "Minerals":
            mineral_names.append(f"{group_name}:{name}")
        else:
            mineral_names.append(name)

        spectra_raw_list.append(interp_raw)

    if not spectra_raw_list:
        raise RuntimeError("Nessuno spettro valido caricato dalla libreria.")

    lib_raw = np.vstack(spectra_raw_list).astype(np.float32)
    return mineral_names, lib_raw

def _project_to_simplex(v):
    v = np.asarray(v, dtype=np.float64)
    n = v.size
    if n == 0:
        return v.astype(np.float32)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0]
    if rho.size == 0:
        w = np.maximum(v, 0.0)
        s = w.sum()
        return (w / s).astype(np.float32) if s > 0 else np.full((n,), 1.0 / n, dtype=np.float32)
    rho = rho[-1]
    theta = (cssv[rho] - 1.0) / float(rho + 1)
    w = np.maximum(v - theta, 0.0)
    s = w.sum()
    return (w / s).astype(np.float32) if s > 0 else np.full((n,), 1.0 / n, dtype=np.float32)

def main():
    logging.info("=== PRISMA -> SMA (EXPORT PER MINERAL + SCORE DOMINANT) ===")
    _assert_paths()

    cube, transform, crs = open_prisma(INPUT_ENVI)
    n_b, r, c = cube.shape
    cube = cube.astype(np.float32, copy=False)

    hdr = INPUT_ENVI if INPUT_ENVI.lower().endswith(".hdr") else INPUT_ENVI + ".hdr"
    wl = read_wavelengths_from_hdr(hdr)
    if wl is None or wl.size != n_b:
        logging.warning("Wavelength PRISMA mancanti/mismatch: uso linspace 0.4–2.5 (fallback)")
        wl = np.linspace(0.4, 2.5, n_b, dtype=float)

    good = mask_good_bands_prisma(wl)
    gidx = np.where(good)[0]
    wl_good = wl[good]
    logging.info(f"Bande buone PRISMA: {good.sum()}/{n_b}")

    wl_usgs = load_usgs_asd_wavelengths(USGS_ASD_WAVELENGTHS_TXT)
    mineral_names, lib_raw = build_library_auto(LIB_MINERALS_DIR, wl_good, wl_usgs)
    logging.info(f"Spettri caricati da libreria: {len(mineral_names)}")

    if len(mineral_names) > SMA_MAX_ENDMEMBERS:
        logging.warning(f"[SMA] Libreria ha {len(mineral_names)} spettri: cap a {SMA_MAX_ENDMEMBERS} per evitare RAM.")
        mineral_names = mineral_names[:SMA_MAX_ENDMEMBERS]
        lib_raw = lib_raw[:SMA_MAX_ENDMEMBERS, :]

    sma_used = list(mineral_names)
    cols = [lib_raw[i, :].astype(np.float32, copy=False) for i in range(len(sma_used))]

    if SMA_ADD_DARKFLAT:
        sma_used.append("DarkFlat")
        cols.append(np.full((wl_good.size,), float(DARKFLAT_LEVEL), dtype=np.float32))

    if len(cols) < 3:
        raise RuntimeError("Endmembers SMA < 3: aggiungi spettri in libreria (min 3).")

    E = np.stack(cols, axis=1).astype(np.float32)
    m_end = E.shape[1]

    json.dump(
        {"sma_endmembers_used": sma_used},
        open(os.path.join(OUTPUT_DIR, "sma_endmembers_used.json"), "w", encoding="utf-8"),
        ensure_ascii=False, indent=2
    )

    sma_frac = np.zeros((m_end, r, c), np.float32)
    sma_rmse = np.full((r, c), np.nan, np.float32)
    valid_total = 0

    for i in range(r):
        if i % 50 == 0:
            logging.info(f"Riga {i+1}/{r}")

        row_raw = np.transpose(cube[:, i, :], (1, 0)).copy()
        row_raw[~np.isfinite(row_raw)] = 0
        row_raw[row_raw < 0] = 0
        row_g = row_raw[:, gidx]

        mean_ref = np.mean(row_g, axis=1)
        p95 = np.percentile(row_g, 95, axis=1)
        p05 = np.percentile(row_g, 5, axis=1)
        rng_ref = p95 - p05
        valid = (mean_ref >= VALID_MEAN_MIN) & (rng_ref >= VALID_RNG_MIN)

        valid_total += int(np.sum(valid))
        if not np.any(valid):
            continue

        for j in range(c):
            if not valid[j]:
                continue
            p = row_g[j, :].astype(np.float32, copy=False)
            if not np.any(p):
                continue

            a_ls, *_ = np.linalg.lstsq(E, p, rcond=None)
            a = _project_to_simplex(a_ls)
            fit = (E @ a).astype(np.float32)

            sma_frac[:, i, j] = a
            sma_rmse[i, j] = float(np.sqrt(np.mean((fit - p) ** 2)))

    base = _scene_basename(INPUT_ENVI)

    save_tif(os.path.join(OUTPUT_DIR, f"{base}_sma_rmse.tif"), sma_rmse, transform, crs, "float32")

    sel = [(bi, name) for bi, name in enumerate(sma_used) if name.lower().strip() != "darkflat"]
    if not sel:
        raise RuntimeError("Solo DarkFlat in SMA: libreria vuota o cap errato.")

    frac_stack = np.stack([sma_frac[bi, :, :] for (bi, _) in sel], axis=0).astype(np.float32)

    fracs_legend = {int(i + 1): sel[i][1] for i in range(len(sel))}
    json.dump(
        {"note": "Band i (1-based) = fraction of endmember i", "bands": fracs_legend},
        open(os.path.join(OUTPUT_DIR, "sma_fracs_legend.json"), "w", encoding="utf-8"),
        ensure_ascii=False, indent=2
    )

    if EXPORT_FRACS_ALL_MULTIBAND:
        save_tif_multiband(os.path.join(OUTPUT_DIR, f"{base}_sma_fracs_all.tif"), frac_stack, transform, crs, "float32")

    if EXPORT_PER_MINERAL_TIFS:
        for band_i, (bi, name) in enumerate(sel, start=1):
            save_tif(
                os.path.join(OUTPUT_DIR, f"{base}_sma_frac_{band_i:03d}_{_safe_name(name)}.tif"),
                sma_frac[bi, :, :].astype(np.float32),
                transform, crs, "float32"
            )

    maxfrac = np.max(frac_stack, axis=0).astype(np.float32)
    argmax = np.argmax(frac_stack, axis=0).astype(np.int32)
    save_tif(os.path.join(OUTPUT_DIR, f"{base}_sma_maxfrac_selected.tif"), maxfrac, transform, crs, "float32")

    if EXPORT_DOMINANT_FRAC:
        dom = np.zeros((r, c), dtype=np.uint16)
        strong = maxfrac >= float(SMA_PURITY_THR)
        dom[strong] = (argmax[strong] + 1).astype(np.uint16)
        save_tif(
            os.path.join(OUTPUT_DIR, f"{base}_sma_dominant_frac_thr{int(SMA_PURITY_THR*100):02d}.tif"),
            dom, transform, crs, "uint16"
        )

    if EXPORT_DOMINANT_SCORE:
        valid_mask = np.isfinite(sma_rmse)
        stats = {}
        p50s = np.zeros((len(sel),), dtype=np.float32)
        p90s = np.zeros((len(sel),), dtype=np.float32)

        for i_em, (bi, name) in enumerate(sel):
            v = sma_frac[bi, :, :][valid_mask]
            if v.size < 50:
                p50 = float(np.nanmedian(v)) if v.size else 0.0
                p90 = float(np.nanpercentile(v, 90)) if v.size else 1.0
            else:
                p50 = float(np.nanpercentile(v, SCORE_P50))
                p90 = float(np.nanpercentile(v, SCORE_P90))
            if not np.isfinite(p50): p50 = 0.0
            if not np.isfinite(p90): p90 = max(p50 + 1e-6, 1e-3)

            p50s[i_em] = p50
            p90s[i_em] = p90
            stats[name] = {"p50": p50, "p90": p90}

        denom = (p90s - p50s)
        denom = np.where(denom <= 1e-9, 1e-9, denom).astype(np.float32)

        score_stack = np.empty_like(frac_stack, dtype=np.float32)
        for i_em in range(len(sel)):
            score_stack[i_em, :, :] = np.clip((frac_stack[i_em, :, :] - p50s[i_em]) / denom[i_em], 0.0, 1.0)

        maxscore = np.max(score_stack, axis=0).astype(np.float32)
        argmaxs = np.argmax(score_stack, axis=0).astype(np.int32)

        save_tif(os.path.join(OUTPUT_DIR, f"{base}_sma_maxscore_selected.tif"), maxscore, transform, crs, "float32")

        doms = np.zeros((r, c), dtype=np.uint16)
        strong_s = maxscore >= float(SCORE_PURITY_THR)
        doms[strong_s] = (argmaxs[strong_s] + 1).astype(np.uint16)

        save_tif(
            os.path.join(OUTPUT_DIR, f"{base}_sma_dominant_score_thr{int(SCORE_PURITY_THR*100):02d}.tif"),
            doms, transform, crs, "uint16"
        )

        json.dump(
            {
                "note": "Score normalization per endmember: score = clip((frac - p50)/(p90-p50),0..1). Percentiles computed on valid pixels (rmse finite).",
                "SCORE_P50": SCORE_P50,
                "SCORE_P90": SCORE_P90,
                "SCORE_PURITY_THR": float(SCORE_PURITY_THR),
                "per_endmember": stats
            },
            open(os.path.join(OUTPUT_DIR, "sma_score_norm_stats.json"), "w", encoding="utf-8"),
            ensure_ascii=False, indent=2
        )

        json.dump(
            {"dominant_score_codes": fracs_legend},
            open(os.path.join(OUTPUT_DIR, "sma_dominant_score_legend.json"), "w", encoding="utf-8"),
            ensure_ascii=False, indent=2
        )

    json.dump(
        {
            "paths": {
                "input_envi": INPUT_ENVI,
                "lib_minerals_dir": LIB_MINERALS_DIR,
                "output_dir": OUTPUT_DIR,
                "asd_wavelengths_txt": USGS_ASD_WAVELENGTHS_TXT
            },
            "counts": {
                "pixels_total": int(r * c),
                "valid_pixels": int(valid_total),
                "endmembers_exported": int(len(sel)),
            },
            "config": {
                "SMA_ADD_DARKFLAT": bool(SMA_ADD_DARKFLAT),
                "DARKFLAT_LEVEL": float(DARKFLAT_LEVEL),
                "SMA_MAX_ENDMEMBERS": int(SMA_MAX_ENDMEMBERS),
                "VALID_MEAN_MIN": float(VALID_MEAN_MIN),
                "VALID_RNG_MIN": float(VALID_RNG_MIN),
                "EXPORT_PER_MINERAL_TIFS": bool(EXPORT_PER_MINERAL_TIFS),
                "EXPORT_DOMINANT_SCORE": bool(EXPORT_DOMINANT_SCORE),
                "SCORE_P50": int(SCORE_P50),
                "SCORE_P90": int(SCORE_P90),
                "SCORE_PURITY_THR": float(SCORE_PURITY_THR),
            }
        },
        open(os.path.join(OUTPUT_DIR, "report.json"), "w", encoding="utf-8"),
        ensure_ascii=False, indent=2
    )

    logging.info("Fatto. Per-mineral frazioni esportate + dominant_score (Strategia A).")

if __name__ == "__main__":
    main()
