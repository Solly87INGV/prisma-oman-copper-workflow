#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRISMA L2D quick-look pro (ENVI-friendly):
- Accoppia DATA/HDR (ENVI) o usa GeoTIFF
- Legge wavelengths (µm) da .hdr/tag
- Seleziona bande chiave via nearest wavelength
- Calcola: NDVI, IOI2 (R^2/(B*G)), BD1000 (Fe2+), BD2200 (Clay Al-OH), BD2330 (Carbonati),
  PUR (purezza via var 3x3, NaN-safe), SCORE di supporto (opzionale)
- Calcola anche i RAW ratios (per visual "forte"): FE2_raw=NIR/Red, IOI_raw=Red/Blue, CLAY_raw=220/210
- Applica mask NDVI come NoData (non per moltiplicazione)
- Scrive GeoTIFF quantitativi + "_view" con stretch percentili (NoData-safe)
- Scrive metadati (λ usate, formule, finestre, NDVI_thr, % keep)
- Genera VRT RGB "explore_raw_RGB.vrt" (R=FE2_raw_view, G=IOI_raw_view, B=CLAY_raw_view)

AGGIUNTO:
- BD_MgOH_2335 (serpentino/clorite) con centro ~2.34 µm e spalle 2.28–2.40 µm
- VRT RGB "explore_mgoh_RGB.vrt" (R=BD_MgOH_view, G=IOI2_view, B=PUR_view)

Autore: (tuo progetto)
"""

import argparse, re, json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Compression
from scipy.ndimage import uniform_filter

DEFAULT_CUBE = r"E:\INGV\1_Human Mobility\PRISMA\PRS_L2D_STD_20220609065_219\PRS_L2D_STD_20220609065219_20220609065223_0001\VNIR_SWIR_latlon_219"
DEFAULT_OUT  = r"E:\INGV\1_Human Mobility\PRISMA\PRS_L2D_STD_20220609065_219\Quicklook_219"

# Bande VNIR note (1-based) per coerenza visiva
DEFAULT_BBLUE, DEFAULT_BGREEN, DEFAULT_BRED = 13, 22, 32

NODATA_VAL = -9999.0

# ------------------------- Utility ENVI / wavelengths -------------------------

def parse_envi_wavelengths(hdr_path: Path|None):
    if not hdr_path or not hdr_path.exists():
        return None
    txt = hdr_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"wavelength\s*=\s*\{([^}]*)\}", txt, flags=re.IGNORECASE|re.DOTALL)
    if not m:
        return None
    vals = [v.strip() for v in m.group(1).replace("\n"," ").split(",") if v.strip()]
    wls = []
    for v in vals:
        try:
            wls.append(float(v))
        except Exception:
            pass
    wls = np.array(wls, dtype=float)
    if len(wls) == 0:
        return None
    # se in nm, porta a µm
    if np.nanmedian(wls) > 3.0:
        wls = wls / 1000.0
    return wls  # µm

def resolve_envi_pair(cube_arg: Path):
    """
    Ritorna (data_path, hdr_path).
    - Se .hdr -> trova data file
    - Se senza estensione -> tratta come DATA, cerca .hdr accanto
    - Se .tif/.tiff -> usa opzionale .hdr accanto
    """
    if cube_arg.suffix.lower() == ".hdr":
        hdr_path = cube_arg
        txt = hdr_path.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"data\s*file\s*=\s*([^\r\n]+)", txt, flags=re.IGNORECASE)
        if m:
            df = m.group(1).strip().strip('{}"\' ')
            cand = (hdr_path.parent / df).resolve()
            if cand.exists():
                return cand, hdr_path
        base = hdr_path.with_suffix("")
        if base.exists():
            return base, hdr_path
        for ext in (".dat",".img",".bin",".bsq",".bil",".bip"):
            cand = base.with_suffix(ext)
            if cand.exists():
                return cand, hdr_path
        raise FileNotFoundError(f"Non trovo il file dati ENVI per: {hdr_path}")

    elif cube_arg.suffix.lower() in (".tif", ".tiff"):
        data_path = cube_arg
        hdr_path  = cube_arg.with_suffix(".hdr")
        if not hdr_path.exists():
            hdr_path = None
        return data_path, hdr_path

    else:
        data_path = cube_arg
        hdr_path  = cube_arg.with_suffix(".hdr")
        if not hdr_path.exists():
            hdr_path = None
        return data_path, hdr_path

def nearest_band(wls: np.ndarray, target_um: float) -> int:
    idx = int(np.abs(wls - target_um).argmin())
    return idx + 1  # 1-based per rasterio

# ------------------------------ Helper numerici ------------------------------

def clamp01(a):
    return np.clip(a, 0.0, 1.0)

def stretch01_percentiles(x, p_lo=2, p_hi=98):
    lo = np.nanpercentile(x, p_lo)
    hi = np.nanpercentile(x, p_hi)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return clamp01(y).astype(np.float32)

def mean_window(ds, idx_center, w=1):
    """Media bande in finestra ±w (1-based)."""
    idxs = [i for i in range(idx_center - w, idx_center + w + 1) if 1 <= i <= ds.count]
    arrs = [ds.read(i).astype(np.float32) for i in idxs]
    if len(arrs) == 1:
        return arrs[0]
    return np.mean(arrs, axis=0)

def continuum_reflectance(RL, wlL, RR, wlR, wlC):
    """R_cont al centro via interpolazione lineare delle spalle."""
    denom = max(float(wlR - wlL), 1e-6)
    t = (wlC - wlL) / denom
    return RL * (1.0 - t) + RR * t

def band_depth_from_arrays(RC, RL, RR, wlC, wlL, wlR):
    """Band Depth = 1 - R(center)/R_cont(center)."""
    Rcont = continuum_reflectance(RL, wlL, RR, wlR, wlC)
    return clamp01(1.0 - (RC / (Rcont + 1e-6))).astype(np.float32)

def autoscale_reflectance_if_needed(arrs):
    """
    Se i dati paiono in 0–10000, scala a 0–1 (in-place like).
    Restituisce fattore scala usato (1.0 o 1/10000.0).
    """
    sample = np.concatenate([a.ravel()[::100000] for a in arrs if a.size > 0])
    if sample.size == 0:
        return 1.0
    p99 = np.nanpercentile(sample, 99.5)
    if p99 > 2.0:  # riflettanza non dovrebbe superare 1
        for i in range(len(arrs)):
            arrs[i] = arrs[i] / 10000.0
        return 1.0/10000.0
    return 1.0

# --------- Robustezza BD (carbonati) e PUR (NaN-safe, normalizzato) ----------

def safe_band_depth(RC, RL, RR, wlC, wlL, wlR, nodata=NODATA_VAL):
    """Band depth con guard-rail su spalle/continuum."""
    bad = (RC <= 0) | (RL <= 0) | (RR <= 0) | ~np.isfinite(RC) | ~np.isfinite(RL) | ~np.isfinite(RR)
    Rcont = continuum_reflectance(RL, wlL, RR, wlR, wlC)
    bad |= (~np.isfinite(Rcont)) | (Rcont <= 1e-4)
    BD = np.full_like(RC, nodata, dtype=np.float32)
    good = ~bad
    BD[good] = clamp01(1.0 - (RC[good] / (Rcont[good] + 1e-6))).astype(np.float32)
    return BD

def local_variance_nansafe(img, k=3):
    """Varianza locale robusta a NoData/NaN."""
    valid = np.isfinite(img).astype(np.float32)
    img0  = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    w  = uniform_filter(valid, size=k, mode="nearest")
    m  = uniform_filter(img0,  size=k, mode="nearest")
    m2 = uniform_filter(img0*img0, size=k, mode="nearest")
    m  = np.where(w>0, m/np.maximum(w,1e-6), np.nan)
    m2 = np.where(w>0, m2/np.maximum(w,1e-6), np.nan)
    var = m2 - m*m
    var[var < 0] = 0.0
    return var

def purity_from_variance(var):
    """PUR = 1 - VAR normalizzata sui percentili 2–98 della scena."""
    p2  = np.nanpercentile(var, 2)
    p98 = np.nanpercentile(var, 98)
    if not np.isfinite(p2):  p2 = 0.0
    if not np.isfinite(p98) or p98 <= p2: p98 = p2 + 1e-6
    varn = clamp01((var - p2) / (p98 - p2))
    return (1.0 - varn).astype(np.float32)

# ------------------------------ I/O utilities --------------------------------

def save_tif_like(ref_path, out_path, arr, nodata=NODATA_VAL, tags=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(ref_path) as ref:
        prof = ref.profile.copy()
    prof.update(
        count=1,
        dtype=rasterio.float32,
        compress=Compression.deflate.value,
        predictor=2,  # migliore per float
        tiled=True,
        blockxsize=256,
        blockysize=256,
        bigtiff="IF_SAFER",
        nodata=nodata
    )
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(arr.astype(np.float32), 1)
        if tags:
            dst.update_tags(**tags)
    print("[OK] Salvato:", out_path)
    return out_path

def apply_keep_as_nodata(arr, keep_mask, nodata=NODATA_VAL):
    out = arr.astype(np.float32).copy()
    out[~keep_mask] = nodata
    return out

# View stretch che IGNORA i NoData (per evitare il "tutto a 1")
def stretch01_view(x, nodata=NODATA_VAL, p_lo=2, p_hi=98):
    m = np.isfinite(x) & (x != nodata)
    out = np.full_like(x, 0, dtype=np.float32)
    if not m.any():
        return out
    lo = np.nanpercentile(x[m], p_lo)
    hi = np.nanpercentile(x[m], p_hi)
    if hi > lo:
        out[m] = clamp01((x[m] - lo) / (hi - lo)).astype(np.float32)
    return out

def write_rgb_vrt(ref_tif: Path, out_vrt: Path, r_path: Path, g_path: Path, b_path: Path):
    """Crea un VRT RGB usando dimensioni/proiezione del ref_tif."""
    out_vrt = Path(out_vrt)
    with rasterio.open(ref_tif) as src:
        w, h = src.width, src.height
        transform = src.transform
        crs = src.crs.to_wkt() if src.crs else ""
    # percorsi relativi per portabilità
    r_rel = Path(r_path).name
    g_rel = Path(g_path).name
    b_rel = Path(b_path).name
    vrt = f"""<VRTDataset rasterXSize="{w}" rasterYSize="{h}">
  <SRS>{crs}</SRS>
  <GeoTransform>{",".join(map(str,[transform.c, transform.a, transform.b, transform.f, transform.d, transform.e]))}</GeoTransform>
  <VRTRasterBand dataType="Float32" band="1"><ColorInterp>Red</ColorInterp><SimpleSource><SourceFilename relativeToVRT="1">{r_rel}</SourceFilename><SourceBand>1</SourceBand></SimpleSource></VRTRasterBand>
  <VRTRasterBand dataType="Float32" band="2"><ColorInterp>Green</ColorInterp><SimpleSource><SourceFilename relativeToVRT="1">{g_rel}</SourceFilename><SourceBand>1</SourceBand></SimpleSource></VRTRasterBand>
  <VRTRasterBand dataType="Float32" band="3"><ColorInterp>Blue</ColorInterp><SimpleSource><SourceFilename relativeToVRT="1">{b_rel}</SourceFilename><SourceBand>1</SourceBand></SimpleSource></VRTRasterBand>
</VRTDataset>
"""
    out_vrt.write_text(vrt, encoding="utf-8")
    print("[OK] VRT creato:", out_vrt)

# ----------------------------------- Main ------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube",   default=DEFAULT_CUBE, help="Path al DATA ENVI (senza est.) o al .hdr o a un GeoTIFF")
    ap.add_argument("--outdir", default=DEFAULT_OUT,  help="Cartella output")
    ap.add_argument("--ndvi_thr", type=float, default=0.25, help="Soglia per NON vegetato (keep: NDVI<thr)")
    # override bande VNIR 1-based per coerenza visiva
    ap.add_argument("--bblue",  type=int, default=DEFAULT_BBLUE)
    ap.add_argument("--bgreen", type=int, default=DEFAULT_BGREEN)
    ap.add_argument("--bred",   type=int, default=DEFAULT_BRED)
    # opzionali override 1-based
    ap.add_argument("--bnir",   type=int, default=None)
    ap.add_argument("--b210",   type=int, default=None)  # ~2.10 µm (Clay L)
    ap.add_argument("--b220",   type=int, default=None)  # ~2.20 µm (Clay C)
    ap.add_argument("--b225",   type=int, default=None)  # ~2.25 µm (Carb L)
    ap.add_argument("--b233",   type=int, default=None)  # ~2.33 µm (Carb C)
    # opzionali Mg-OH (se vuoi forzarli manualmente)
    ap.add_argument("--b228",   type=int, default=None)  # ~2.28 µm (Mg-OH L)
    ap.add_argument("--b234",   type=int, default=None)  # ~2.34 µm (Mg-OH C)
    ap.add_argument("--b240",   type=int, default=None)  # ~2.40 µm (Mg-OH R)
    args = ap.parse_args()

    cube_arg = Path(args.cube)
    out_dir  = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Estrai le ultime 3 cifre dal percorso del cubo per il suffisso
    cube_suffix = ""
    cube_str = str(cube_arg)
    # Cerca pattern come _904, _909, ecc. alla fine del percorso
    match = re.search(r'_(\d{3})(?:[/\\]|$)', cube_str)
    if match:
        cube_suffix = f"_{match.group(1)}"
    print(f"[INFO] Suffisso cubo: '{cube_suffix}'")

    # Risolvi data/hdr
    data_path, hdr_path = resolve_envi_pair(cube_arg)
    print(f"[INFO] DATA: {data_path}")
    print(f"[INFO] HDR : {hdr_path if hdr_path else 'non disponibile'}")

    # Leggi wavelengths dall'header o dai tag
    wls = parse_envi_wavelengths(hdr_path) if hdr_path else None
    with rasterio.open(data_path) as ds:
        if wls is None:
            meta = ds.tags()
            if "wavelength" in meta:
                raw = meta["wavelength"].replace("{","").replace("}","")
                try:
                    arr = np.array([float(x) for x in raw.split(",")], dtype=float)
                    if np.nanmedian(arr) > 3.0: arr = arr/1000.0
                    wls = arr
                except Exception:
                    wls = None

        # Determina indici bande
        if wls is not None and len(wls) >= ds.count:
            bBlue, bGreen, bRed = args.bblue, args.bgreen, args.bred
            bNIR = args.bnir or nearest_band(wls, 0.86)

            # Finestre diagnostiche (Clay R=2.30; Carb R=2.38)
            b210 = args.b210 or nearest_band(wls, 2.10)   # Clay L
            b220 = args.b220 or nearest_band(wls, 2.20)   # Clay C
            b230 =              nearest_band(wls, 2.30)   # Clay R

            b225 = args.b225 or nearest_band(wls, 2.25)   # Carb L
            b233 = args.b233 or nearest_band(wls, 2.33)   # Carb C
            b239 =              nearest_band(wls, 2.38)   # Carb R

            # Mg-OH (serpentino/clorite) ~2.34 µm (L=2.28; C=2.34; R=2.40)
            b228 = args.b228 or nearest_band(wls, 2.28)
            b234 = args.b234 or nearest_band(wls, 2.34)
            b240 = args.b240 or nearest_band(wls, 2.40)

            picked = dict(
                bBlue=bBlue, bGreen=bGreen, bRed=bRed, bNIR=bNIR,
                b210=b210, b220=b220, b230=b230,
                b225=b225, b233=b233, b239=b239,
                b088=nearest_band(wls, 0.88),
                b100=nearest_band(wls, 1.00),
                b105=nearest_band(wls, 1.05),
                b228=b228, b234=b234, b240=b240
            )
            wl_info = {k: float(wls[v-1]) for k,v in picked.items()}
        else:
            need = ["bblue","bgreen","bred","bnir","b210","b220","b225","b233"]
            got  = {k:getattr(args,k) for k in need}
            if any(v is None for v in got.values()):
                missing = [k for k,v in got.items() if v is None]
                raise RuntimeError(f"Wavelengths non disponibili. Passa questi indici: {missing}")
            # fallback approssimati per bande derivate (senza wls)
            picked = dict(
                bBlue=args.bblue, bGreen=args.bgreen, bRed=args.bred, bNIR=args.bnir,
                b210=args.b210, b220=args.b220, b230=(args.b220+4 if args.b220 else None),
                b225=args.b225, b233=args.b233, b239=(args.b233+5 if args.b233 else None),
                b088=(args.bnir-5 if args.bnir else None),
                b100=(args.bnir+3 if args.bnir else None),
                b105=(args.bnir+6 if args.bnir else None),
                b228=(args.b233-5 if args.b233 else None),   # ~2.28 ≈ C233-5
                b234=(args.b233+1 if args.b233 else None),   # ~2.34 ≈ C233+1
                b240=(args.b233+7 if args.b233 else None)    # ~2.40 ≈ C233+7
            )
            wl_info = None

        print("[INFO] Bande (1-based):", picked)
        if wl_info: print("[INFO] λ (µm):", json.dumps(wl_info, indent=2))

        # --- Lettura bande con medie su finestre ---
        Blue  = mean_window(ds, picked["bBlue"],  w=1)
        Green = mean_window(ds, picked["bGreen"], w=1)
        Red   = mean_window(ds, picked["bRed"],   w=1)
        NIR   = mean_window(ds, picked["bNIR"],   w=1)

        # Clay windows (w=2)
        L210 = mean_window(ds, picked["b210"], w=2)
        C220 = mean_window(ds, picked["b220"], w=2)
        R230 = mean_window(ds, picked["b230"], w=2)

        # Carbonates windows (w=3, con R≈2.38)
        L225 = mean_window(ds, picked["b225"], w=3)
        C233 = mean_window(ds, picked["b233"], w=3)
        R239 = mean_window(ds, picked["b239"], w=3)

        # Fe BD1000 windows (w=1)
        L088 = mean_window(ds, picked["b088"], w=1)
        C100 = mean_window(ds, picked["b100"], w=1)
        R105 = mean_window(ds, picked["b105"], w=1)

        # Mg-OH windows (w=2)
        L228 = mean_window(ds, picked["b228"], w=2)
        C234 = mean_window(ds, picked["b234"], w=2)
        R240 = mean_window(ds, picked["b240"], w=2)

    # Autoscale riflettanza se necessari (0–10000 -> 0–1)
    _scale = autoscale_reflectance_if_needed(
        [Blue, Green, Red, NIR,
         L210, C220, R230,
         L225, C233, R239,
         L088, C100, R105,
         L228, C234, R240]
    )
    if _scale != 1.0:
        print(f"[INFO] Riflettanza autoscala applicata (x{_scale:.5f})")

    # -------------------------- Indici & maschere -----------------------------

    ndvi = (NIR - Red) / (NIR + Red + 1e-6)
    keep_nonveg = ndvi < args.ndvi_thr  # True = mantieni (non vegetato)
    keep_pct = float(np.sum(keep_nonveg)) / keep_nonveg.size * 100.0
    print(f"[INFO] % pixel KEEP (non-veg): {keep_pct:.2f}%")

    # IOI2
    IOI2 = (Red*Red) / (Blue*Green + 1e-6)

    # BD1000 (Fe2+)
    if wl_info is not None:
        BD1000 = band_depth_from_arrays(
            C100, L088, R105,
            wl_info["b100"], wl_info["b088"], wl_info["b105"]
        )
    else:
        BD1000 = clamp01(1.0 - (C100 / (0.5*(L088+R105) + 1e-6)))

    # BD2200 (Clay, Al-OH) con spalla a 2.30 µm
    if wl_info is not None:
        BD2200 = band_depth_from_arrays(
            C220, L210, R230,
            wl_info["b220"], wl_info["b210"], wl_info["b230"]
        )
    else:
        BD2200 = clamp01(1.0 - (C220 / (0.5*(L210+R230) + 1e-6)))

    # BD2330 (Carbonati) robusto (R≈2.38, w=3) con guard-rail
    if wl_info is not None:
        BD2330 = safe_band_depth(
            C233, L225, R239,
            wl_info["b233"], wl_info["b225"], wl_info["b239"], nodata=NODATA_VAL
        )
    else:
        Rcont = 0.5*(L225 + R239)
        bad = (C233<=0) | (Rcont<=1e-4) | ~np.isfinite(C233) | ~np.isfinite(Rcont)
        BD2330 = np.full_like(C233, NODATA_VAL, dtype=np.float32)
        good = ~bad
        BD2330[good] = clamp01(1.0 - (C233[good] / (Rcont[good] + 1e-6)))

    # BD_MgOH (Serpentino/Clorite) ~2.34 µm con spalle 2.28–2.40 (guard-rail)
    if wl_info is not None:
        BD_MgOH = safe_band_depth(
            C234, L228, R240,
            wl_info["b234"], wl_info["b228"], wl_info["b240"], nodata=NODATA_VAL
        )
    else:
        Rcont_mg = 0.5*(L228 + R240)
        bad = (C234<=0) | (Rcont_mg<=1e-4) | ~np.isfinite(C234) | ~np.isfinite(Rcont_mg)
        BD_MgOH = np.full_like(C234, NODATA_VAL, dtype=np.float32)
        good = ~bad
        BD_MgOH[good] = clamp01(1.0 - (C234[good] / (Rcont_mg[good] + 1e-6)))

    # Purezza spettrale (NaN-safe)
    VNIRmean = (Blue + Green + Red) / 3.0
    VAR = local_variance_nansafe(VNIRmean, k=3)
    PUR = purity_from_variance(VAR)

    # ---- RAW RATIOS (per visual "forte") ----
    eps = 1e-6
    FE2_RAW   = (NIR / (Red + eps)).astype(np.float32)     # NIR/Red
    IOI_RAW   = (Red / (Blue + eps)).astype(np.float32)    # Red/Blue
    CLAY_RAW  = (C220 / (L210 + eps)).astype(np.float32)   # 2.20 / 2.10

    # DEBUG: statistiche rapide prima della mask
    def _stat(name, a):
        finite = np.isfinite(a)
        if finite.any():
            print(f"[STAT] {name}: min={np.nanmin(a):.6f} max={np.nanmax(a):.6f} NaN%={(~finite).mean()*100:.2f}%")
        else:
            print(f"[STAT] {name}: all NaN")
    for nm, arr in [("BD2200", BD2200), ("BD2330", BD2330), ("BD_MgOH", BD_MgOH),
                    ("IOI2", IOI2), ("PUR", PUR),
                    ("FE2_RAW", FE2_RAW), ("IOI_RAW", IOI_RAW), ("CLAY_RAW", CLAY_RAW)]:
        _stat(nm, arr)

    # sostituisci NaN con NODATA prima della mask
    for a in (BD2330, BD2200, BD_MgOH, IOI2, PUR, FE2_RAW, IOI_RAW, CLAY_RAW):
        a[~np.isfinite(a)] = NODATA_VAL

    # SCORE (0.75 clay, 0.25 IOI2), pesato da PUR  [lasciato com'è per compatibilità]
    SCORE = clamp01(0.75*stretch01_percentiles(BD2200) + 0.25*stretch01_percentiles(IOI2)) * stretch01_percentiles(PUR)

    # ----------------------------- Salvataggi --------------------------------

    meta_common = {
        "source": "PRISMA L2D reflectance (BOA)",
        "ndvi_thr": f"{args.ndvi_thr}",
        "mask": "NoData applied where NDVI >= ndvi_thr",
        "windows": "Clay ±2; Carbonates ±3; Mg-OH ±2; VNIR/Fe ±1",
        "autoscale_reflectance": f"{_scale}",
        "keep_pct": f"{keep_pct:.2f}"
    }

    def add_formula_tags(base_tags, extra):
        t = dict(base_tags); t.update(extra); return t

    formula_IOI2    = "IOI2 = Red^2 / (Blue * Green)"
    formula_BD1000  = "BD1000 = 1 - R(1.00µm)/Rcont(0.88–1.05µm)"
    formula_BD2200  = "BD2200 = 1 - R(2.20µm)/Rcont(2.10–2.30µm)"
    formula_BD2330  = "BD2330 = 1 - R(2.33µm)/Rcont(2.25–2.38µm)"
    formula_BD_MgOH = "BD_MgOH_2335 = 1 - R(2.34µm)/Rcont(2.28–2.40µm)"
    formula_PUR     = "PUR = 1 - norm(VAR_3x3, p2–p98)"
    formula_FEraw   = "FE2_raw = NIR / Red"
    formula_IOIraw  = "IOI_raw = Red / Blue"
    formula_CLAYraw = "CLAY_raw = R(2.20µm) / R(2.10µm)"

    # File naming (indici "scientifici") - AGGIUNTO SUFFISSO
    out_dir = Path(out_dir)
    out_ndvi      = out_dir / f"NDVI{cube_suffix}.tif"
    out_mask      = out_dir / f"MASK_nonveg_1_keep{cube_suffix}.tif"
    out_i         = out_dir / f"IOI2_red2_over_blue_green{cube_suffix}.tif"
    out_bd1000    = out_dir / f"BD1000_fe2{cube_suffix}.tif"
    out_bd2200    = out_dir / f"BD2200_clay{cube_suffix}.tif"
    out_bd2330    = out_dir / f"BD2330_carb{cube_suffix}.tif"
    out_bdmg      = out_dir / f"BD_MgOH_2335{cube_suffix}.tif"
    out_pur       = out_dir / f"PUR_var3x3{cube_suffix}.tif"
    out_score     = out_dir / f"SCORE_quicklook{cube_suffix}.tif"

    # View variants - AGGIUNTO SUFFISSO
    out_i_view      = out_dir / f"IOI2_red2_over_blue_green_view{cube_suffix}.tif"
    out_bd1000_view = out_dir / f"BD1000_fe2_view{cube_suffix}.tif"
    out_bd2200_view = out_dir / f"BD2200_clay_view{cube_suffix}.tif"
    out_bd2330_view = out_dir / f"BD2330_carb_view{cube_suffix}.tif"
    out_bdmg_view   = out_dir / f"BD_MgOH_2335_view{cube_suffix}.tif"
    out_pur_view    = out_dir / f"PUR_var3x3_view{cube_suffix}.tif"
    out_score_view  = out_dir / f"SCORE_quicklook_view{cube_suffix}.tif"

    # RAW outputs - AGGIUNTO SUFFISSO
    out_fe_raw        = out_dir / f"FE2_ratio_nir_red{cube_suffix}.tif"
    out_ioi_raw       = out_dir / f"IOI_ratio_red_blue{cube_suffix}.tif"
    out_clay_raw      = out_dir / f"CLAY_ratio_220_210{cube_suffix}.tif"
    out_fe_raw_view   = out_dir / f"FE2_ratio_nir_red_view{cube_suffix}.tif"
    out_ioi_raw_view  = out_dir / f"IOI_ratio_red_blue_view{cube_suffix}.tif"
    out_clay_raw_view = out_dir / f"CLAY_ratio_220_210_view{cube_suffix}.tif"

    # Applica mask come NoData
    ndvi_q = ndvi.astype(np.float32)
    mask_u8 = np.where(keep_nonveg, 1, 0).astype(np.uint8)

    IOI2_q    = apply_keep_as_nodata(IOI2,    keep_nonveg)
    BD1000_q  = apply_keep_as_nodata(BD1000, keep_nonveg)
    BD2200_q  = apply_keep_as_nodata(BD2200, keep_nonveg)
    BD2330_q  = apply_keep_as_nodata(BD2330, keep_nonveg)
    BD_MgOH_q = apply_keep_as_nodata(BD_MgOH, keep_nonveg)
    PUR_q     = apply_keep_as_nodata(PUR,     keep_nonveg)
    SCORE_q   = apply_keep_as_nodata(SCORE,   keep_nonveg)

    FE2_RAW_q   = apply_keep_as_nodata(FE2_RAW, keep_nonveg)
    IOI_RAW_q   = apply_keep_as_nodata(IOI_RAW, keep_nonveg)
    CLAY_RAW_q  = apply_keep_as_nodata(CLAY_RAW, keep_nonveg)

    # Tag comuni + formule
    tags_i  = add_formula_tags(meta_common, {"index":"IOI2","formula":formula_IOI2})
    tags_f  = add_formula_tags(meta_common, {"index":"BD1000_Fe2","formula":formula_BD1000})
    tags_c  = add_formula_tags(meta_common, {"index":"BD2200_Clay","formula":formula_BD2200})
    tags_cb = add_formula_tags(meta_common, {"index":"BD2330_Carbonates","formula":formula_BD2330})
    tags_mg = add_formula_tags(meta_common, {"index":"BD_MgOH_2335","formula":formula_BD_MgOH})
    tags_p  = add_formula_tags(meta_common, {"index":"PUR","formula":formula_PUR})
    tags_s  = add_formula_tags(meta_common, {"index":"SCORE","formula":"0.75*stretch(BD2200)+0.25*stretch(IOI2) * stretch(PUR)"})
    tags_fr = add_formula_tags(meta_common, {"index":"FE2_raw","formula":formula_FEraw})
    tags_ir = add_formula_tags(meta_common, {"index":"IOI_raw","formula":formula_IOIraw})
    tags_cr = add_formula_tags(meta_common, {"index":"CLAY_raw","formula":formula_CLAYraw})

    if wl_info is not None:
        tags_i.update({
            "lambda_blue": f'{wl_info["bBlue"]:.6f}',
            "lambda_green": f'{wl_info["bGreen"]:.6f}',
            "lambda_red": f'{wl_info["bRed"]:.6f}',
        })
        tags_f.update({
            "lambda_L": f'{wl_info["b088"]:.6f}', "lambda_C": f'{wl_info["b100"]:.6f}', "lambda_R": f'{wl_info["b105"]:.6f}'
        })
        tags_c.update({
            "lambda_L": f'{wl_info["b210"]:.6f}', "lambda_C": f'{wl_info["b220"]:.6f}', "lambda_R": f'{wl_info["b230"]:.6f}'
        })
        tags_cb.update({
            "lambda_L": f'{wl_info["b225"]:.6f}', "lambda_C": f'{wl_info["b233"]:.6f}', "lambda_R": f'{wl_info["b239"]:.6f}'
        })
        tags_mg.update({
            "lambda_L": f'{wl_info["b228"]:.6f}', "lambda_C": f'{wl_info["b234"]:.6f}', "lambda_R": f'{wl_info["b240"]:.6f}'
        })

    # Salva quantitativi (scientifici)
    save_tif_like(data_path, out_ndvi, ndvi_q, nodata=NODATA_VAL,
                  tags=add_formula_tags(meta_common, {"index":"NDVI","formula":"(NIR-Red)/(NIR+Red)"}))
    save_tif_like(data_path, out_mask, mask_u8.astype(np.float32), nodata=NODATA_VAL,
                  tags=add_formula_tags(meta_common, {"mask_desc":"1=keep (non-veg), 0=masked"}))

    save_tif_like(data_path, out_i,       IOI2_q,   nodata=NODATA_VAL, tags=tags_i)
    save_tif_like(data_path, out_bd1000,  BD1000_q, nodata=NODATA_VAL, tags=tags_f)
    save_tif_like(data_path, out_bd2200,  BD2200_q, nodata=NODATA_VAL, tags=tags_c)
    save_tif_like(data_path, out_bd2330,  BD2330_q, nodata=NODATA_VAL, tags=tags_cb)
    save_tif_like(data_path, out_bdmg,    BD_MgOH_q, nodata=NODATA_VAL, tags=tags_mg)
    save_tif_like(data_path, out_pur,     PUR_q,    nodata=NODATA_VAL, tags=tags_p)
    save_tif_like(data_path, out_score,   SCORE_q,  nodata=NODATA_VAL, tags=tags_s)

    # Salva versioni VIEW (stretch percentile 2–98) — NoData-safe
    save_tif_like(data_path, out_i_view,       apply_keep_as_nodata(stretch01_view(IOI2),     keep_nonveg), tags=tags_i)
    save_tif_like(data_path, out_bd1000_view,  apply_keep_as_nodata(stretch01_view(BD1000),   keep_nonveg), tags=tags_f)
    save_tif_like(data_path, out_bd2200_view,  apply_keep_as_nodata(stretch01_view(BD2200),   keep_nonveg), tags=tags_c)
    save_tif_like(data_path, out_bd2330_view,  apply_keep_as_nodata(stretch01_view(BD2330),   keep_nonveg), tags=tags_cb)
    save_tif_like(data_path, out_bdmg_view,    apply_keep_as_nodata(stretch01_view(BD_MgOH),  keep_nonveg), tags=tags_mg)
    save_tif_like(data_path, out_pur_view,     apply_keep_as_nodata(stretch01_view(PUR),      keep_nonveg), tags=tags_p)
    save_tif_like(data_path, out_score_view,   apply_keep_as_nodata(stretch01_view(SCORE),    keep_nonveg), tags=tags_s)

    # ---- RAW + VIEW ----
    save_tif_like(data_path, out_fe_raw,     FE2_RAW_q,  nodata=NODATA_VAL, tags=tags_fr)
    save_tif_like(data_path, out_ioi_raw,    IOI_RAW_q,  nodata=NODATA_VAL, tags=tags_ir)
    save_tif_like(data_path, out_clay_raw,   CLAY_RAW_q, nodata=NODATA_VAL, tags=tags_cr)

    save_tif_like(data_path, out_fe_raw_view,     apply_keep_as_nodata(stretch01_view(FE2_RAW),  keep_nonveg), tags=tags_fr)
    save_tif_like(data_path, out_ioi_raw_view,    apply_keep_as_nodata(stretch01_view(IOI_RAW),  keep_nonveg), tags=tags_ir)
    save_tif_like(data_path, out_clay_raw_view,   apply_keep_as_nodata(stretch01_view(CLAY_RAW), keep_nonveg), tags=tags_cr)

    # ---- VRT "explorative" (RAW) ----
    vrt_path = out_dir / f"explore_raw_RGB{cube_suffix}.vrt"
    ref_tif = out_fe_raw_view
    write_rgb_vrt(ref_tif, vrt_path, r_path=out_fe_raw_view, g_path=out_ioi_raw_view, b_path=out_clay_raw_view)

    # ---- VRT "ophiolite-alteration" (Mg-OH) ----
    vrt_mg = out_dir / f"explore_mgoh_RGB{cube_suffix}.vrt"
    ref2 = out_bdmg_view  # usa il mgoh_view come riferimento
    write_rgb_vrt(ref2, vrt_mg, r_path=out_bdmg_view, g_path=out_i_view, b_path=out_pur_view)

    print("\n[FINITO] Output in:", out_dir)

if __name__ == "__main__":
    main()