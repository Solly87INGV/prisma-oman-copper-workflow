#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRISMA → SAM (rame) — versione selettiva
- Calcolo angolo SAM globale (tutti i minerali) e Cu-only nello stesso loop
- Soglia copper = percentile basso (angoli piccoli) + conf minima
- Maschere senza OR permissivi
- Prospectivity ribilanciata
"""

import os, re, math, json, logging
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape as shp_shape
import fiona
from scipy.ndimage import median_filter, binary_dilation, binary_erosion

# ==================== CONFIG ====================
INPUT_ENVI = r"D:\INGV\1_Human Mobility\PRISMA\PRS_L2D_STD_20201112065___422\VNIR_SWIR_latlon_CLEAN_422"
SPECTRAL_DIR = r"E:\1_Human Mobility\Hyperspectral\Hyperspectral\selected_minerals"
OUTPUT_DIR   = r"D:\INGV\1_Human Mobility\PRISMA\PRS_L2D_STD_20201112065___422\output_SAM_422"

# opzionale: statistiche per geologia
GEOLOGY_SHP   = None   # es: r"D:\...\geology.shp"
GEOLOGY_FIELD = None   # es: "UNIT"

COPPER_MINERALS = ["Malachite","Azurite","Cuprite","Chalcopyrite","Bornite","Chrysocolla","Tennantite","Covellite"]
ALTER_MINERALS  = ["Chlorite","Sericite","Kaolinite","Alunite","Muscovite","Hematite","Goethite","Jarosite","Pyrite"]

# --- Soglie/adattività ---
BASE_ANGLE_THR_RAD  = 0.35     # soglia QA globale (~20°) per "mask_ok" (solo diagnostica/alter)
CU_PERCENTILE       = 0.10     # **percentile basso** su ang_cu (top 10% migliori)
MIN_CONF_CU         = 0.70     # **confidence minima** per dichiarare rame

# --- Pesi prospectivity (0–1) ---
W_CU   = 0.60
W_ALT  = 0.25
W_CONF = 0.15

# --- Certainty tiers (su confidence globale) ---
CERT_HIGH = 0.80
CERT_MED  = 0.60
CERT_LOW  = 0.45

# --- Pulizia spaziale ---
DO_MEDIAN_3x3 = True
DILATE_ITERS  = 1
ERODE_ITERS   = 0

# ------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(os.path.join(OUTPUT_DIR,"processing_log_selective.txt"), encoding="utf-8"),
              logging.StreamHandler()]
)

def read_wavelengths_from_hdr(hdr_path):
    try:
        txt = open(hdr_path,'r',encoding='utf-8',errors='ignore').read()
        m = re.search(r"wavelength\s*=\s*\{([^}]*)\}", txt, flags=re.I|re.S)
        if not m: return None
        vals = [v.strip() for v in m.group(1).replace("\n"," ").split(",")]
        wls = []
        for v in vals:
            try: wls.append(float(v))
            except: pass
        wls = np.array(wls, dtype=float)
        if np.nanmedian(wls) > 100: wls = wls/1000.0
        return wls
    except Exception as e:
        logging.warning(f"WL HDR: {e}")
        return None

def mask_good_bands(wl_um):
    wl = np.array(wl_um)
    return (wl>=0.45) & (wl<=2.45) & ~(((wl>=1.35)&(wl<=1.45)) | ((wl>=1.80)&(wl<=1.95)))

def parse_usgs_txt(path):
    xs, ys = [], []
    with open(path,'r',encoding='utf-8',errors='ignore') as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith(("splib","Version","Description","#",";")): continue
            parts=line.replace(","," ").split()
            nums=[]
            for p in parts:
                try: nums.append(float(p))
                except: pass
            if len(nums)>=2:
                xs.append(nums[0]); ys.append(nums[1])
            elif len(nums)==1:
                ys.append(nums[0])
    if len(xs)>=5 and len(ys)>=5 and len(xs)==len(ys):
        wl = np.array(xs,float); refl = np.array(ys,float)
    elif len(ys)>=20:
        wl = np.linspace(0.35,2.5,len(ys)); refl = np.array(ys,float)
    else:
        return None, None
    if np.nanmax(refl)>1.5: refl = refl/100.0
    return np.clip(wl,0,5), np.clip(refl,0,1.2)

def continuum_removal(y):
    x = np.arange(len(y))
    hull=[]
    for i in range(len(x)):
        while len(hull)>=2:
            x1,y1=hull[-2]; x2,y2=hull[-1]; x3,y3=x[i],y[i]
            if (y2-y1)*(x3-x2) >= (y3-y2)*(x2-x1):
                hull.pop()
            else:
                break
        hull.append((x[i],y[i]))
    hx=np.array([p[0] for p in hull]); hy=np.array([p[1] for p in hull])
    cont=np.interp(x,hx,hy); cont[cont==0]=1e-6
    return np.clip(y/cont, 0, 2.0)

def sam_angle(a,b):
    a = a.astype(np.float64); b=b.astype(np.float64)
    na=np.linalg.norm(a); nb=np.linalg.norm(b)
    if na==0 or nb==0: return math.pi/2
    cosang = np.clip(np.dot(a,b)/(na*nb), -1.0, 1.0)
    return math.acos(cosang)

def open_prisma(envi_base):
    data_file=envi_base
    if not os.path.exists(data_file):
        if os.path.exists(envi_base+".dat"):
            data_file=envi_base+".dat"
        else:
            raise FileNotFoundError(f"File dati non trovato: {envi_base}")
    with rasterio.open(data_file) as ds:
        cube = ds.read()  # (bands,rows,cols)
        prof= ds.profile.copy()
        transform,crs = ds.transform, ds.crs
    return cube, prof, transform, crs

def save_tif(path, arr, transform, crs, dtype=None):
    if dtype is None: dtype = arr.dtype
    with rasterio.open(
        path,'w',driver='GTiff',
        height=arr.shape[0], width=arr.shape[1], count=1,
        dtype=dtype, crs=crs, transform=transform, compress='DEFLATE'
    ) as dst:
        dst.write(arr.astype(dtype),1)

def main():
    logging.info("=== PRISMA SAM (selekt) ===")
    cube, prof, transform, crs = open_prisma(INPUT_ENVI)      # (bands,rows,cols)
    n_b, r, c = cube.shape
    img = np.transpose(cube,(1,2,0))                          # (r,c,b)

    hdr = INPUT_ENVI if INPUT_ENVI.endswith(".hdr") else INPUT_ENVI+".hdr"
    wl = read_wavelengths_from_hdr(hdr)
    if wl is None or len(wl)!=n_b:
        logging.warning("Wavelengths mancanti/mismatch: uso 0.4–2.5 linspace")
        wl = np.linspace(0.4,2.5,n_b)
    good = mask_good_bands(wl); gidx = np.where(good)[0]
    logging.info(f"Bande buone: {good.sum()}/{n_b}")

    # Preprocess PRISMA: clamp/NaN, L2-norm e continuum removal sulle bande buone
    data = img.reshape(-1, n_b)
    data[~np.isfinite(data)] = 0
    data[data<0] = 0
    dg = data[:, gidx]
    norms = np.linalg.norm(dg, axis=1, keepdims=True) + 1e-9
    dg = dg / norms
    for i in range(dg.shape[0]):
        dg[i,:] = continuum_removal(dg[i,:])
    img_g = dg.reshape(r,c,-1)

    # Carica libreria
    files = [f for f in os.listdir(SPECTRAL_DIR) if f.lower().endswith(".txt")]
    minerals = COPPER_MINERALS + ALTER_MINERALS
    lib = {}
    for m in minerals:
        cand = [f for f in files if m.lower() in f.lower()]
        if not cand:
            logging.warning(f"LIB mancante: {m}"); continue
        wl_u, refl_u = parse_usgs_txt(os.path.join(SPECTRAL_DIR, cand[0]))
        if wl_u is None:
            logging.warning(f"LIB non valida: {m}"); continue
        interp = np.interp(wl[good], wl_u, refl_u, left=0, right=0)
        interp = interp/(np.linalg.norm(interp)+1e-9)
        interp = continuum_removal(interp)
        lib[m]=interp
    if not lib:
        raise RuntimeError("Nessuno spettro valido caricato.")

    mineral_list = list(lib.keys())
    json.dump({i+1:m for i,m in enumerate(mineral_list)},
              open(os.path.join(OUTPUT_DIR,"legend_minerals.json"),"w",encoding="utf-8"),
              ensure_ascii=False, indent=2)
    logging.info(f"Minerali usati ({len(mineral_list)}): {mineral_list}")

    # --- SAM loop: best su tutti i minerali + min angolo SOLO rame ---
    best_idx = np.zeros((r,c), np.uint16)
    best_ang = np.full((r,c), math.pi/2, np.float32)
    ang_cu   = np.full((r,c), math.pi/2, np.float32)  # min angolo tra soli minerali Cu

    cu_set = set([m for m in COPPER_MINERALS if m in mineral_list])

    for i in range(r):
        if i%50==0: logging.info(f"Riga {i+1}/{r}")
        row = img_g[i,:,:]          # (c, gb)
        for j in range(c):
            s = row[j,:]
            if not np.any(s): continue
            ba_all = math.pi/2; bidx=0
            ba_cu  = math.pi/2
            for k,m in enumerate(mineral_list):
                ang = sam_angle(lib[m], s)
                # best globale
                if ang < ba_all:
                    ba_all=ang; bidx=k+1
                # best tra soli Cu
                if m in cu_set and ang < ba_cu:
                    ba_cu = ang
            best_idx[i,j]=bidx
            best_ang[i,j]=ba_all
            ang_cu[i,j]  = ba_cu

    # Confidence globale e copper_conf
    confidence   = 1.0 - (best_ang/(math.pi/2))
    copper_conf  = 1.0 - (ang_cu  /(math.pi/2))
    copper_conf  = np.clip(copper_conf, 0, 1)

    # QA: mask_ok globale (non usata per copper, solo diagnostica/alter)
    mask_ok = best_ang <= BASE_ANGLE_THR_RAD

    # ---- Soglia copper CONSERVATIVA (percentile basso) + conf minima ----
    valid_cu = np.isfinite(ang_cu)
    if not valid_cu.any():
        logging.warning("ang_cu vuoto: copper mask sarà tutta 0.")
        thr_cu = math.pi/2
    else:
        thr_cu = np.quantile(ang_cu[valid_cu], CU_PERCENTILE)
    thr_cu_deg = float(np.degrees(thr_cu))

    copper_mask = ((ang_cu <= thr_cu) & (confidence >= MIN_CONF_CU)).astype(np.uint8)

    # Alteration mask (best ∈ alter & QA globale)
    is_best_alt = np.isin(best_idx, [mineral_list.index(m)+1 for m in ALTER_MINERALS if m in mineral_list])
    alteration_mask = (is_best_alt & mask_ok).astype(np.uint8)

    # Pulizia spaziale
    if DO_MEDIAN_3x3:
        copper_mask     = median_filter(copper_mask, size=3)
        alteration_mask = median_filter(alteration_mask, size=3)
    if DILATE_ITERS>0:
        for _ in range(DILATE_ITERS):
            copper_mask = binary_dilation(copper_mask).astype(np.uint8)
    if ERODE_ITERS>0:
        for _ in range(ERODE_ITERS):
            copper_mask = binary_erosion(copper_mask).astype(np.uint8)

    # Prospectivity (0–1): rame + alter + conf (bilanciati)
    alter_dil = binary_dilation(alteration_mask, structure=np.ones((3,3))).astype(np.uint8)
    prospectivity = (W_CU*copper_conf + W_ALT*alter_dil + W_CONF*confidence).astype(np.float32)
    prospectivity = np.clip(prospectivity, 0, 1)

    # Certainty map
    certainty_map = np.zeros((r,c), np.uint8)
    certainty_map[(confidence>=CERT_LOW) & (confidence<CERT_MED)]  = 1
    certainty_map[(confidence>=CERT_MED) & (confidence<CERT_HIGH)] = 2
    certainty_map[(confidence>=0.90)] = 3

    # ---- Salvataggi ----
    base = os.path.splitext(os.path.basename(INPUT_ENVI))[0]
    save_tif(os.path.join(OUTPUT_DIR,f"{base}_mineral_map.tif"), best_idx, transform, crs, 'uint16')
    save_tif(os.path.join(OUTPUT_DIR,f"{base}_angle_deg.tif"), np.degrees(best_ang).astype(np.float32), transform, crs, 'float32')
    save_tif(os.path.join(OUTPUT_DIR,f"{base}_confidence.tif"), confidence.astype(np.float32), transform, crs, 'float32')
    save_tif(os.path.join(OUTPUT_DIR,f"{base}_angcu_deg.tif"), np.degrees(ang_cu).astype(np.float32), transform, crs, 'float32')
    save_tif(os.path.join(OUTPUT_DIR,f"{base}_copper_conf.tif"), copper_conf.astype(np.float32), transform, crs, 'float32')
    save_tif(os.path.join(OUTPUT_DIR,f"{base}_copper_mask.tif"), (copper_mask*255).astype(np.uint8), transform, crs, 'uint8')
    save_tif(os.path.join(OUTPUT_DIR,f"{base}_alteration_mask.tif"), (alteration_mask*255).astype(np.uint8), transform, crs, 'uint8')
    save_tif(os.path.join(OUTPUT_DIR,f"{base}_prospectivity.tif"), prospectivity.astype(np.float32), transform, crs, 'float32')
    save_tif(os.path.join(OUTPUT_DIR,f"{base}_certainty_map.tif"), certainty_map, transform, crs, 'uint8')

    # ---- Report/Log numerico (percentili e % pixel) ----
    def pct(n): return 100.0*n/(r*c)
    def stats(arr, msk=None):
        if msk is not None: arr = arr[msk>0]
        arr = arr[np.isfinite(arr)]
        if arr.size==0: return {}
        return {
            "min": float(np.nanmin(arr)),
            "p10": float(np.nanpercentile(arr,10)),
            "p25": float(np.nanpercentile(arr,25)),
            "p50": float(np.nanpercentile(arr,50)),
            "p75": float(np.nanpercentile(arr,75)),
            "p90": float(np.nanpercentile(arr,90)),
            "max": float(np.nanmax(arr))
        }

    rep = {
        "pixels_total": int(r*c),
        "copper_pixels": int(copper_mask.sum()),
        "copper_pct": pct(copper_mask.sum()),
        "alter_pixels": int(alteration_mask.sum()),
        "alter_pct": pct(alteration_mask.sum()),
        "angle_deg_stats": stats(np.degrees(best_ang)),
        "angcu_deg_stats": stats(np.degrees(ang_cu)),
        "confidence_stats": stats(confidence),
        "copper_conf_stats": stats(copper_conf, copper_mask),
        "prospectivity_stats": stats(prospectivity),
        "thresholds": {
            "global_QA_deg": float(np.degrees(BASE_ANGLE_THR_RAD)),
            "copper_percentile": CU_PERCENTILE,
            "copper_thr_deg": thr_cu_deg,
            "min_conf_cu": MIN_CONF_CU
        },
        "legend": {i+1:m for i,m in enumerate(mineral_list)}
    }
    json.dump(rep, open(os.path.join(OUTPUT_DIR,"report_selective.json"),"w",encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # (Opz.) statistiche per geologia
    if GEOLOGY_SHP and GEOLOGY_FIELD:
        with fiona.open(GEOLOGY_SHP,'r') as src:
            geoms=[]; classes=[]
            for f in src:
                geoms.append(shp_shape(f["geometry"]).__geo_interface__)
                classes.append(f["properties"][GEOLOGY_FIELD])
        class_set = {v:i+1 for i,v in enumerate(sorted(set(classes)))}
        shapes = [(g, class_set[c]) for g,c in zip(geoms, classes)]
        class_ras = rasterize(shapes, out_shape=(r,c), transform=transform, fill=0, dtype='uint16')
        rows=[["class","pixels","angcu_p50","angcu_p90","conf_p90","cuconf_p90","pros_p90","copper_pct"]]
        for name, code in class_set.items():
            m = (class_ras==code)
            if m.sum()==0: continue
            def P(arr,p): 
                a=arr[m]; a=a[np.isfinite(a)]
                return float(np.nanpercentile(a,p)) if a.size else float("nan")
            rows.append([
                name, int(m.sum()),
                P(np.degrees(ang_cu),50), P(np.degrees(ang_cu),90),
                P(confidence,90), P(copper_conf,90), P(prospectivity,90),
                pct((copper_mask & m).sum())
            ])
        import csv
        with open(os.path.join(OUTPUT_DIR,"zonal_geology_selective.csv"),"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerows(rows)

    logging.info("Fatto. Controlla report_selective.json e gli output TIFF.")

if __name__ == "__main__":
    main()
