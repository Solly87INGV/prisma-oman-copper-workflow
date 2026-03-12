import os
import json
import csv
import numpy as np
import rasterio

from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from rasterio.windows import transform as window_transform
from rasterio.warp import transform_geom
import fiona

# =========================
# CONFIG (UNA SOLA SCENA)
# =========================
NAME = "VNIR_SWIR_latlon_219"

COPPER_MASK = r"D:\INGV\1_Human Mobility\PRISMA\output_SAM_selective\VNIR_SWIR_latlon_219_copper_mask.tif"  # 0..255
RMSE_TIF    = r"D:\INGV\Hyperspectral\NEW_Readapt_scrpt_USGS_SpecLib\new selection minerals usgs\MASPAG 2025\outputs\output_219\output_Gossan\VNIR_SWIR_latlon_219_sma_rmse.tif"
MAXSCORE_TIF= r"D:\INGV\Hyperspectral\NEW_Readapt_scrpt_USGS_SpecLib\new selection minerals usgs\MASPAG 2025\outputs\output_219\output_Gossan\VNIR_SWIR_latlon_219_sma_maxscore_selected.tif"

# ROI polygon (area campioni)
ROI_SHP = r"D:\INGV\Hyperspectral\NEW_Readapt_scrpt_USGS_SpecLib\new selection minerals usgs\MASPAG 2025\Analisi_GIS\Area_Samples.shp"
APPLY_ROI = True

OUT_DIR = r"D:\INGV\Hyperspectral\NEW_Readapt_scrpt_USGS_SpecLib\new selection minerals usgs\MASPAG 2025\Analisi_GIS\output_gos"

WRITE_MASKED_TIFS = True
NODATA_OUT = -9999.0
MASK_VALUE = 255
# =========================


def _stats(arr: np.ndarray) -> dict:
    """Stats su array 1D già filtrato (solo pixel validi nella mask)."""
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


def _read_band1(path: str):
    with rasterio.open(path) as ds:
        a = ds.read(1)
        meta = ds.meta.copy()
        nodata = ds.nodata
        transform = ds.transform
        crs = ds.crs
        height = ds.height
        width = ds.width
    return a, meta, nodata, transform, crs, height, width


def _aligned_check(ref_meta, other_meta, label=""):
    ok = True
    if ref_meta["height"] != other_meta["height"] or ref_meta["width"] != other_meta["width"]:
        ok = False
    if ref_meta.get("transform") != other_meta.get("transform"):
        ok = False
    if (ref_meta.get("crs") or None) != (other_meta.get("crs") or None):
        ok = False
    if not ok:
        raise RuntimeError(
            f"Raster non allineati ({label}). "
            f"Serve stessa griglia: extent/transform/shape/crs."
        )


def _write_masked_tif(out_path: str, data: np.ndarray, ref_meta: dict, transform):
    meta = ref_meta.copy()
    meta.update({
        "dtype": "float32",
        "count": 1,
        "nodata": NODATA_OUT,
        "compress": "DEFLATE",
        "transform": transform
    })
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(data.astype(np.float32), 1)


def _read_roi_geoms_and_window(shp_path, raster_crs, raster_transform, raster_height, raster_width):
    with fiona.open(shp_path, "r") as src:
        shp_crs = src.crs_wkt or src.crs
        geoms = []
        for feat in src:
            if not feat or "geometry" not in feat or feat["geometry"] is None:
                continue
            g = feat["geometry"]
            if shp_crs:
                g = transform_geom(shp_crs, raster_crs.to_string(), g, antimeridian_cutting=True, precision=8)
            geoms.append(g)

    if not geoms:
        raise RuntimeError("ROI shapefile: nessuna geometria valida trovata.")

    # bounds union
    minx = miny = maxx = maxy = None
    for g in geoms:
        bx, by, bX, bY = rasterio.features.bounds(g)
        minx = bx if minx is None else min(minx, bx)
        miny = by if miny is None else min(miny, by)
        maxx = bX if maxx is None else max(maxx, bX)
        maxy = bY if maxy is None else max(maxy, bY)

    win = from_bounds(minx, miny, maxx, maxy, transform=raster_transform)
    win = win.round_offsets().round_lengths()

    # clamp
    col_off = max(0, int(win.col_off))
    row_off = max(0, int(win.row_off))
    width = int(win.width)
    height = int(win.height)

    if col_off + width > raster_width:
        width = raster_width - col_off
    if row_off + height > raster_height:
        height = raster_height - row_off
    if width <= 0 or height <= 0:
        raise RuntimeError("ROI window fuori dal raster o vuota.")

    return geoms, (row_off, col_off, height, width)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"=== {NAME} | MASK={MASK_VALUE} | ROI={APPLY_ROI} ===")

    mask, meta_mask, nodata_mask, tr, crs, H, W = _read_band1(COPPER_MASK)
    rmse, meta_rmse, nodata_rmse, tr_r, crs_r, Hr, Wr = _read_band1(RMSE_TIF)
    ms,   meta_ms,   nodata_ms,   tr_m, crs_m, Hm, Wm = _read_band1(MAXSCORE_TIF)

    # allineamento griglia
    _aligned_check(meta_mask, meta_rmse, label="mask vs rmse")
    _aligned_check(meta_mask, meta_ms,   label="mask vs maxscore")

    # ROI: crop window + mask
    if APPLY_ROI:
        if not os.path.exists(ROI_SHP):
            raise FileNotFoundError(f"ROI_SHP non trovato: {ROI_SHP}")

        geoms, (row_off, col_off, r, c) = _read_roi_geoms_and_window(ROI_SHP, crs, tr, H, W)
        win = rasterio.windows.Window(col_off=col_off, row_off=row_off, width=c, height=r)
        out_transform = window_transform(win, tr)

        # crop arrays alla window ROI
        mask_w = mask[row_off:row_off + r, col_off:col_off + c]
        rmse_w = rmse[row_off:row_off + r, col_off:col_off + c]
        ms_w   = ms[row_off:row_off + r, col_off:col_off + c]

        # roi_mask True dentro poligono
        roi_mask = geometry_mask(geoms, out_shape=(r, c), transform=out_transform, invert=True)

        # aggiorna meta per output croppati
        meta_out = meta_rmse.copy()
        meta_out.update({"height": r, "width": c})

    else:
        # nessuna ROI: usa tutto
        mask_w, rmse_w, ms_w = mask, rmse, ms
        roi_mask = np.ones_like(mask, dtype=bool)
        out_transform = tr
        meta_out = meta_rmse.copy()

    # inside = mask==255 AND dentro ROI
    inside = (mask_w == MASK_VALUE) & roi_mask

    # valid RMSE
    valid_rmse = inside.copy()
    if nodata_rmse is not None:
        valid_rmse &= (rmse_w != nodata_rmse)
    valid_rmse &= np.isfinite(rmse_w)

    # valid maxscore
    valid_ms = inside.copy()
    if nodata_ms is not None:
        valid_ms &= (ms_w != nodata_ms)
    valid_ms &= np.isfinite(ms_w)

    rmse_in = rmse_w[valid_rmse].astype(np.float64)
    ms_in   = ms_w[valid_ms].astype(np.float64)

    st_rmse = _stats(rmse_in)
    st_ms   = _stats(ms_in)

    results = {
        "scene": NAME,
        "paths": {
            "copper_mask": COPPER_MASK,
            "rmse": RMSE_TIF,
            "maxscore": MAXSCORE_TIF,
            "roi_shp": ROI_SHP if APPLY_ROI else None
        },
        "mask_value": int(MASK_VALUE),
        "roi_applied": bool(APPLY_ROI),
        "rmse_stats": st_rmse,
        "maxscore_stats": st_ms,
        "note": "Statistiche calcolate SOLO su pixel dove (copper_mask==MASK_VALUE) AND (inside ROI polygon) AND (raster finite/non-nodata). Output TIFF (se abilitati) sono croppati alla bounding box della ROI e mascherati fuori ROI."
    }

    print("RMSE:", st_rmse)
    print("MaxScore:", st_ms)

    # scrivi tifs mascherati (croppati ROI)
    if WRITE_MASKED_TIFS:
        rmse_masked = np.full_like(rmse_w, NODATA_OUT, dtype=np.float32)
        rmse_masked[valid_rmse] = rmse_w[valid_rmse].astype(np.float32)

        ms_masked = np.full_like(ms_w, NODATA_OUT, dtype=np.float32)
        ms_masked[valid_ms] = ms_w[valid_ms].astype(np.float32)

        _write_masked_tif(os.path.join(OUT_DIR, f"{NAME}_ROI_rmse_masked.tif"), rmse_masked, meta_out, out_transform)
        _write_masked_tif(os.path.join(OUT_DIR, f"{NAME}_ROI_maxscore_masked.tif"), ms_masked, meta_out, out_transform)

    # JSON
    json_path = os.path.join(OUT_DIR, f"mask_stats_{NAME}_ROI.json" if APPLY_ROI else f"mask_stats_{NAME}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # CSV (una riga)
    csv_path = os.path.join(OUT_DIR, f"mask_stats_{NAME}_ROI.csv" if APPLY_ROI else f"mask_stats_{NAME}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scene","roi_applied",
                    "rmse_count","rmse_mean","rmse_median","rmse_min","rmse_max","rmse_p10","rmse_p25","rmse_p75","rmse_p90",
                    "maxscore_count","maxscore_mean","maxscore_median","maxscore_min","maxscore_max","maxscore_p10","maxscore_p25","maxscore_p75","maxscore_p90"])
        w.writerow([NAME, bool(APPLY_ROI),
                    st_rmse["count"], st_rmse["mean"], st_rmse["median"], st_rmse["min"], st_rmse["max"], st_rmse["p10"], st_rmse["p25"], st_rmse["p75"], st_rmse["p90"],
                    st_ms["count"], st_ms["mean"], st_ms["median"], st_ms["min"], st_ms["max"], st_ms["p10"], st_ms["p25"], st_ms["p75"], st_ms["p90"]])

    print("\n=== DONE ===")
    print("Saved:", json_path)
    print("Saved:", csv_path)
    if WRITE_MASKED_TIFS:
        print("Saved ROI masked tifs in:", OUT_DIR)


if __name__ == "__main__":
    main()