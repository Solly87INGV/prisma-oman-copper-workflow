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

NAME = "VNIR_SWIR_latlon_219"
RUN_LABEL = "Alteration"

COPPER_MASK = "VNIR_SWIR_latlon_219_copper_mask.tif"
RMSE_TIF = "VNIR_SWIR_latlon_219_sma_rmse.tif"
MAXSCORE_TIF = "VNIR_SWIR_latlon_219_sma_maxscore_selected.tif"

ROI_SHP = "Area_Samples.shp"
APPLY_ROI = True

OUT_DIR = "output_stats"
WRITE_MASKED_TIFS = True
NODATA_OUT = -9999.0
MASK_VALUE = 255

def _stats(arr: np.ndarray) -> dict:
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

    minx = miny = maxx = maxy = None
    for g in geoms:
        bx, by, bX, bY = rasterio.features.bounds(g)
        minx = bx if minx is None else min(minx, bx)
        miny = by if miny is None else min(miny, by)
        maxx = bX if maxx is None else max(maxx, bX)
        maxy = bY if maxy is None else max(maxy, bY)

    win = from_bounds(minx, miny, maxx, maxy, transform=raster_transform)
    win = win.round_offsets().round_lengths()

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

def _compute_stats_for_selector(selector_mask, rmse_w, ms_w, nodata_rmse, nodata_ms):
    valid_rmse = selector_mask.copy()
    if nodata_rmse is not None:
        valid_rmse &= (rmse_w != nodata_rmse)
    valid_rmse &= np.isfinite(rmse_w)

    valid_ms = selector_mask.copy()
    if nodata_ms is not None:
        valid_ms &= (ms_w != nodata_ms)
    valid_ms &= np.isfinite(ms_w)

    rmse_in = rmse_w[valid_rmse].astype(np.float64)
    ms_in   = ms_w[valid_ms].astype(np.float64)

    return _stats(rmse_in), _stats(ms_in), valid_rmse, valid_ms

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"=== {NAME} | {RUN_LABEL} | ROI={APPLY_ROI} ===")

    mask, meta_mask, nodata_mask, tr, crs, H, W = _read_band1(COPPER_MASK)
    rmse, meta_rmse, nodata_rmse, _, _, _, _ = _read_band1(RMSE_TIF)
    ms,   meta_ms,   nodata_ms,   _, _, _, _ = _read_band1(MAXSCORE_TIF)

    _aligned_check(meta_mask, meta_rmse, label="mask vs rmse")
    _aligned_check(meta_mask, meta_ms,   label="mask vs maxscore")

    if APPLY_ROI:
        if not os.path.exists(ROI_SHP):
            raise FileNotFoundError(f"ROI_SHP non trovato: {ROI_SHP}")

        geoms, (row_off, col_off, r, c) = _read_roi_geoms_and_window(ROI_SHP, crs, tr, H, W)
        win = rasterio.windows.Window(col_off=col_off, row_off=row_off, width=c, height=r)
        out_transform = window_transform(win, tr)

        mask_w = mask[row_off:row_off + r, col_off:col_off + c]
        rmse_w = rmse[row_off:row_off + r, col_off:col_off + c]
        ms_w   = ms[row_off:row_off + r, col_off:col_off + c]

        roi_mask = geometry_mask(geoms, out_shape=(r, c), transform=out_transform, invert=True)

        meta_out = meta_rmse.copy()
        meta_out.update({"height": r, "width": c})
    else:
        mask_w, rmse_w, ms_w = mask, rmse, ms
        roi_mask = np.ones_like(mask, dtype=bool)
        out_transform = tr
        meta_out = meta_rmse.copy()

    inside_sel  = (mask_w == MASK_VALUE) & roi_mask
    outside_sel = (mask_w != MASK_VALUE) & roi_mask

    inside_rmse, inside_ms, valid_rmse_in, valid_ms_in = _compute_stats_for_selector(
        inside_sel, rmse_w, ms_w, nodata_rmse, nodata_ms
    )
    outside_rmse, outside_ms, valid_rmse_out, valid_ms_out = _compute_stats_for_selector(
        outside_sel, rmse_w, ms_w, nodata_rmse, nodata_ms
    )

    print("INSIDE (mask==255) RMSE:", inside_rmse)
    print("INSIDE (mask==255) MaxScore:", inside_ms)
    print("OUTSIDE (mask!=255) RMSE:", outside_rmse)
    print("OUTSIDE (mask!=255) MaxScore:", outside_ms)

    if WRITE_MASKED_TIFS:
        rmse_inside = np.full_like(rmse_w, NODATA_OUT, dtype=np.float32)
        rmse_inside[valid_rmse_in] = rmse_w[valid_rmse_in].astype(np.float32)
        _write_masked_tif(os.path.join(OUT_DIR, f"{NAME}_{RUN_LABEL}_ROI_inside_rmse.tif"), rmse_inside, meta_out, out_transform)

        ms_inside = np.full_like(ms_w, NODATA_OUT, dtype=np.float32)
        ms_inside[valid_ms_in] = ms_w[valid_ms_in].astype(np.float32)
        _write_masked_tif(os.path.join(OUT_DIR, f"{NAME}_{RUN_LABEL}_ROI_inside_maxscore.tif"), ms_inside, meta_out, out_transform)

        rmse_outside = np.full_like(rmse_w, NODATA_OUT, dtype=np.float32)
        rmse_outside[valid_rmse_out] = rmse_w[valid_rmse_out].astype(np.float32)
        _write_masked_tif(os.path.join(OUT_DIR, f"{NAME}_{RUN_LABEL}_ROI_outside_rmse.tif"), rmse_outside, meta_out, out_transform)

        ms_outside = np.full_like(ms_w, NODATA_OUT, dtype=np.float32)
        ms_outside[valid_ms_out] = ms_w[valid_ms_out].astype(np.float32)
        _write_masked_tif(os.path.join(OUT_DIR, f"{NAME}_{RUN_LABEL}_ROI_outside_maxscore.tif"), ms_outside, meta_out, out_transform)

    results = {
        "scene": NAME,
        "run_label": RUN_LABEL,
        "paths": {
            "copper_mask": COPPER_MASK,
            "rmse": RMSE_TIF,
            "maxscore": MAXSCORE_TIF,
            "roi_shp": ROI_SHP if APPLY_ROI else None
        },
        "mask_value": int(MASK_VALUE),
        "roi_applied": bool(APPLY_ROI),
        "inside": {
            "definition": f"(mask == {MASK_VALUE}) AND inside ROI",
            "rmse_stats": inside_rmse,
            "maxscore_stats": inside_ms
        },
        "outside": {
            "definition": f"(mask != {MASK_VALUE}) AND inside ROI",
            "rmse_stats": outside_rmse,
            "maxscore_stats": outside_ms
        },
        "note": "Confronto INSIDE vs OUTSIDE calcolato nella sola ROI; pixel con nodata/non-finite esclusi."
    }

    json_path = os.path.join(OUT_DIR, f"mask_stats_{NAME}_{RUN_LABEL}_ROI_inside_outside.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(OUT_DIR, f"mask_stats_{NAME}_{RUN_LABEL}_ROI_inside_outside.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "scene","run_label","subset",
            "rmse_count","rmse_mean","rmse_median","rmse_min","rmse_max","rmse_p10","rmse_p25","rmse_p75","rmse_p90",
            "maxscore_count","maxscore_mean","maxscore_median","maxscore_min","maxscore_max","maxscore_p10","maxscore_p25","maxscore_p75","maxscore_p90"
        ])
        for subset_name, rr, mm in [
            ("INSIDE_mask255", inside_rmse, inside_ms),
            ("OUTSIDE_background", outside_rmse, outside_ms),
        ]:
            w.writerow([
                NAME, RUN_LABEL, subset_name,
                rr["count"], rr["mean"], rr["median"], rr["min"], rr["max"], rr["p10"], rr["p25"], rr["p75"], rr["p90"],
                mm["count"], mm["mean"], mm["median"], mm["min"], mm["max"], mm["p10"], mm["p25"], mm["p75"], mm["p90"],
            ])

    print("\n=== DONE ===")
    print("Saved:", json_path)
    print("Saved:", csv_path)
    if WRITE_MASKED_TIFS:
        print("Saved 4 ROI masked tifs in:", OUT_DIR)

if __name__ == "__main__":
    main()
