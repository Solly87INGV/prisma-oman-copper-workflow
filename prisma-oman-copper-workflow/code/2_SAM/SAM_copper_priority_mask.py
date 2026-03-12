#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PRISMA selective SAM workflow for copper-related screening.
"""

import argparse
import csv
import json
import logging
import math
import os
import re
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import binary_dilation, binary_erosion, median_filter

COPPER_MINERALS = [
    "Malachite",
    "Azurite",
    "Cuprite",
    "Chalcopyrite",
    "Bornite",
    "Chrysocolla",
    "Tennantite",
    "Covellite",
]

ALTER_MINERALS = [
    "Chlorite",
    "Sericite",
    "Kaolinite",
    "Alunite",
    "Muscovite",
    "Hematite",
    "Goethite",
    "Jarosite",
    "Pyrite",
]

BASE_ANGLE_THR_RAD = 0.35
CU_PERCENTILE = 0.10
MIN_CONF_CU = 0.70

W_CU = 0.60
W_ALT = 0.25
W_CONF = 0.15

CERT_HIGH = 0.80
CERT_MED = 0.60
CERT_LOW = 0.45

DO_MEDIAN_3X3 = True
DILATE_ITERS = 1
ERODE_ITERS = 0


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "processing_log_selective.txt"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def read_wavelengths_from_hdr(hdr_path: Path):
    try:
        txt = hdr_path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"wavelength\s*=\s*\{([^}]*)\}", txt, flags=re.I | re.S)
        if not match:
            return None

        vals = [v.strip() for v in match.group(1).replace("\n", " ").split(",")]
        wls = []
        for v in vals:
            try:
                wls.append(float(v))
            except Exception:
                continue

        wls = np.array(wls, dtype=float)
        if np.nanmedian(wls) > 100:
            wls = wls / 1000.0
        return wls
    except Exception as exc:
        logging.warning(f"Failed to read wavelengths from HDR: {exc}")
        return None


def mask_good_bands(wl_um):
    wl = np.array(wl_um)
    return (wl >= 0.45) & (wl <= 2.45) & ~(
        ((wl >= 1.35) & (wl <= 1.45)) | ((wl >= 1.80) & (wl <= 1.95))
    )


def parse_usgs_txt(path: Path):
    xs, ys = [], []

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith(("splib", "Version", "Description", "#", ";")):
                continue

            parts = line.replace(",", " ").split()
            nums = []
            for part in parts:
                try:
                    nums.append(float(part))
                except Exception:
                    continue

            if len(nums) >= 2:
                xs.append(nums[0])
                ys.append(nums[1])
            elif len(nums) == 1:
                ys.append(nums[0])

    if len(xs) >= 5 and len(ys) >= 5 and len(xs) == len(ys):
        wl = np.array(xs, float)
        refl = np.array(ys, float)
    elif len(ys) >= 20:
        wl = np.linspace(0.35, 2.5, len(ys))
        refl = np.array(ys, float)
    else:
        return None, None

    if np.nanmax(refl) > 1.5:
        refl = refl / 100.0

    return np.clip(wl, 0, 5), np.clip(refl, 0, 1.2)


def continuum_removal(y):
    x = np.arange(len(y))
    hull = []

    for i in range(len(x)):
        while len(hull) >= 2:
            x1, y1 = hull[-2]
            x2, y2 = hull[-1]
            x3, y3 = x[i], y[i]
            if (y2 - y1) * (x3 - x2) >= (y3 - y2) * (x2 - x1):
                hull.pop()
            else:
                break
        hull.append((x[i], y[i]))

    hx = np.array([p[0] for p in hull])
    hy = np.array([p[1] for p in hull])
    cont = np.interp(x, hx, hy)
    cont[cont == 0] = 1e-6

    return np.clip(y / cont, 0, 2.0)


def sam_angle(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na == 0 or nb == 0:
        return math.pi / 2

    cosang = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
    return math.acos(cosang)


def resolve_data_path(envi_base: Path) -> Path:
    if envi_base.exists():
        return envi_base

    dat_path = envi_base.with_suffix(".dat")
    if dat_path.exists():
        return dat_path

    raise FileNotFoundError(f"Data file not found: {envi_base}")


def resolve_hdr_path(envi_base: Path) -> Path | None:
    if envi_base.suffix.lower() == ".hdr" and envi_base.exists():
        return envi_base

    hdr_path = envi_base.with_suffix(".hdr")
    if hdr_path.exists():
        return hdr_path

    alt_hdr = Path(str(envi_base) + ".hdr")
    if alt_hdr.exists():
        return alt_hdr

    return None


def open_prisma(envi_base: Path):
    data_file = resolve_data_path(envi_base)
    with rasterio.open(data_file) as ds:
        cube = ds.read()
        profile = ds.profile.copy()
        transform = ds.transform
        crs = ds.crs
    return cube, profile, transform, crs


def save_tif(path: Path, arr, transform, crs, dtype=None):
    if dtype is None:
        dtype = arr.dtype

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        compress="DEFLATE",
    ) as dst:
        dst.write(arr.astype(dtype), 1)


def stats(arr, mask=None):
    if mask is not None:
        arr = arr[mask > 0]

    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}

    return {
        "min": float(np.nanmin(arr)),
        "p10": float(np.nanpercentile(arr, 10)),
        "p25": float(np.nanpercentile(arr, 25)),
        "p50": float(np.nanpercentile(arr, 50)),
        "p75": float(np.nanpercentile(arr, 75)),
        "p90": float(np.nanpercentile(arr, 90)),
        "max": float(np.nanmax(arr)),
    }


def percentile_in_mask(arr, mask, p):
    values = arr[mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.nanpercentile(values, p))


def write_geology_stats(
    output_dir: Path,
    geology_shp: Path,
    geology_field: str,
    shape_rc,
    transform,
    ang_cu,
    confidence,
    copper_conf,
    prospectivity,
    copper_mask,
):
    from rasterio.features import rasterize
    import fiona
    from shapely.geometry import shape as shp_shape

    rows_n, cols_n = shape_rc

    with fiona.open(geology_shp, "r") as src:
        geoms = []
        classes = []
        for feat in src:
            geoms.append(shp_shape(feat["geometry"]).__geo_interface__)
            classes.append(feat["properties"][geology_field])

    class_map = {value: i + 1 for i, value in enumerate(sorted(set(classes)))}
    shapes = [(geom, class_map[cls]) for geom, cls in zip(geoms, classes)]

    class_raster = rasterize(
        shapes,
        out_shape=(rows_n, cols_n),
        transform=transform,
        fill=0,
        dtype="uint16",
    )

    def pct(n):
        return 100.0 * n / (rows_n * cols_n)

    rows = [["class", "pixels", "angcu_p50", "angcu_p90", "conf_p90", "cuconf_p90", "pros_p90", "copper_pct"]]

    for name, code in class_map.items():
        mask = class_raster == code
        if mask.sum() == 0:
            continue

        rows.append(
            [
                name,
                int(mask.sum()),
                percentile_in_mask(np.degrees(ang_cu), mask, 50),
                percentile_in_mask(np.degrees(ang_cu), mask, 90),
                percentile_in_mask(confidence, mask, 90),
                percentile_in_mask(copper_conf, mask, 90),
                percentile_in_mask(prospectivity, mask, 90),
                pct((copper_mask & mask).sum()),
            ]
        )

    with (output_dir / "zonal_geology_selective.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Run selective PRISMA SAM analysis.")
    parser.add_argument("--input-envi", required=True, help="Path to ENVI base file")
    parser.add_argument("--spectral-dir", required=True, help="Directory containing spectral library .txt files")
    parser.add_argument("--output-dir", default="sam_output", help="Output directory")
    parser.add_argument("--geology-shp", default=None, help="Optional geology shapefile")
    parser.add_argument("--geology-field", default=None, help="Optional geology class field")
    args = parser.parse_args()

    input_envi = Path(args.input_envi)
    spectral_dir = Path(args.spectral_dir)
    output_dir = Path(args.output_dir)

    setup_logging(output_dir)

    logging.info("=== PRISMA SAM selective ===")

    cube, _, transform, crs = open_prisma(input_envi)
    n_bands, rows_n, cols_n = cube.shape
    img = np.transpose(cube, (1, 2, 0))

    hdr_path = resolve_hdr_path(input_envi)
    wl = read_wavelengths_from_hdr(hdr_path) if hdr_path else None
    if wl is None or len(wl) != n_bands:
        logging.warning("Missing or mismatched wavelengths. Using linear 0.4–2.5 µm spacing.")
        wl = np.linspace(0.4, 2.5, n_bands)

    good = mask_good_bands(wl)
    good_idx = np.where(good)[0]
    logging.info(f"Valid bands: {good.sum()}/{n_bands}")

    data = img.reshape(-1, n_bands)
    data[~np.isfinite(data)] = 0
    data[data < 0] = 0

    data_good = data[:, good_idx]
    norms = np.linalg.norm(data_good, axis=1, keepdims=True) + 1e-9
    data_good = data_good / norms

    for i in range(data_good.shape[0]):
        data_good[i, :] = continuum_removal(data_good[i, :])

    img_good = data_good.reshape(rows_n, cols_n, -1)

    files = [f for f in spectral_dir.iterdir() if f.suffix.lower() == ".txt"]
    minerals = COPPER_MINERALS + ALTER_MINERALS
    library = {}

    for mineral in minerals:
        candidates = [f for f in files if mineral.lower() in f.name.lower()]
        if not candidates:
            logging.warning(f"Missing library spectrum: {mineral}")
            continue

        wl_u, refl_u = parse_usgs_txt(candidates[0])
        if wl_u is None:
            logging.warning(f"Invalid library spectrum: {mineral}")
            continue

        interp = np.interp(wl[good], wl_u, refl_u, left=0, right=0)
        interp = interp / (np.linalg.norm(interp) + 1e-9)
        interp = continuum_removal(interp)
        library[mineral] = interp

    if not library:
        raise RuntimeError("No valid spectra loaded from the library.")

    mineral_list = list(library.keys())

    with (output_dir / "legend_minerals.json").open("w", encoding="utf-8") as handle:
        json.dump({i + 1: m for i, m in enumerate(mineral_list)}, handle, ensure_ascii=False, indent=2)

    logging.info(f"Minerals used ({len(mineral_list)}): {mineral_list}")

    best_idx = np.zeros((rows_n, cols_n), np.uint16)
    best_ang = np.full((rows_n, cols_n), math.pi / 2, np.float32)
    ang_cu = np.full((rows_n, cols_n), math.pi / 2, np.float32)

    cu_set = {m for m in COPPER_MINERALS if m in mineral_list}

    for i in range(rows_n):
        if i % 50 == 0:
            logging.info(f"Row {i + 1}/{rows_n}")

        row = img_good[i, :, :]
        for j in range(cols_n):
            spectrum = row[j, :]
            if not np.any(spectrum):
                continue

            best_angle_all = math.pi / 2
            best_index = 0
            best_angle_cu = math.pi / 2

            for k, mineral in enumerate(mineral_list):
                angle = sam_angle(library[mineral], spectrum)

                if angle < best_angle_all:
                    best_angle_all = angle
                    best_index = k + 1

                if mineral in cu_set and angle < best_angle_cu:
                    best_angle_cu = angle

            best_idx[i, j] = best_index
            best_ang[i, j] = best_angle_all
            ang_cu[i, j] = best_angle_cu

    confidence = 1.0 - (best_ang / (math.pi / 2))
    copper_conf = 1.0 - (ang_cu / (math.pi / 2))
    copper_conf = np.clip(copper_conf, 0, 1)

    mask_ok = best_ang <= BASE_ANGLE_THR_RAD

    valid_cu = np.isfinite(ang_cu)
    if not valid_cu.any():
        logging.warning("ang_cu is empty. Copper mask will be all zeros.")
        thr_cu = math.pi / 2
    else:
        thr_cu = np.quantile(ang_cu[valid_cu], CU_PERCENTILE)

    thr_cu_deg = float(np.degrees(thr_cu))

    copper_mask = ((ang_cu <= thr_cu) & (confidence >= MIN_CONF_CU)).astype(np.uint8)

    alter_indices = [mineral_list.index(m) + 1 for m in ALTER_MINERALS if m in mineral_list]
    is_best_alt = np.isin(best_idx, alter_indices)
    alteration_mask = (is_best_alt & mask_ok).astype(np.uint8)

    if DO_MEDIAN_3X3:
        copper_mask = median_filter(copper_mask, size=3)
        alteration_mask = median_filter(alteration_mask, size=3)

    if DILATE_ITERS > 0:
        for _ in range(DILATE_ITERS):
            copper_mask = binary_dilation(copper_mask).astype(np.uint8)

    if ERODE_ITERS > 0:
        for _ in range(ERODE_ITERS):
            copper_mask = binary_erosion(copper_mask).astype(np.uint8)

    alter_dil = binary_dilation(alteration_mask, structure=np.ones((3, 3))).astype(np.uint8)
    prospectivity = (W_CU * copper_conf + W_ALT * alter_dil + W_CONF * confidence).astype(np.float32)
    prospectivity = np.clip(prospectivity, 0, 1)

    certainty_map = np.zeros((rows_n, cols_n), np.uint8)
    certainty_map[(confidence >= CERT_LOW) & (confidence < CERT_MED)] = 1
    certainty_map[(confidence >= CERT_MED) & (confidence < CERT_HIGH)] = 2
    certainty_map[confidence >= 0.90] = 3

    base_name = input_envi.stem if input_envi.suffix else input_envi.name

    save_tif(output_dir / f"{base_name}_mineral_map.tif", best_idx, transform, crs, "uint16")
    save_tif(output_dir / f"{base_name}_angle_deg.tif", np.degrees(best_ang).astype(np.float32), transform, crs, "float32")
    save_tif(output_dir / f"{base_name}_confidence.tif", confidence.astype(np.float32), transform, crs, "float32")
    save_tif(output_dir / f"{base_name}_angcu_deg.tif", np.degrees(ang_cu).astype(np.float32), transform, crs, "float32")
    save_tif(output_dir / f"{base_name}_copper_conf.tif", copper_conf.astype(np.float32), transform, crs, "float32")
    save_tif(output_dir / f"{base_name}_copper_mask.tif", (copper_mask * 255).astype(np.uint8), transform, crs, "uint8")
    save_tif(output_dir / f"{base_name}_alteration_mask.tif", (alteration_mask * 255).astype(np.uint8), transform, crs, "uint8")
    save_tif(output_dir / f"{base_name}_prospectivity.tif", prospectivity.astype(np.float32), transform, crs, "float32")
    save_tif(output_dir / f"{base_name}_certainty_map.tif", certainty_map, transform, crs, "uint8")

    def pct(n):
        return 100.0 * n / (rows_n * cols_n)

    report = {
        "pixels_total": int(rows_n * cols_n),
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
            "global_qa_deg": float(np.degrees(BASE_ANGLE_THR_RAD)),
            "copper_percentile": CU_PERCENTILE,
            "copper_thr_deg": thr_cu_deg,
            "min_conf_cu": MIN_CONF_CU,
        },
        "legend": {i + 1: m for i, m in enumerate(mineral_list)},
    }

    with (output_dir / "report_selective.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    if args.geology_shp and args.geology_field:
        write_geology_stats(
            output_dir=output_dir,
            geology_shp=Path(args.geology_shp),
            geology_field=args.geology_field,
            shape_rc=(rows_n, cols_n),
            transform=transform,
            ang_cu=ang_cu,
            confidence=confidence,
            copper_conf=copper_conf,
            prospectivity=prospectivity,
            copper_mask=copper_mask,
        )

    logging.info("Done. Check report_selective.json and output GeoTIFF files.")


if __name__ == "__main__":
    main()
