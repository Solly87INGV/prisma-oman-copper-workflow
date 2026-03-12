# prisma-oman-copper-workflow
PRISMA-based workflow for pre-survey spectral screening, copper priority mapping, and post-survey refinement in the Oman ophiolite.
Overview

This repository contains the custom Python codes and supporting spectral libraries used in a PRISMA VNIR-SWIR workflow developed for copper-related alteration targeting in the Wadi al-Ma'awil sector, northern Oman.

The workflow was designed as a reproducible two-stage approach:

Pre-survey screening

generation of quicklook spectral products;

selective SAM-based screening;

production of a copper priority mask used as an operational targeting layer during field survey;

Post-survey refinement

suite-based spectral mixture analysis (SMA);

ROI-based comparison between INSIDE and OUTSIDE mask areas;

qualitative interpretation supported by laboratory ASD FieldSpec measurements.

The workflow is intended as a screening and decision-support approach, not as a deterministic mineral identification system.

Repository structure
code/

Python scripts used in the workflow.

1_quicklook_index/

generation of quicklook products such as indices and band depths from the PRISMA clean cube.

2_SAM/

selective Spectral Angle Mapper (SAM) screening;

generation of the copper priority mask and associated diagnostic outputs.

3_SMA/

suite-based Spectral Mixture Analysis (SMA);

generation of fraction maps, RMSE, maxscore, and dominant endmember outputs.

4_RastCalc_IN&OUT/

raster calculations and ROI-based statistics;

comparison between INSIDE and OUTSIDE areas of the copper priority mask.

metadata/SMA_SpecLibUSGS_.../

Supporting spectral libraries and metadata used for SMA analysis.

Subfolders include:

Alteration/

Copper/

Gossan/

Host_rock/

These folders contain selected USGS-derived endmember spectra and supporting files organized by interpretive suite.

Main workflow logic

The workflow follows these general steps:

Start from a georeferenced and spectrally cleaned PRISMA L2D VNIR-SWIR cube;

Generate quicklook products for preliminary scene inspection;

Apply selective SAM screening to produce a copper priority mask;

Use the mask to support field targeting and sample collection;

Run suite-based SMA on selected endmember groups;

Compare INSIDE versus OUTSIDE mask behavior within the survey ROI;

Support interpretation through qualitative comparison with FieldSpec laboratory spectra.

Notes

The scripts were developed for the specific study presented in the associated manuscript and may require path adaptation and minor adjustments before reuse.

The repository is organized to keep the workflow lightweight and transparent rather than fully automated as a standalone package.

Outputs should be interpreted conservatively, especially in arid ophiolitic settings affected by sub-pixel mixing, supergene overprint, and spectrally similar alteration assemblages.

Author

Marco Solinas

Citation

If you use this repository, please cite the related manuscript once available.
