# Bangladesh Seasonal Reclassification

Analysis code and processed data for the manuscript:

> **A Cluster Analysis of Bangladesh's Monthly Climate and Its Comparison 
> with the Traditional Six-Season Calendar**
>
> Md. Matiur Rahman Molla, Md. Hridoy Hossen, Ayesha Siddika
> *Theoretical and Applied Climatology* (2026)

## Overview

This repository contains the analysis pipeline and processed dataset 
for a clustering-based reclassification of Bangladesh's seasons. Using 
30 years (1985–2014) of monthly climate records from 33 Bangladesh 
Meteorological Department (BMD) stations, we cluster the twelve calendar 
months in a six-variable climate space (Tmax, Tmin, rainfall, humidity, 
wind speed, cloud cover) and compare the resulting partition with the 
traditional six-season calendar.

**Key finding:** The monthly climatology supports a four-season 
description (Winter, Pre-Monsoon, Monsoon, Post-Monsoon), with October 
as a separate transition month.

## Repository structure

├── code/ Analysis scripts
├── data/ Processed station-month dataset
├── tables/ Tables 1-8 and S1-S8 (CSV)
├── figures/ Figures 1-8 and FigS1 (PNG)
├── docs/ Project documentation
├── requirements.txt Python package versions
├── LICENSE MIT License
└── README.md


## Requirements

- Python 3.11
- See `requirements.txt` for package versions

Install dependencies:
pip install -r requirements.txt


## Reproducing the analysis
python code/Complete_Analysis.py


Outputs are written to `tables/` and `figures/`.

## Data

Raw station records were obtained from the Bangladesh Meteorological 
Department (BMD; https://live3.bmd.gov.bd/) and are subject to BMD's 
data-sharing policy. Access requests should be directed to BMD.

Processed data (`data/master_clean.csv`) contains 10,865 station-months 
across 33 stations and 30 years.

## Citation

If you use this code or data, please cite the manuscript.

## License

MIT License — see `LICENSE`.

## Contact

- **Md. Hridoy Hossen** — hridoy1922025.stat@std.iu.ac.bd
- **Md. Matiur Rahman Molla** — matiuriustat@gmail.com
