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
