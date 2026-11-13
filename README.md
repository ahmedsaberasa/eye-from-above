# 🛰️ Eye from Above — AI-Powered NISAR Flood Rescue Dashboard

> **2026 NASA Space Apps Challenge | Challenge: Dancing with the SARs**
>
> **Team:** EYE FROM ABOVE (Ahmed Saber & Mohamed Haitham)
>
> *Turning all-weather NASA-ISRO SAR (NISAR) L-Band radar scans into simple, life-saving flood maps, safe rescue roads, and 3-day emergency supply plans.*

---

## 🌐 Live Interactive 7-Slide Presentation (Demo)

👉 **[Click Here to Launch the Interactive 7-Slide Presentation (GitHub Pages)](https://ahmedsaberasa.github.io/eye-from-above/)**  
*(Features live Esri satellite maps, interactive flood-zone illumination, Before/After NISAR radar sweep, AI base ranking, and a `+250m` flood surge simulator).*

---

## 🌍 1. The Problem & Radar Physics (Physical Drivers)

During monsoon floods in the **Koshi River Basin (Terai Plain, Nepal)**, thick storm clouds block normal optical satellites for days. Rescue teams cannot see which villages are trapped or which roads are under water during the critical first 72 hours.

### 📡 How NISAR L-Band Radar Solves This:

* **Sees Through Clouds Day & Night:** Unlike passive optical cameras, **NISAR** is an active microwave radar operating at **L-band (**$\approx 24\text{ cm}$ **wavelength)** that penetrates heavy monsoon clouds, smoke, and rain.

* **Surface Roughness & Dielectric Constant:** Rough dry land and vegetation scatter radar energy back to the satellite (bright pixels). Calm standing floodwater has a high dielectric constant and a smooth surface, causing **specular reflection** (radar pulses bounce away from the sensor like a mirror). This creates sharp dark zones ($\sigma^0 < -17.5\text{ dB}$), allowing our system to separate **Deep Water (Boats Only)** from **Shallow Floodwater**.

* **Multi-Temporal Change Detection:** By comparing baseline radar scans (**August 19, 2026**) against peak flood scans (**August 31, 2026**) from **NISAR GCOV** data, we isolate $35.86\text{ km}^2$ of active flood inundation.

## 🚀 2. Key Features

* **🎯 Priority Village Ranking:** Combines NISAR flood zones with OpenStreetMap settlement data to rank the top 5 at-risk villages by trapped population rather than raw water area alone.

* **🤖 AI Rescue Base Selection:** Uses a Multi-Criteria Decision Analysis (MCDA) model to score 5 safe rescue bases out of `100` based on fastest dry road time ($45\%$), facility capability ($35\%$), and route safety ($20\%$).

* **⚠️ Predictive Flood Expansion Simulator:** A live slider (`+0m` to `+300m`) that predicts which roads will flood next if water rises and **automatically moves the boat drop-off point** (Adaptive Staging Relocation) to the nearest dry high ground.

* **📦 3-Day Emergency Supply Calculator:** Automatically calculates clean drinking water ($3\text{ L/person/day}$), ready-to-eat meals, family tents, supply trucks, and rescue boats based on international Sphere humanitarian standards.

* **📋 Offline Field Map & GPS Export:** Generates a standalone `.html` map that works without internet, `.geojson` GPS tracks, and a `.md` mission summary (SitRep) for field responders.

## 📂 3. Repository Structure

```text
eye_from_above/
├── index.html                    # Interactive 7-Slide Presentation (GitHub Pages Demo)
├── app.py                        # Main Streamlit interactive dashboard
├── analyze_disaster.py           # SAR backscatter & OSM road network processing pipeline
├── fix_zones.py                  # Spatial deduplication for 5 villages & 5 rescue hubs
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── data/
    ├── disaster_summary.json     # Processed village, hub, and flood statistics
    ├── nepal_peak_flood_zones.geojson
    ├── deep_flood_zones.geojson
    ├── flooded_roads.geojson
    └── safe_road_network.graphml # Verified dry road graph for emergency routing
```

## ⚡ 4. How to Run Locally

1. **Clone the repository:**

   ```bash
   git clone https://github.com/YOUR_USERNAME/eye-from-above.git
   cd eye-from-above
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the dashboard:**

   ```bash
   streamlit run app.py
   ```

## 👥 5. Team Members (EYE FROM ABOVE)

* **Ahmed Saber** — SAR Data Science, AI Routing & Geospatial Backend Lead

* **Mohamed Haitham** — Interactive Dashboard UI/UX, Logistics Modeling & Visual Storytelling Lead

## 📚 6. Data Sources, Tools & AI Disclosure

* **Satellite Data:** NASA-ISRO SAR (NISAR) L-Band GCOV Backscatter Imagery accessed via **NASA Alaska Satellite Facility (ASF) Vertex**.

* **GIS & Road Infrastructure:** OpenStreetMap (OSM) via Overpass API, Esri World Imagery.

* **Tech Stack:** Python, Streamlit, GeoPandas, Rasterio, OSMnx, NetworkX, Folium, Shapely, Matplotlib, NumPy, Leaflet.js, Tailwind CSS.

* **AI Disclosure:** Includes a custom Multi-Criteria Decision Analysis (MCDA) AI ranking engine for rescue hub selection; AI assistance was utilized for code optimization and documentation in compliance with NASA Space Apps guidelines.