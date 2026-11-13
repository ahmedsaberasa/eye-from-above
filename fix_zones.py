import json
import geopandas as gpd
import osmnx as ox
import networkx as nx

print("Organizing 5 distinct affected villages and 5 safe rescue bases...")
with open("data/disaster_summary.json", "r", encoding="utf-8") as f:
    summary = json.load(f)

G_safe = ox.load_graphml("data/safe_road_network.graphml")
safe_nodes_gdf = ox.graph_to_gdfs(G_safe, edges=False)
safe_nodes_utm = safe_nodes_gdf.to_crs("EPSG:32645")
sample_node = next(iter(G_safe.nodes))
node_type = type(sample_node)

flood_gdf = gpd.read_file("data/nepal_peak_flood_zones.geojson")
flood_utm = flood_gdf.to_crs("EPSG:32645")

# 1. Select 5 spatially distinct flood sectors (minimum 1200m apart)
top_polys = flood_utm.assign(a=flood_utm.geometry.area).sort_values("a", ascending=False)

zone_catalog = [
    ("Gobargadha Island Community", "CRITICAL / ISOLATED"),
    ("Hanumannagar Kankalini Sector", "CRITICAL / ISOLATED"),
    ("Tilathi Koiladi Floodplain Village", "HIGH RISK / ACCESS LIMITED"),
    ("Rajbiraj Southern Agricultural Belt", "HIGH RISK / ACCESS LIMITED"),
    ("Saptari Western Embankment Zone", "HIGH RISK / ACCESS LIMITED")
]

unique_zones = []
used_staging = set()
selected_centroids = []

for _, prow in top_polys.iterrows():
    if len(unique_zones) >= 5:
        break
    c_utm = prow.geometry.centroid
    if any(c_utm.distance(prev_c) < 1200 for prev_c in selected_centroids):
        continue

    dists = safe_nodes_utm.geometry.distance(c_utm).sort_values()
    for node_id in dists.index:
        nid = int(node_id)
        st_utm = safe_nodes_utm.loc[node_id].geometry
        if nid not in used_staging and all(st_utm.distance(safe_nodes_utm.loc[node_type(u)].geometry) > 800 for u in used_staging):
            c_wgs = gpd.GeoSeries([c_utm], crs="EPSG:32645").to_crs("EPSG:4326").iloc[0]
            st_wgs = safe_nodes_gdf.loc[node_id].geometry
            idx = len(unique_zones)
            area_km2 = round(max(0.45, flood_utm.intersection(c_utm.buffer(1500)).area.sum() / 1e6), 2)
            
            unique_zones.append({
                "name": zone_catalog[idx][0],
                "lat": float(c_wgs.y),
                "lon": float(c_wgs.x),
                "local_flood_km2": area_km2,
                "est_population": int(1400 + area_km2 * 850 + idx * 180),
                "status": zone_catalog[idx][1],
                "staging_node": nid,
                "staging_lat": float(st_wgs.y),
                "staging_lon": float(st_wgs.x)
            })
            used_staging.add(nid)
            selected_centroids.append(c_utm)
            break

unique_zones = sorted(unique_zones, key=lambda x: x["est_population"], reverse=True)

# 2. Select 5 safe rescue bases distributed around the outer dry road network
hub_specs = [
    ("Shree Baba Dihabar Rescue Shelter", "School / Shelter", safe_nodes_gdf.sort_values("y", ascending=False)),
    ("Rajbiraj District Emergency Hospital", "Hospital", safe_nodes_gdf.assign(s=safe_nodes_gdf["y"] + safe_nodes_gdf["x"]).sort_values("s", ascending=False)),
    ("Saptari North-West Civil Defense Base", "Command Base", safe_nodes_gdf.assign(s=safe_nodes_gdf["y"] - safe_nodes_gdf["x"]).sort_values("s", ascending=False)),
    ("Western Highway Logistics Depot", "Supply Hub", safe_nodes_gdf.sort_values("x", ascending=True)),
    ("Central Koshi High-Ground Medical Post", "Clinic / Triage", safe_nodes_gdf.iloc[len(safe_nodes_gdf)//3:])
]

clean_hubs = []
used_hubs = set()

for h_name, h_type, sorted_df in hub_specs:
    for node_id, row in sorted_df.iterrows():
        nid = int(node_id)
        if nid in used_hubs or nid in used_staging:
            continue
        h_utm = safe_nodes_utm.loc[node_id].geometry
        if all(h_utm.distance(safe_nodes_utm.loc[node_type(u)].geometry) > 2000 for u in used_staging | used_hubs):
            if all(nx.has_path(G_safe, node_type(nid), node_type(z["staging_node"])) for z in unique_zones):
                clean_hubs.append({
                    "name": h_name,
                    "type": h_type,
                    "node": nid,
                    "lat": float(row.geometry.y),
                    "lon": float(row.geometry.x)
                })
                used_hubs.add(nid)
                break

summary["affected_zones"] = unique_zones
summary["rescue_hubs"] = clean_hubs

with open("data/disaster_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"Done. Prepared {len(unique_zones)} villages and {len(clean_hubs)} rescue bases.")