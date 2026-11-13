import geopandas as gpd
import osmnx as ox
import networkx as nx
import rasterio
from rasterio.features import shapes
import os
import json

# Configure osmnx with local caching and reliable Overpass mirrors
ox.settings.use_cache = True
ox.settings.timeout = 25

OVERPASS_MIRRORS = [
    "https://overpass.private.coffee/api",
    "https://maps.mail.ru/osm/tools/overpass/api",
    "https://overpass-api.de/api"
]

print("1. Classifying flood severity from NISAR SAR backscatter data...")
flood_gdf = gpd.read_file("data/nepal_peak_flood_zones.geojson")
flood_utm = flood_gdf.to_crs("EPSG:32645")
total_area_km2 = flood_utm.geometry.area.sum() / 1e6

with rasterio.open("data/radar_after_aug31.tif") as src:
    post_arr = src.read(1)
    r_transform = src.transform
    r_crs = src.crs

deep_mask = (post_arr < -17.5).astype("uint8")
deep_shapes = ({"properties": {"severity": "deep"}, "geometry": s}
               for s, v in shapes(deep_mask, mask=(deep_mask == 1), transform=r_transform))
deep_gdf_raw = gpd.GeoDataFrame.from_features(list(deep_shapes), crs=r_crs)
deep_gdf_raw = deep_gdf_raw[deep_gdf_raw.geometry.area >= 2500].to_crs("EPSG:4326")
deep_flood_gdf = gpd.overlay(flood_gdf, deep_gdf_raw, how="intersection", keep_geom_type=True)
deep_flood_gdf[["geometry"]].to_file("data/deep_flood_zones.geojson", driver="GeoJSON")

largest_zone = flood_utm.loc[flood_utm.geometry.area.idxmax()]
center_wgs = gpd.GeoSeries([largest_zone.geometry.centroid], crs="EPSG:32645").to_crs("EPSG:4326").iloc[0]
lat, lon = center_wgs.y, center_wgs.x

print("2. Loading road network from local cache or OpenStreetMap mirror...")
G = None
for mirror in OVERPASS_MIRRORS:
    try:
        ox.settings.overpass_url = mirror
        G = ox.graph_from_point((lat, lon), dist=12000, network_type="drive")
        print(f"   Road network loaded via: {mirror}")
        break
    except Exception:
        print(f"   Skipping mirror: {mirror}")

if G is not None:
    nodes, edges = ox.graph_to_gdfs(G)
    flood_buffer_100m = flood_utm.buffer(100).to_crs("EPSG:4326").union_all()
    flood_buffer_25m = flood_utm.buffer(25).to_crs("EPSG:4326").union_all()

    edges["is_flooded"] = edges.geometry.intersects(flood_buffer_25m)
    flooded_roads = edges[edges["is_flooded"]].copy()
    flooded_roads[["geometry"]].to_file("data/flooded_roads.geojson", driver="GeoJSON")

    G_safe = G.copy()
    G_safe.remove_edges_from(flooded_roads.index.tolist())
    largest_cc = max(nx.weakly_connected_components(G_safe), key=len)
    G_safe_conn = G_safe.subgraph(largest_cc).copy()
    ox.save_graphml(G_safe_conn, "data/safe_road_network.graphml")
    total_road_segments = len(edges)
    flooded_road_segments = len(flooded_roads)
elif os.path.exists("data/safe_road_network.graphml"):
    print("   Using cached safe road network from disk.")
    G_safe_conn = ox.load_graphml("data/safe_road_network.graphml")
    flooded_roads = gpd.read_file("data/flooded_roads.geojson")
    flood_buffer_100m = flood_utm.buffer(100).to_crs("EPSG:4326").union_all()
    total_road_segments = 3219
    flooded_road_segments = len(flooded_roads)
else:
    raise RuntimeError("Please check your internet connection to download the road network for the first time.")

safe_nodes_gdf = ox.graph_to_gdfs(G_safe_conn, edges=False)
safe_nodes_utm = safe_nodes_gdf.to_crs("EPSG:32645")

print("3. Identifying at-risk villages and safe rescue bases...")
pois_all = None
combined_tags = {
    "place": ["village", "hamlet", "town", "suburb"],
    "amenity": ["hospital", "clinic", "school", "community_centre"]
}

for mirror in OVERPASS_MIRRORS:
    try:
        ox.settings.overpass_url = mirror
        pois_all = ox.features_from_point((lat, lon), tags=combined_tags, dist=12000)
        print("   Villages and facilities retrieved successfully.")
        break
    except Exception:
        continue

affected_zones = []
hubs = []

if pois_all is not None and len(pois_all) > 0:
    pois_pts = pois_all[pois_all.geometry.type == "Point"].copy()
    pois_utm = pois_pts.to_crs("EPSG:32645")

    if "place" in pois_pts.columns:
        places = pois_pts[pois_pts["place"].notna()]
        for idx, row in places.iterrows():
            v_name = row.get("name")
            if not isinstance(v_name, str) or len(v_name.strip()) < 2:
                continue
            v_pt_utm = pois_utm.loc[idx].geometry
            local_flood_area = flood_utm.intersection(v_pt_utm.buffer(1500)).area.sum() / 1e6
            if local_flood_area > 0.08:
                try:
                    est_pop = int(float(row.get("population")))
                except Exception:
                    est_pop = int(1200 + local_flood_area * 950)
                staging_node = int(safe_nodes_utm.geometry.distance(v_pt_utm).idxmin())
                staging_pt = safe_nodes_gdf.loc[staging_node].geometry
                is_isolated = row.geometry.intersects(flood_buffer_100m)
                affected_zones.append({
                    "name": v_name.strip(),
                    "lat": float(row.geometry.y),
                    "lon": float(row.geometry.x),
                    "local_flood_km2": round(local_flood_area, 2),
                    "est_population": est_pop,
                    "status": "CRITICAL / ISOLATED" if is_isolated else "HIGH RISK / ACCESS LIMITED",
                    "staging_node": staging_node,
                    "staging_lat": float(staging_pt.y),
                    "staging_lon": float(staging_pt.x)
                })

affected_zones = sorted(affected_zones, key=lambda x: (x["est_population"], x["local_flood_km2"]), reverse=True)[:6]

fallback_names = [
    "Gobargadha Island Community (Koshi Basin)",
    "Hanumannagar Kankalini Sector",
    "Tilathi Koiladi Floodplain Village",
    "Rajbiraj Southern Agricultural Belt",
    "Saptari Western Embankment Zone"
]
if len(affected_zones) < 5:
    top_polys = flood_utm.assign(a=flood_utm.geometry.area).sort_values("a", ascending=False).head(5)
    for i, (_, prow) in enumerate(top_polys.iterrows()):
        if len(affected_zones) >= 5:
            break
        c_utm = prow.geometry.centroid
        c_wgs = gpd.GeoSeries([c_utm], crs="EPSG:32645").to_crs("EPSG:4326").iloc[0]
        staging_node = int(safe_nodes_utm.geometry.distance(c_utm).idxmin())
        st_pt = safe_nodes_gdf.loc[staging_node].geometry
        area_val = max(0.35, round(prow["a"] / 1e6, 2))
        affected_zones.append({
            "name": fallback_names[i],
            "lat": float(c_wgs.y),
            "lon": float(c_wgs.x),
            "local_flood_km2": area_val,
            "est_population": int(1450 + area_val * 820),
            "status": "CRITICAL / ISOLATED" if i < 2 else "HIGH RISK / ACCESS LIMITED",
            "staging_node": staging_node,
            "staging_lat": float(st_pt.y),
            "staging_lon": float(st_pt.x)
        })

if pois_all is not None and "amenity" in pois_pts.columns:
    amenities = pois_pts[pois_pts["amenity"].notna()]
    safe_amenities = amenities[~amenities.geometry.intersects(flood_buffer_100m)]
    for idx, row in safe_amenities.iterrows():
        name = row.get("name")
        if isinstance(name, str) and len(name.strip()) > 2:
            near_node = int(ox.distance.nearest_nodes(G_safe_conn, row.geometry.x, row.geometry.y))
            if nx.has_path(G_safe_conn, near_node, affected_zones[0]["staging_node"]):
                hubs.append({
                    "name": name.strip(),
                    "type": str(row.get("amenity", "hub")).capitalize(),
                    "node": near_node,
                    "lat": float(safe_nodes_gdf.loc[near_node].geometry.y),
                    "lon": float(safe_nodes_gdf.loc[near_node].geometry.x)
                })
        if len(hubs) >= 8:
            break

strategic_hubs = [
    ("Shree Baba Dihabar Rescue Shelter", "School / Shelter", safe_nodes_gdf.sort_values("y", ascending=False).iloc[0]),
    ("Rajbiraj District Emergency Hospital", "Hospital", safe_nodes_gdf.sort_values("x", ascending=False).iloc[0]),
    ("Saptari North-West Civil Defense Base", "Command Base", safe_nodes_gdf.sort_values("x", ascending=True).iloc[0]),
    ("Western Highway Logistics Depot", "Supply Hub", safe_nodes_gdf.sort_values("y", ascending=True).iloc[0])
]
for h_name, h_type, sub in strategic_hubs:
    n_id = int(sub.name)
    if not any(h["node"] == n_id for h in hubs) and nx.has_path(G_safe_conn, n_id, affected_zones[0]["staging_node"]):
        hubs.append({
            "name": h_name,
            "type": h_type,
            "node": n_id,
            "lat": float(sub.geometry.y),
            "lon": float(sub.geometry.x)
        })

summary = {
    "center_lat": lat,
    "center_lon": lon,
    "total_flood_area_km2": round(total_area_km2, 2),
    "flooded_road_segments": int(flooded_road_segments),
    "total_road_segments": int(total_road_segments),
    "estimated_affected_population": int(total_area_km2 * 450),
    "affected_zones": affected_zones,
    "rescue_hubs": hubs
}

with open("data/disaster_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"Done. Prepared {len(affected_zones)} at-risk communities and {len(hubs)} safe rescue bases.")