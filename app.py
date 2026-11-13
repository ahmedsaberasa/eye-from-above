import streamlit as st
import geopandas as gpd
import osmnx as ox
import networkx as nx
import folium
from streamlit_folium import st_folium
from shapely.geometry import LineString
import rasterio
import matplotlib.pyplot as plt
import json
import os
import math

st.set_page_config(
    page_title="Eye from Above | NISAR Flood Response",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #060b18; color: #e2e8f0; }
    [data-testid="stSidebar"] { background-color: #0b132b; border-right: 1px solid #1e293b; }
    .metric-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
    }
    .logistics-card {
        background: #0f172a;
        border-left: 4px solid #f0a830;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .ai-box {
        background: linear-gradient(135deg, #064e3b 0%, #0f172a 100%);
        border: 1px solid #10b981;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 12px;
    }
    .metric-value { font-size: 26px; font-weight: 800; margin: 4px 0; }
    .metric-label { font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.8px; }
    h1, h2, h3 { color: #f8fafc !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_base_data():
    with open("data/disaster_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    
    # Simplify polygons slightly for fast browser rendering
    flood_gdf = gpd.read_file("data/nepal_peak_flood_zones.geojson")[["geometry"]].copy()
    flood_gdf["geometry"] = flood_gdf.geometry.simplify(0.00015, preserve_topology=True)
    flood_json = flood_gdf.to_json()

    deep_json = None
    if os.path.exists("data/deep_flood_zones.geojson"):
        deep_gdf = gpd.read_file("data/deep_flood_zones.geojson")[["geometry"]].copy()
        if len(deep_gdf) > 0:
            deep_gdf["geometry"] = deep_gdf.geometry.simplify(0.00015, preserve_topology=True)
            deep_json = deep_gdf.to_json()

    flooded_roads = gpd.read_file("data/flooded_roads.geojson")[["geometry"]]
    roads_json = flooded_roads.to_json()
    
    return summary, flood_gdf, flood_json, deep_json, roads_json, len(flooded_roads)

@st.cache_resource
def load_safe_graph():
    G = ox.load_graphml("data/safe_road_network.graphml")
    edges_gdf = ox.graph_to_gdfs(G, nodes=False)
    nodes_gdf = ox.graph_to_gdfs(G, edges=False)
    return G, edges_gdf, nodes_gdf

@st.cache_data
def compute_expansion_scenario(_flood_gdf, _edges_gdf, buffer_m):
    if buffer_m <= 0:
        return None, 0.0, []
    flood_utm = _flood_gdf.to_crs("EPSG:32645")
    buffered_utm = flood_utm.buffer(buffer_m)
    sim_area_km2 = round(buffered_utm.area.sum() / 1e6, 2)
    
    buffered_wgs = gpd.GeoDataFrame(geometry=buffered_utm, crs="EPSG:32645").to_crs("EPSG:4326")
    buffered_wgs["geometry"] = buffered_wgs.geometry.simplify(0.00025, preserve_topology=True)
    
    union_poly = buffered_wgs.geometry.union_all()
    newly_blocked_idx = _edges_gdf[_edges_gdf.geometry.intersects(union_poly)].index.tolist()
    return buffered_wgs.to_json(), sim_area_km2, newly_blocked_idx

def compute_route_for_hub(G_emergency, safe_nodes_gdf, hub, target_zone):
    sample_node = next(iter(G_emergency.nodes))
    node_type = type(sample_node)
    orig_n = node_type(hub["node"])
    dest_n = node_type(target_zone["staging_node"])

    st_lat = target_zone["staging_lat"]
    st_lon = target_zone["staging_lon"]
    relocated = False

    if orig_n in G_emergency:
        reachable_nodes = nx.node_connected_component(G_emergency, orig_n)
        # If the original drop-off point is flooded, move to the nearest dry road node
        if dest_n not in reachable_nodes and len(reachable_nodes) > 1:
            reach_gdf = safe_nodes_gdf.loc[list(reachable_nodes)]
            dists = ((reach_gdf["y"] - target_zone["lat"])**2 + (reach_gdf["x"] - target_zone["lon"])**2)
            dest_n = dists.idxmin()
            st_lat = float(reach_gdf.loc[dest_n, "y"])
            st_lon = float(reach_gdf.loc[dest_n, "x"])
            relocated = True

    try:
        route_nodes = nx.shortest_path(G_emergency, orig_n, dest_n, weight="length")
        route_geoms = []
        lengths = []
        for u, v in zip(route_nodes[:-1], route_nodes[1:]):
            edge_data = G_emergency.get_edge_data(u, v)
            first_key = next(iter(edge_data))
            lengths.append(edge_data[first_key].get("length", 100))
            if "geometry" in edge_data[first_key]:
                route_geoms.append(edge_data[first_key]["geometry"])
            else:
                p1 = (safe_nodes_gdf.loc[u, "x"], safe_nodes_gdf.loc[u, "y"])
                p2 = (safe_nodes_gdf.loc[v, "x"], safe_nodes_gdf.loc[v, "y"])
                route_geoms.append(LineString([p1, p2]))

        route_gdf = gpd.GeoDataFrame({"length": lengths}, geometry=route_geoms, crs="EPSG:4326")
        dist_km = round(sum(lengths) / 1000.0, 2)
        eta_mins = max(1, int((dist_km / 35.0) * 60))
        return True, route_gdf, dist_km, eta_mins, st_lat, st_lon, relocated
    except Exception:
        return False, None, 999.0, 999, st_lat, st_lon, relocated

def rank_hubs_with_ai(G_emergency, safe_nodes_gdf, hubs, target_zone):
    """Score and rank safe rescue hubs from 0 to 100 based on speed, capacity, and road safety."""
    is_critical = "CRITICAL" in target_zone["status"]
    type_weights = {
        "Hospital": 98 if is_critical else 88,
        "Command Base": 95 if is_critical else 90,
        "Supply Hub": 90 if not is_critical else 85,
        "Clinic / Triage": 86,
        "School / Shelter": 78
    }
    
    evaluated = []
    for h in hubs:
        ok, r_gdf, dist_km, eta_mins, st_lat, st_lon, relocated = compute_route_for_hub(G_emergency, safe_nodes_gdf, h, target_zone)
        # 1. Travel distance & speed score (45%)
        dist_score = max(10.0, 100.0 - (dist_km * 2.4)) if ok else 0.0
        # 2. Facility capability score (35%)
        cap_score = float(type_weights.get(h["type"], 80))
        # 3. Road safety & direct access score (20%)
        resilience_score = 100.0 if (ok and not relocated) else (75.0 if ok else 0.0)
        
        ai_score = round(0.45 * dist_score + 0.35 * cap_score + 0.20 * resilience_score, 1)
        
        reasons = []
        if dist_km <= 14:
            reasons.append(f"Fastest dry road ({dist_km} km, ~{eta_mins} min drive)")
        else:
            reasons.append(f"Safe dry road ({dist_km} km, ~{eta_mins} min drive)")
        if h["type"] in ["Hospital", "Command Base", "Supply Hub"]:
            reasons.append(f"Strong {h['type'].lower()} support")
        if not relocated and ok:
            reasons.append("Direct access to boat drop-off")
            
        evaluated.append({
            **h,
            "ai_score": ai_score,
            "ai_reason": " • ".join(reasons),
            "route_ok": ok,
            "route_gdf": r_gdf,
            "dist_km": dist_km,
            "eta_mins": eta_mins,
            "st_lat": st_lat,
            "st_lon": st_lon,
            "relocated": relocated
        })
        
    return sorted(evaluated, key=lambda x: x["ai_score"], reverse=True)

summary, flood_gdf, flood_json, deep_json, roads_json, num_blocked = load_base_data()
G_safe, safe_edges_gdf, safe_nodes_gdf = load_safe_graph()

# Left Control Panel
with st.sidebar:
    st.markdown("## 🛰️ EYE FROM ABOVE")
    st.caption("Flood Rescue & Satellite Map")
    st.divider()
    
    st.markdown("### 🎯 1. Choose Flooded Village")
    zone_labels = [f"#{i+1} {z['name']} (~{z['est_population']:,} people)" for i, z in enumerate(summary["affected_zones"])]
    z_idx = st.selectbox("Select Village to Help:", range(len(zone_labels)), format_func=lambda i: zone_labels[i])
    target_zone = summary["affected_zones"][z_idx]
    
    st.markdown("### ⚠️ 2. Predict Flood Expansion")
    expansion_m = st.slider(
        "Predicted Flood Spread (Meters):",
        min_value=0, max_value=300, value=0, step=50,
        help="Predict how far floodwater could expand (+0m to +300m) and auto-adjust safe rescue routes."
    )

# Build active road network and rank rescue hubs
sim_json, sim_area_km2, extra_blocked_edges = compute_expansion_scenario(flood_gdf, safe_edges_gdf, expansion_m)
if expansion_m > 0 and len(extra_blocked_edges) > 0:
    G_active = G_safe.copy()
    G_active.remove_edges_from(extra_blocked_edges)
else:
    G_active = G_safe

G_emergency = G_active.to_undirected()
ranked_hubs = rank_hubs_with_ai(G_emergency, safe_nodes_gdf, summary["rescue_hubs"], target_zone)

with st.sidebar:
    st.markdown("### 🚑 3. Choose Rescue Base")
    auto_ai_mode = st.checkbox("✨ Pick Best Base Automatically (AI)", value=True)
    
    hub_labels = [
        f"{'⭐ Best: ' if i == 0 else ''}{h['name']} ({h['ai_score']}/100)"
        for i, h in enumerate(ranked_hubs)
    ]
    if auto_ai_mode:
        h_idx = 0
        st.selectbox("Recommended Rescue Base:", range(len(hub_labels)), index=0, format_func=lambda i: hub_labels[i], disabled=True)
    else:
        h_idx = st.selectbox("Pick a Different Base:", range(len(hub_labels)), index=0, format_func=lambda i: hub_labels[i])
        
    selected_hub = ranked_hubs[h_idx]
    
    st.markdown(f"""
    <div class="ai-box">
        <div style="font-size:12px; color:#34d399; font-weight:700;">✨ WHY THIS BASE? (Rating: {selected_hub['ai_score']}/100)</div>
        <div style="font-size:12px; color:#e2e8f0; margin-top:4px;">{selected_hub['ai_reason']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🗺️ 4. Map Layers")
    map_style = st.radio("Map View", ["Satellite View", "Street Map"], index=0)
    show_flood = st.checkbox("🌊 Flooded Area", value=True)
    show_deep = st.checkbox("🌀 Deep Water (Boats Only)", value=True)
    show_blocked = st.checkbox("🚧 Flooded Roads", value=True)
    show_all_sos = st.checkbox("🆘 All Villages in Danger", value=True)

# Extract route metrics for the selected hub
route_available = selected_hub["route_ok"]
route_gdf = selected_hub["route_gdf"]
live_route_km = selected_hub["dist_km"] if route_available else 0.0
est_eta_mins = selected_hub["eta_mins"] if route_available else 0
active_staging_lat = selected_hub["st_lat"]
active_staging_lon = selected_hub["st_lon"]
relocated_staging = selected_hub["relocated"]

active_flood_area = sim_area_km2 if expansion_m > 0 else summary["total_flood_area_km2"]
active_blocked_count = num_blocked + len(extra_blocked_edges)
pop_scale = 1.0 + (expansion_m / 600.0)
active_target_pop = int(target_zone["est_population"] * pop_scale)

# Calculate 72-hour relief supplies (Sphere standards)
water_liters_72h = active_target_pop * 3 * 3
food_rations_72h = active_target_pop * 3
family_tents = math.ceil(active_target_pop / 5)
rescue_boats = max(2, math.ceil(active_target_pop / 250))
supply_trucks = max(1, math.ceil((water_liters_72h / 1000.0 + food_rations_72h * 0.0015) / 8.0))

st.markdown("# 🛰️ Eye from Above — Flood Rescue Dashboard")
st.markdown("Find flooded areas through clouds using NASA NISAR radar, see which villages need help, and plan safe rescue routes.")

c1, c2, c3, c4 = st.columns(4)
with c1:
    delta_txt = f"+{round(active_flood_area - summary['total_flood_area_km2'], 2)} km² extra spread" if expansion_m > 0 else "Detected by NISAR Radar"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Flooded Area</div>
        <div class="metric-value" style="color: #38bdf8;">{active_flood_area} km²</div>
        <div style="font-size:11px; color:#f0a830;">{delta_txt}</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">People in Selected Village</div>
        <div class="metric-value" style="color: #f0a830;">~{active_target_pop:,}</div>
        <div style="font-size:11px; color:#64748b;">{target_zone['name']}</div>
    </div>""", unsafe_allow_html=True)

with c3:
    extra_r_txt = f"(+{len(extra_blocked_edges)} more at +{expansion_m}m)" if expansion_m > 0 else "Closed to Vehicles"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Flooded Road Sections</div>
        <div class="metric-value" style="color: #e74c3c;">{active_blocked_count}</div>
        <div style="font-size:11px; color:#64748b;">{extra_r_txt}</div>
    </div>""", unsafe_allow_html=True)

with c4:
    r_disp = f"{live_route_km} km" if route_available else "No Dry Road"
    eta_disp = f"~{est_eta_mins} min drive (New drop-off)" if relocated_staging else (f"~{est_eta_mins} min safe drive" if route_available else "Use Boats or Airlift")
    r_color = "#27ae60" if route_available else "#e74c3c"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Safe Rescue Road</div>
        <div class="metric-value" style="color: {r_color};">{r_disp}</div>
        <div style="font-size:11px; color:#64748b;">{eta_disp}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Live Rescue Map",
    "📦 Needed Supplies (3 Days)",
    "📡 Before & After Radar Scans",
    "📋 Download Offline Map & Report"
])

with tab1:
    col_map, col_info = st.columns([3, 1])
    with col_map:
        selected_tiles = "Esri.WorldImagery" if map_style == "Satellite View" else "OpenStreetMap"
        m = folium.Map(location=[target_zone["lat"], target_zone["lon"]], zoom_start=12, tiles=selected_tiles)
        
        if expansion_m > 0 and sim_json is not None:
            folium.GeoJson(
                sim_json,
                style_function=lambda x: {"fillColor": "#f59e0b", "color": "#d97706", "weight": 1.5, "fillOpacity": 0.4, "dashArray": "4, 4"},
                tooltip=f"⚠️ Extra Flood Spread (+{expansion_m}m)"
            ).add_to(m)

        if show_flood:
            folium.GeoJson(
                flood_json,
                style_function=lambda x: {"fillColor": "#38bdf8", "color": "#0284c7", "weight": 1, "fillOpacity": 0.45},
                tooltip="🌊 Flooded Land"
            ).add_to(m)

        if show_deep and deep_json is not None:
            folium.GeoJson(
                deep_json,
                style_function=lambda x: {"fillColor": "#1d4ed8", "color": "#1e3a8a", "weight": 1, "fillOpacity": 0.75},
                tooltip="🌀 Deep Water (Boats Only)"
            ).add_to(m)
            
        if show_blocked and num_blocked > 0:
            folium.GeoJson(
                roads_json,
                style_function=lambda x: {"color": "#e74c3c", "weight": 4, "opacity": 0.9, "dashArray": "5, 5"},
                tooltip="⚠️ Flooded Road (Closed)"
            ).add_to(m)

        if show_all_sos:
            for i, z in enumerate(summary["affected_zones"]):
                is_selected = (i == z_idx)
                folium.Marker(
                    [z["lat"], z["lon"]],
                    popup=f"🆘 #{i+1} {z['name']}<br>People: ~{z['est_population']:,}<br>Risk: {z['status']}",
                    icon=folium.Icon(color="red" if is_selected else "darkred", icon="exclamation-sign")
                ).add_to(m)
            
        if route_available and route_gdf is not None:
            folium.GeoJson(
                route_gdf[["geometry"]].to_json(),
                style_function=lambda x: {"color": "#27ae60", "weight": 6, "opacity": 0.95},
                tooltip=f"✅ Safe Road from {selected_hub['name']} ({live_route_km} km)"
            ).add_to(m)
            
        folium.Marker(
            [selected_hub["lat"], selected_hub["lon"]],
            popup=f"🚑 Rescue Base: {selected_hub['name']} (Rating: {selected_hub['ai_score']}/100)",
            icon=folium.Icon(color="green", icon="plus-sign")
        ).add_to(m)
        
        folium.Marker(
            [active_staging_lat, active_staging_lon],
            popup=f"🚤 {'New ' if relocated_staging else ''}Boat Drop-Off Point for {target_zone['name']}",
            icon=folium.Icon(color="orange", icon="flag")
        ).add_to(m)

        # Auto-zoom map to show rescue base, drop-off point, and village together
        bounds = [
            [min(selected_hub["lat"], active_staging_lat, target_zone["lat"]) - 0.01,
             min(selected_hub["lon"], active_staging_lon, target_zone["lon"]) - 0.01],
            [max(selected_hub["lat"], active_staging_lat, target_zone["lat"]) + 0.01,
             max(selected_hub["lon"], active_staging_lon, target_zone["lon"]) + 0.01]
        ]
        m.fit_bounds(bounds)

        offline_html_bytes = m.get_root().render().encode("utf-8")
        st_folium(m, width=None, height=550, returned_objects=[])
        
    with col_info:
        st.markdown("### 🆘 Village Details")
        st.error(f"**{target_zone['name']}**\n\n• **Risk Level:** {target_zone['status']}\n• **People at Risk:** ~{active_target_pop:,}\n• **Nearby Flood:** {target_zone['local_flood_km2']} km²")
        
        st.markdown("### 🚑 Rescue Plan")
        st.success(f"**Start Base:** {selected_hub['name']}\n\n• **Base Rating:** `{selected_hub['ai_score']}/100`\n• **Base Type:** {selected_hub['type']}")
        st.markdown(f"**Boat Drop-Off Point:** `{active_staging_lat:.4f}, {active_staging_lon:.4f}`")
        if relocated_staging:
            st.info(f"🔄 **Drop-Off Moved:** Original spot flooded at +{expansion_m}m, so the route moved to the nearest dry road.")
        if route_available:
            st.markdown(f"**Safe Road Distance:** `{live_route_km} km` (~`{est_eta_mins} min` drive)")
        
        gmaps_url = f"https://www.google.com/maps/dir/?api=1&origin={selected_hub['lat']},{selected_hub['lon']}&destination={active_staging_lat},{active_staging_lon}&travelmode=driving"
        st.link_button("📲 Open in Google Maps", gmaps_url)
        
        st.download_button(
            label="📦 Save Offline Map (.html)",
            data=offline_html_bytes,
            file_name=f"Offline_Map_{target_zone['name'].replace(' ', '_')}.html",
            mime="text/html"
        )
        
        st.divider()
        st.markdown("**Map Guide:**")
        st.markdown("🟦 **Dark Blue:** Deep Water (Boats Only)")
        st.markdown("🩵 **Light Blue:** Flooded Land")
        if expansion_m > 0:
            st.markdown(f"🟧 **Orange Dashed:** +{expansion_m}m Extra Spread")
        st.markdown("🟥 **Red Dashed:** Closed Roads")
        st.markdown("🟩 **Green Line:** Safe Rescue Road")

with tab2:
    st.markdown(f"### 📦 3-Day Emergency Supplies for {target_zone['name']}")
    st.caption(f"Sent from **{selected_hub['name']}** (Base Rating: {selected_hub['ai_score']}/100) for ~{active_target_pop:,} people.")
    
    l1, l2, l3 = st.columns(3)
    with l1:
        st.markdown(f"""
        <div class="logistics-card">
            <div class="metric-label">💧 Clean Drinking Water</div>
            <div class="metric-value" style="color: #38bdf8;">{water_liters_72h:,} Liters</div>
            <div style="font-size:12px; color:#94a3b8;">3 liters per person each day (for 3 days)</div>
        </div>
        <div class="logistics-card">
            <div class="metric-label">🚤 Rescue Boats</div>
            <div class="metric-value" style="color: #f0a830;">{rescue_boats} Boats</div>
            <div style="font-size:12px; color:#94a3b8;">Motorboats to reach flooded homes</div>
        </div>
        """, unsafe_allow_html=True)
        
    with l2:
        st.markdown(f"""
        <div class="logistics-card">
            <div class="metric-label">🍱 Ready-to-Eat Meals</div>
            <div class="metric-value" style="color: #27ae60;">{food_rations_72h:,} Meals</div>
            <div style="font-size:12px; color:#94a3b8;">3 food packs per person (for 3 days)</div>
        </div>
        <div class="logistics-card">
            <div class="metric-label">🚛 Supply Trucks</div>
            <div class="metric-value" style="color: #e2e8f0;">{supply_trucks} Trucks</div>
            <div style="font-size:12px; color:#94a3b8;">To carry water and food from {selected_hub['name']}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with l3:
        st.markdown(f"""
        <div class="logistics-card">
            <div class="metric-label">⛺ Family Tents</div>
            <div class="metric-value" style="color: #a855f7;">{family_tents:,} Tents</div>
            <div style="font-size:12px; color:#94a3b8;">1 waterproof tent for every 5 people</div>
        </div>
        <div class="logistics-card">
            <div class="metric-label">🩺 First-Aid & Clean Water Kits</div>
            <div class="metric-value" style="color: #ec4899;">{math.ceil(active_target_pop / 100)} Kits</div>
            <div style="font-size:12px; color:#94a3b8;">Medical aid and water cleaning tablets</div>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("### 📡 Satellite Radar View (Sees Through Clouds)")
    st.caption("Dark areas in the August 31 radar scan show smooth standing floodwater detected by NASA NISAR.")
    
    @st.cache_data
    def load_radar_preview():
        with rasterio.open("data/radar_before_aug19.tif") as s1, rasterio.open("data/radar_after_aug31.tif") as s2:
            pre = s1.read(1, out_shape=(s1.height // 3, s1.width // 3))
            post = s2.read(1, out_shape=(s2.height // 3, s2.width // 3))
        return pre, post

    if os.path.exists("data/radar_before_aug19.tif") and os.path.exists("data/radar_after_aug31.tif"):
        pre_img, post_img = load_radar_preview()
        fig, ax = plt.subplots(1, 2, figsize=(14, 6), facecolor="#060b18")
        ax[0].imshow(pre_img, cmap="gray", vmin=-22, vmax=-5)
        ax[0].set_title("Before Flood (Aug 19, 2026)", color="white", fontsize=13)
        ax[0].axis("off")
        ax[1].imshow(post_img, cmap="gray", vmin=-22, vmax=-5)
        ax[1].set_title("Peak Flood (Aug 31, 2026)", color="#f0a830", fontsize=13)
        ax[1].axis("off")
        plt.tight_layout()
        st.pyplot(fig)

with tab4:
    st.markdown("### 📋 Rescue Summary Report & Offline Files")
    sitrep_text = f"""# EYE FROM ABOVE — RESCUE MISSION SUMMARY
- **Village to Help:** {target_zone['name']} (~{active_target_pop:,} people | Risk: {target_zone['status']})
- **Total Flooded Area:** {active_flood_area} km² (Extra Spread Test: +{expansion_m}m)
- **Closed Roads:** {active_blocked_count} road sections under water
- **Selected Rescue Base:** {selected_hub['name']} (Rating: {selected_hub['ai_score']}/100 | {selected_hub['ai_reason']})
- **Boat Drop-Off Point:** {active_staging_lat:.5f}, {active_staging_lon:.5f} {'(Moved to higher dry road)' if relocated_staging else ''}
- **Safe Road Distance:** {live_route_km} km (~{est_eta_mins} min drive)

## 3-DAY NEEDED SUPPLIES
- **Drinking Water:** {water_liters_72h:,} Liters
- **Ready-to-Eat Meals:** {food_rations_72h:,} Meals
- **Family Tents:** {family_tents:,} Tents
- **Vehicles Needed:** {supply_trucks} Supply Trucks + {rescue_boats} Rescue Boats
"""
    st.markdown(sitrep_text)
    
    c_dl1, c_dl2, c_dl3 = st.columns(3)
    with c_dl1:
        st.download_button("📥 Download Summary (.md)", data=sitrep_text, file_name=f"Rescue_Summary_{target_zone['name'].replace(' ', '_')}.md", mime="text/markdown")
    with c_dl2:
        if route_available and route_gdf is not None:
            st.download_button("🛰️ Download GPS Route (.geojson)", data=route_gdf[["geometry"]].to_json(), file_name=f"Safe_Route_{target_zone['name'].replace(' ', '_')}.geojson", mime="application/json")
    with c_dl3:
        st.download_button("📦 Download Offline Map (.html)", data=offline_html_bytes, file_name=f"Offline_Map_{target_zone['name'].replace(' ', '_')}.html", mime="text/html")