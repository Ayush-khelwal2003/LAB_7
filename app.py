"""
Lab 7: Proximity Search API
CS559: Computer Systems Design
Author: Ayush

API Specification:
  - Route: IP/search/ (supporting /search/, /search, /P/search/, /P/search)
  - Method: GET, POST
  - Fields: lat, long, cat, rad, link
  - Output: 10 recommended location IDs matching category within radius,
            ordered by grid traversal distance (shortest path on road network).
"""

import csv
import heapq
import json
import math
import os
import sys
import time
from collections import defaultdict
from io import StringIO

import numpy as np
from flask import Flask, request, jsonify, Response

app = Flask(__name__)

# ── Global In-Memory Data Structures ─────────────────────────────────────────
locations = {}           # id -> (lat, lon, category)
coords_by_id = {}        # id -> (lat, lon)
categories_by_id = {}    # id -> category
category_coords = {}     # category_name -> np.ndarray of shape (N, 2)
category_ids = {}        # category_name -> np.ndarray of shape (N,)
categories_list = []

DATA_LOADED = False
LOADED_FILENAME = ""


def find_dataset_file(custom_path=None):
    """Find the dataset file from various common names/locations."""
    candidates = []
    if custom_path:
        candidates.append(custom_path)
    env_path = os.environ.get("LOCATIONS_CSV")
    if env_path:
        candidates.append(env_path)

    # Local directory candidate filenames
    dir_path = os.path.dirname(os.path.abspath(__file__))
    filenames = [
        "locations - locations.csv",
        "locations.csv",
        "locations-locations.csv",
        "test_locations.csv",
    ]
    for fn in filenames:
        candidates.append(os.path.join(dir_path, fn))
        candidates.append(fn)

    for c in candidates:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    return None


def load_locations(filepath=None):
    """Load dataset into memory and construct fast spatial and categorical indices."""
    global locations, coords_by_id, categories_by_id
    global category_coords, category_ids, categories_list
    global DATA_LOADED, LOADED_FILENAME

    target_file = find_dataset_file(filepath)
    if not target_file:
        print("[WARN] locations CSV file not found in current directory.")
        return False

    print(f"[INFO] Loading locations from {target_file} ...")
    t0 = time.time()

    locations.clear()
    coords_by_id.clear()
    categories_by_id.clear()
    category_coords.clear()
    category_ids.clear()

    cat_coords_builder = defaultdict(list)
    cat_ids_builder = defaultdict(list)

    with open(target_file, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or len(row) < 4:
                continue
            try:
                loc_id = int(row[0].strip())
                lat = float(row[1].strip())
                lon = float(row[2].strip())
                cat = row[3].strip().lower()

                locations[loc_id] = (lat, lon, cat)
                coords_by_id[loc_id] = (lat, lon)
                categories_by_id[loc_id] = cat

                cat_coords_builder[cat].append([lat, lon])
                cat_ids_builder[cat].append(loc_id)
            except (ValueError, IndexError):
                continue

    # Convert per-category data into contiguous NumPy arrays for vectorized operations
    for cat in cat_coords_builder:
        category_coords[cat] = np.array(cat_coords_builder[cat], dtype=np.float64)
        category_ids[cat] = np.array(cat_ids_builder[cat], dtype=np.int64)

    categories_list = sorted(list(category_coords.keys()))
    DATA_LOADED = True
    LOADED_FILENAME = os.path.basename(target_file)
    elapsed = time.time() - t0
    print(f"[INFO] Successfully loaded {len(locations)} locations across {len(categories_list)} categories in {elapsed*1000:.2f}ms")
    return True


# ── Linkage Graph Parsing ───────────────────────────────────────────────────
def parse_linkage(link_text):
    """
    Parse road linkage information into an adjacency list representation.
    Each line in link_text has format: 'a b'
    Where a and b are integer IDs of points on the grid.
    Edge weights represent Euclidean distance between coordinates of a and b.
    """
    graph = defaultdict(list)
    if not link_text or not link_text.strip():
        return graph

    for line in link_text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            a = int(parts[0])
            b = int(parts[1])
            if a == b:
                continue

            # Compute Euclidean distance between grid nodes
            if a in coords_by_id and b in coords_by_id:
                la, lo_a = coords_by_id[a]
                lb, lo_b = coords_by_id[b]
                w = math.hypot(la - lb, lo_a - lo_b)
            else:
                w = 1.0 / 99.0  # default grid step

            graph[a].append((b, w))
            graph[b].append((a, w))
        except ValueError:
            continue

    return graph


# ── Shortest Path on Grid Graph (Dijkstra) ───────────────────────────────────
def dijkstra_search(graph, query_lat, query_lon, candidate_ids, radius):
    """
    Perform Dijkstra's algorithm on the grid graph to rank candidate locations
    by actual traversal distance.

    Features:
      - Snapping: Automatically maps the query point to the nearest grid/road node.
      - Early Stopping: Terminates as soon as 10 targets are confirmed to have
        minimal traversal distance.
      - Fallback: Unreachable candidates in the radius are appended deterministically.
    """
    if not graph:
        # No graph provided: pure Euclidean distance
        scored = []
        for cid in candidate_ids:
            clat, clon = coords_by_id[cid]
            d = math.hypot(query_lat - clat, query_lon - clon)
            scored.append((d, cid))
        scored.sort(key=lambda x: (x[0], x[1]))
        return [cid for _, cid in scored[:10]]

    # Step 1: Find best starting node in graph
    # Check if (query_lat, query_lon) exactly corresponds to a grid coordinate
    gi = int(round(query_lat * 99))
    gj = int(round(query_lon * 99))
    exact_id = gi * 100 + gj + 1

    pq = []
    dist = {}

    if exact_id in graph and exact_id in coords_by_id:
        elat, elon = coords_by_id[exact_id]
        if math.hypot(query_lat - elat, query_lon - elon) < 1e-5:
            # Exact match on grid node
            dist[exact_id] = 0.0
            pq.append((0.0, exact_id))

    if not pq:
        # Find the closest node in the graph to (query_lat, query_lon)
        best_node = None
        best_d = float("inf")
        for node in graph:
            if node in coords_by_id:
                nlat, nlon = coords_by_id[node]
                d = math.hypot(query_lat - nlat, query_lon - nlon)
                if d < best_d:
                    best_d = d
                    best_node = node

        if best_node is not None:
            dist[best_node] = best_d
            pq.append((best_d, best_node))
        else:
            # Graph has no valid location nodes
            scored = []
            for cid in candidate_ids:
                clat, clon = coords_by_id[cid]
                d = math.hypot(query_lat - clat, query_lon - clon)
                scored.append((d, cid))
            scored.sort(key=lambda x: (x[0], x[1]))
            return [cid for _, cid in scored[:10]]

    # Step 2: Run Dijkstra with early stopping
    target_set = set(candidate_ids)
    found = {}  # cid -> traversal_distance

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue

        if u in target_set and u not in found:
            found[u] = d
            if len(found) >= 10:
                # Early termination condition:
                # Since edge weights are non-negative, the 10th smallest distance
                # among found targets cannot be beaten by any future node with d >= d10.
                d10 = sorted(found.values())[9]
                if d >= d10:
                    break

        for v, w in graph[u]:
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))

    # Step 3: Rank candidates by traversal distance
    scored = []
    for cid in candidate_ids:
        if cid in found:
            scored.append((found[cid], cid))
        else:
            # Not reachable via graph: assign large penalty + Euclidean distance
            clat, clon = coords_by_id[cid]
            euc = math.hypot(query_lat - clat, query_lon - clon)
            scored.append((1000.0 + euc, cid))

    scored.sort(key=lambda x: (x[0], x[1]))
    return [cid for _, cid in scored[:10]]


# ── Core Proximity Search Function ───────────────────────────────────────────
def perform_proximity_search(lat, lon, category, radius, link_text=""):
    """
    1. Filter locations by requested category.
    2. Filter locations within circular radius (Euclidean distance <= radius).
    3. Re-rank locations by grid traversal distance using Dijkstra.
    4. Return top 10 location IDs.
    """
    if not DATA_LOADED:
        load_locations()

    cat_clean = category.strip().lower()
    if cat_clean not in category_coords:
        return []

    # Vectorized Euclidean distance filtering for radius
    coords = category_coords[cat_clean]
    ids = category_ids[cat_clean]
    query_point = np.array([lat, lon], dtype=np.float64)

    diffs = coords - query_point
    euc_dists = np.sqrt(np.sum(diffs ** 2, axis=1))

    mask = euc_dists <= radius
    cand_ids = ids[mask].tolist()

    if not cand_ids:
        return []

    # Graph traversal re-ranking
    graph = parse_linkage(link_text)
    return dijkstra_search(graph, lat, lon, cand_ids, radius)


# ── Request Parameter Extraction Helper ──────────────────────────────────────
def extract_request_data(req):
    """Robustly extract lat, long, cat, rad, link from any request format."""
    # 1. Gather all potential key-value sources
    params = {}
    if req.args:
        params.update(req.args.to_dict())
    if req.form:
        params.update(req.form.to_dict())
    if req.is_json:
        json_data = req.get_json(silent=True)
        if isinstance(json_data, dict):
            params.update(json_data)

    def get_val(*keys, default=None):
        for k in keys:
            if k in params and params[k] is not None:
                return params[k]
        return default

    # Latitude
    try:
        lat = float(get_val("lat", "latitude", default=0.0))
    except (ValueError, TypeError):
        lat = 0.0

    # Longitude
    try:
        lon = float(get_val("long", "lon", "lng", "longitude", default=0.0))
    except (ValueError, TypeError):
        lon = 0.0

    # Category
    cat = str(get_val("cat", "category", default="")).strip()

    # Radius
    try:
        rad = float(get_val("rad", "radius", default=0.0))
    except (ValueError, TypeError):
        rad = 0.0

    # Linkage information
    link_text = ""
    # Check if a file was uploaded under 'link' or any key
    if req.files:
        if "link" in req.files:
            link_text = req.files["link"].read().decode("utf-8", errors="ignore")
        else:
            for k in req.files:
                link_text = req.files[k].read().decode("utf-8", errors="ignore")
                break

    if not link_text:
        raw_link = get_val("link", "linkage", default="")
        if isinstance(raw_link, str):
            raw_link_str = raw_link.strip()
            # If it's a file path that exists, read it
            if raw_link_str and not "\n" in raw_link_str and len(raw_link_str) < 300 and os.path.isfile(raw_link_str):
                try:
                    with open(raw_link_str, "r", encoding="utf-8", errors="ignore") as f:
                        link_text = f.read()
                except Exception:
                    link_text = raw_link_str
            else:
                link_text = raw_link_str

    # Output formatting flag (format=dict or format=list)
    fmt = str(get_val("format", "fmt", default="list")).strip().lower()

    return lat, lon, cat, rad, link_text, fmt


# ── API Route Handlers ───────────────────────────────────────────────────────
@app.route("/search/", methods=["GET", "POST"])
@app.route("/search", methods=["GET", "POST"])
@app.route("/P/search/", methods=["GET", "POST"])
@app.route("/P/search", methods=["GET", "POST"])
def search_endpoint():
    """
    Main Proximity Search API Endpoint.
    Accepts: (lat, long, cat, rad, link)
    Returns: 10 recommended location IDs.
    """
    lat, lon, cat, rad, link_text, fmt = extract_request_data(request)

    t0 = time.time()
    results = perform_proximity_search(lat, lon, cat, rad, link_text)
    elapsed = time.time() - t0

    print(f"[QUERY] lat={lat}, lon={lon}, cat='{cat}', rad={rad}, "
          f"links={len(link_text.splitlines())} -> {len(results)} results in {elapsed*1000:.2f}ms")

    # If dict format is explicitly requested
    if fmt in ("dict", "object", "verbose", "json_object"):
        return jsonify({
            "status": "success",
            "results": results,
            "count": len(results),
            "query": {
                "lat": lat,
                "long": lon,
                "cat": cat,
                "rad": rad
            },
            "time_ms": round(elapsed * 1000, 2)
        })

    # Default: Return 10 recommended location IDs directly as a JSON list
    return jsonify(results)


@app.route("/", methods=["GET"])
def home():
    """Interactive Documentation and Test Console."""
    if not DATA_LOADED:
        load_locations()

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lab 7: Proximity Search API - CS559</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: #0f172a;
            color: #f8fafc;
            margin: 0;
            padding: 30px;
        }}
        .container {{
            max-width: 860px;
            margin: 0 auto;
            background: #1e293b;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
        }}
        h1 {{ color: #38bdf8; margin-top: 0; }}
        h2 {{ color: #94a3b8; font-size: 1.2rem; border-bottom: 1px solid #334155; padding-bottom: 8px; }}
        .badge {{
            display: inline-block;
            background: #059669;
            color: white;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: bold;
        }}
        .form-group {{
            margin-bottom: 15px;
        }}
        label {{
            display: block;
            margin-bottom: 5px;
            color: #cbd5e1;
            font-size: 0.9rem;
        }}
        input, select, textarea {{
            width: 100%;
            padding: 10px;
            background: #0f172a;
            border: 1px solid #334155;
            color: #f8fafc;
            border-radius: 6px;
            box-sizing: border-box;
            font-size: 0.95rem;
        }}
        button {{
            background: #0284c7;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1rem;
            font-weight: bold;
            transition: background 0.2s;
        }}
        button:hover {{ background: #0369a1; }}
        pre {{
            background: #090d16;
            padding: 15px;
            border-radius: 6px;
            overflow-x: auto;
            color: #4ade80;
            border: 1px solid #1e293b;
        }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }}
        code {{ color: #f43f5e; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📍 Lab 7: Proximity Search API</h1>
        <p><strong>Course:</strong> CS559 Computer Systems Design | <strong>Dataset:</strong> <span class="badge">{len(locations)} locations</span> ({LOADED_FILENAME})</p>
        <p>Available Categories: <code>{', '.join(categories_list)}</code></p>

        <h2>Test the Search Endpoint</h2>
        <form id="searchForm">
            <div class="grid">
                <div class="form-group">
                    <label>Current Latitude (lat):</label>
                    <input type="number" step="any" id="lat" value="0.5" required>
                </div>
                <div class="form-group">
                    <label>Current Longitude (long):</label>
                    <input type="number" step="any" id="long" value="0.5" required>
                </div>
            </div>
            <div class="grid">
                <div class="form-group">
                    <label>Category (cat):</label>
                    <select id="cat">
                        {"".join(f'<option value="{c}">{c}</option>' for c in categories_list)}
                    </select>
                </div>
                <div class="form-group">
                    <label>Radius (rad):</label>
                    <input type="number" step="any" id="rad" value="0.2" required>
                </div>
            </div>
            <div class="form-group">
                <label>Road Linkage Info (link - optional text lines 'a b'):</label>
                <textarea id="link" rows="4" placeholder="5051 5052&#10;5052 5053"></textarea>
            </div>
            <button type="button" onclick="runSearch()">Execute Search</button>
        </form>

        <h2>API Response</h2>
        <pre id="output">Results will appear here...</pre>

        <h2>Quick cURL Example</h2>
        <pre>curl -X POST "http://localhost:5000/search/" \\
     -F "lat=0.5" \\
     -F "long=0.5" \\
     -F "cat=restaurant" \\
     -F "rad=0.2" \\
     -F "link=@links.txt"</pre>
    </div>

    <script>
        async function runSearch() {{
            const lat = document.getElementById('lat').value;
            const long = document.getElementById('long').value;
            const cat = document.getElementById('cat').value;
            const rad = document.getElementById('rad').value;
            const link = document.getElementById('link').value;

            const out = document.getElementById('output');
            out.innerText = "Querying API...";

            try {{
                const res = await fetch(`/search/?lat=${{lat}}&long=${{long}}&cat=${{cat}}&rad=${{rad}}`, {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ link: link }})
                }});
                const data = await res.json();
                out.innerText = JSON.stringify(data, null, 2);
            }} catch (err) {{
                out.innerText = "Error: " + err.message;
            }}
        }}
    </script>
</body>
</html>"""
    return html_content


# ── Server Entry Point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else None
    load_locations(csv_file)
    port = int(os.environ.get("PORT", 5000))
    print(f"[INFO] Server starting on http://0.0.0.0:{port} ...")
    app.run(host="0.0.0.0", port=port, debug=False)
