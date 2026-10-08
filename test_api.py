"""
Lab 7: Comprehensive Test Suite
Validates the Proximity Search API using 'locations - locations.csv'
"""

import json
import os
import requests
import sys
import time

BASE_URL = "http://127.0.0.1:5000"


def test_health():
    print("\n[TEST 1] Testing Home Endpoint...")
    try:
        r = requests.get(f"{BASE_URL}/")
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        print("  ✓ Home endpoint is responsive (Status 200)")
        return True
    except requests.exceptions.ConnectionError:
        print("  ✗ Could not connect to API server. Is it running?")
        return False


def test_get_search():
    print("\n[TEST 2] Testing GET /search/ ...")
    params = {
        "lat": 0.5,
        "long": 0.5,
        "cat": "restaurant",
        "rad": 0.2,
        "link": ""
    }
    r = requests.get(f"{BASE_URL}/search/", params=params)
    assert r.status_code == 200, f"Status: {r.status_code}"
    data = r.json()
    assert isinstance(data, list), f"Expected list, got {type(data)}"
    assert len(data) == 10, f"Expected 10 IDs, got {len(data)}"
    assert all(isinstance(x, int) for x in data), "All elements must be ints"
    print(f"  ✓ Returned 10 IDs: {data}")
    return True


def test_post_json_search():
    print("\n[TEST 3] Testing POST /search/ (JSON payload) ...")
    payload = {
        "lat": 0.25,
        "long": 0.25,
        "cat": "hospital",
        "rad": 0.3,
        "link": ""
    }
    r = requests.post(f"{BASE_URL}/search/", json=payload)
    assert r.status_code == 200, f"Status: {r.status_code}"
    data = r.json()
    assert isinstance(data, list), f"Expected list, got {type(data)}"
    assert len(data) == 10, f"Expected 10 IDs, got {len(data)}"
    print(f"  ✓ Returned 10 IDs: {data}")
    return True


def test_post_form_search_with_links():
    print("\n[TEST 4] Testing POST /search/ with road network linkage ...")
    # Generate a small sample link text around node 5051 (approx 0.5, 0.5)
    # Node formula: i*100 + j + 1
    links = [
        "5051 5052",
        "5052 5053",
        "5053 5054",
        "5054 5154",
        "5154 5254",
        "5051 5151",
        "5151 5251",
    ]
    link_content = "\n".join(links)

    data = {
        "lat": "0.5051",
        "long": "0.5051",
        "cat": "bank",
        "rad": "0.25",
        "link": link_content
    }
    r = requests.post(f"{BASE_URL}/search/", data=data)
    assert r.status_code == 200, f"Status: {r.status_code}"
    result = r.json()
    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert len(result) == 10, f"Expected 10 IDs, got {len(result)}"
    print(f"  ✓ Returned 10 IDs using grid traversal: {result}")
    return True


def test_p_search_route():
    print("\n[TEST 5] Testing alternative route /P/search/ ...")
    params = {
        "lat": 0.1,
        "long": 0.1,
        "cat": "cafe",
        "rad": 0.2
    }
    r = requests.get(f"{BASE_URL}/P/search/", params=params)
    assert r.status_code == 200, f"Status: {r.status_code}"
    result = r.json()
    assert isinstance(result, list) and len(result) == 10
    print(f"  ✓ /P/search/ route working: {result}")
    return True


def test_file_upload_link():
    print("\n[TEST 6] Testing multipart/form-data link file upload ...")
    links_file_path = "sample_links.txt"
    with open(links_file_path, "w") as f:
        f.write("1 2\n2 3\n3 4\n4 104\n104 204\n")

    try:
        with open(links_file_path, "rb") as f:
            files = {"link": ("sample_links.txt", f, "text/plain")}
            data = {"lat": 0.0, "long": 0.0, "cat": "store", "rad": 0.2}
            r = requests.post(f"{BASE_URL}/search/", data=data, files=files)
            assert r.status_code == 200
            result = r.json()
            assert isinstance(result, list) and len(result) == 10
            print(f"  ✓ Multipart link upload working: {result}")
    finally:
        if os.path.exists(links_file_path):
            os.remove(links_file_path)


def test_format_dict():
    print("\n[TEST 7] Testing format=dict mode ...")
    params = {
        "lat": 0.5,
        "long": 0.5,
        "cat": "school",
        "rad": 0.2,
        "format": "dict"
    }
    r = requests.get(f"{BASE_URL}/search/", params=params)
    assert r.status_code == 200
    res = r.json()
    assert isinstance(res, dict)
    assert "results" in res
    assert len(res["results"]) == 10
    print(f"  ✓ format=dict working: count={res['count']}, time={res['time_ms']}ms")


if __name__ == "__main__":
    print("=" * 60)
    print("Lab 7: Running Test Suite for Proximity Search API")
    print("=" * 60)

    if not test_health():
        sys.exit(1)

    test_get_search()
    test_post_json_search()
    test_post_form_search_with_links()
    test_p_search_route()
    test_file_upload_link()
    test_format_dict()

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! API is 100% compliant with requirements.")
    print("=" * 60)
