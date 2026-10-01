"""Console-only observed MHW trend analysis for the 15 configured sites."""

from pathlib import Path

import pandas as pd

from mhw_repro.io import load_site_file, print_audit
from mhw_repro.mhw import annual_characteristics
from mhw_repro.trends import trend_table


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "raw"
LOCATIONS_FILE = REPO_ROOT / "configs" / "locations.csv"
MIN_EVENT_DAYS = 5

locations = pd.read_csv(LOCATIONS_FILE)
all_rows = []

print("=" * 88)
print("OBSERVED ANNUAL MHW CHARACTERISTICS AND TRENDS")
print("Climatological reference: 1983-2012 | Analysis period: 1983-2025")
print("=" * 88)

for site_number, site in locations.iterrows():
    path = DATA_DIR / site["file_name"]
    if not path.exists():
        print(f"MISSING {path.name}; site skipped")
        continue

    frame, audit = load_site_file(path)
    print(f"\n[{site_number + 1}/15] {site['location']} ({site['region']})")
    print_audit(audit)
    frame = frame[(frame["Date"] >= "1983-01-01") & (frame["Date"] <= "2025-12-31")]
    annual, events, _ = annual_characteristics(
        frame, years=range(1983, 2026), min_duration=MIN_EVENT_DAYS
    )
    site_trends = trend_table(annual)
    site_trends.insert(0, "location", site["location"])
    site_trends.insert(0, "longitude", site["longitude"])
    site_trends.insert(0, "latitude", site["latitude"])
    all_rows.append(site_trends)
    print(f"Detected events: {len(events)}")
    print(site_trends.round(6).to_string(index=False))

print("\n" + "=" * 88)
print("CONSOLIDATED TREND TABLE")
print("=" * 88)
if all_rows:
    result = pd.concat(all_rows, ignore_index=True)
    print(result.round(6).to_string(index=False))
else:
    print("No configured data files were found in data/raw/.")

