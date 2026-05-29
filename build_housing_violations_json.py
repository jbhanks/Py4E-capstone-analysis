from pathlib import Path
import json
import time
import urllib.parse
import urllib.request


DATASET_ID = "wvxf-dwi5"
BASE_URL = "https://data.cityofnewyork.us"

OUT = Path(
    "/home/james/Massive/PROJECTDATA/nyc_real_estate_data/downloads/"
    "housing_violations.json"
)
TMP = OUT.with_suffix(".json.tmp")

PAGE_SIZE = 50_000
SLEEP_SECONDS = 1.0


def get_json(url):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "nyc-real-estate-data-builder/1.0",
            "Accept": "application/json",
        },
    )

    with urllib.request.urlopen(req, timeout=180) as response:
        return json.load(response)


def resource_url(offset):
    params = urllib.parse.urlencode(
        {
            "$limit": PAGE_SIZE,
            "$offset": offset,
        }
    )
    return f"{BASE_URL}/resource/{DATASET_ID}.json?{params}"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    print("Fetching metadata", flush=True)
    metadata = get_json(f"{BASE_URL}/api/views/{DATASET_ID}")

    columns = [
        col
        for col in metadata["columns"]
        if col.get("fieldName") and not col["fieldName"].startswith(":")
    ]

    field_names = [col["fieldName"] for col in columns]

    # Keep only the columns that match the row arrays we are going to write.
    metadata = dict(metadata)
    metadata["columns"] = columns

    print(f"Writing {TMP}", flush=True)

    row_count = 0
    offset = 0
    first_row = True

    with TMP.open("w", encoding="utf-8") as f:
        f.write('{"meta":{"view":')
        json.dump(metadata, f, ensure_ascii=False)
        f.write('},"data":[\n')

        while True:
            url = resource_url(offset)
            rows = get_json(url)

            if not rows:
                break

            for row in rows:
                row_values = [row.get(field) for field in field_names]

                if not first_row:
                    f.write(",\n")

                json.dump(row_values, f, ensure_ascii=False)
                first_row = False
                row_count += 1

            print(f"rows written: {row_count:,}", flush=True)

            offset += PAGE_SIZE
            time.sleep(SLEEP_SECONDS)

        f.write("\n]}\n")

    print("Validating JSON", flush=True)

    with TMP.open("r", encoding="utf-8") as f:
        json.load(f)

    TMP.replace(OUT)

    print(f"Done: {OUT}", flush=True)
    print(f"rows: {row_count:,}", flush=True)


if __name__ == "__main__":
    main()

