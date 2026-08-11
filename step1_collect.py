import sys
import os
import json
import time
import requests

def fetch_chapter(chapter_num, max_retries=3, retry_delay=2):
    ch_str = f"{chapter_num:02d}"
    url = f"https://hts.usitc.gov/reststop/exportList?from={ch_str}01&to={ch_str}99&format=JSON&styles=false"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    return data
            print(f"Warning: Chapter {ch_str} returned HTTP status {response.status_code} (attempt {attempt}/{max_retries})", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Error fetching chapter {ch_str} (attempt {attempt}/{max_retries}): {e}", file=sys.stderr)
        
        if attempt < max_retries:
            time.sleep(retry_delay)

    return []

def main():
    print("Starting HTS bulk data collection by chapters (01 to 99)...")
    
    unique_records = {}
    chapter_breakdown = {}

    for ch in range(1, 100):
        ch_str = f"{ch:02d}"
        records = fetch_chapter(ch)
        count = len(records)
        chapter_breakdown[ch_str] = count

        for item in records:
            key = (item.get("htsno"), item.get("description"))
            unique_records[key] = item

        print(f"Chapter {ch_str}: {count} records returned.")

    merged_list = list(unique_records.values())

    print("\n" + "=" * 50)
    print("PER-CHAPTER BREAKDOWN:")
    print("=" * 50)
    for ch_str, count in chapter_breakdown.items():
        note = " (EMPTY/RESERVED)" if count == 0 else ""
        print(f"  Chapter {ch_str}: {count} records{note}")
    print("=" * 50)

    print(f"\nTotal unique HTS records collected across all chapters: {len(merged_list)}")

    output_filename = "hts_full.json"
    output_path = os.path.abspath(output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_list, f, indent=2, ensure_ascii=False)

    print(f"\nSuccessfully saved '{output_filename}' ({len(merged_list)} records) at:")
    print(output_path)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print("Verification SUCCESS: hts_full.json exists and is non-empty.")
    else:
        print("Verification ERROR: hts_full.json was not created properly.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
