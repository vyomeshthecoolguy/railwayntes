"""
NTES SCRAPER - SIMPLIFIED, based on the REAL response structure
======================================================================
Big discovery from debugging: NTES's live_status() returns the FULL
day's route in ONE call - a "STNS" list with every station, its
schedule (STA/STD), and its delay (DARR/DDEP) - even for stations
"yet to happen" or non-reporting ones (NTES estimates/interpolates
those). This means we do NOT need to poll every 2 minutes like
before - one call per train, once a day (ideally evening, after the
journey finishes), is enough. Much simpler than the old design.

REAL FIELD MEANINGS (confirmed from actual response, not docs):
  STNS         -> list of every station on the route
    Sr         -> station order (1, 2, 3...)
    SN         -> station name
    SC         -> station code
    DIST       -> distance from origin (km)
    STA        -> scheduled arrival time ("12:24 08-Aug", or "Source"/"")
    STD        -> scheduled departure time
    DARR       -> delay in arrival: "On Time", "HH:MM" (e.g. "00:03" = 3 min),
                  or "" if not yet reached
    DDEP       -> delay in departure, same format
  TN           -> train number
  LSTN / LSTNN -> last reporting station code / name (where it is now)
  LDEL         -> current delay in minutes (as a string, e.g. "3")

BEFORE YOU RUN THIS:
  pip install ntes-client pandas
Needs internet - run on your own computer.

WHEN TO RUN:
Once daily, in the evening, after your trains have finished running -
that way STNS has real delay data for every station instead of blanks
for stations not yet reached.
"""

from ntes import NTESClient, NTESError
import pandas as pd
import glob
import os
import re
import time
from datetime import datetime

# ---------------- CONFIG ----------------
# Curated for route + time-of-day diversity across the MSB<->TBM corridor
# (TBM local, CGL sub local, GI local, TMLP semi-fast, CJ, CGL-AJJ, AC EMU,
# MEMU, GPD variants), spanning 03:58-23:08 daily service. Kept to 30 trains
# total (vs. ~99 possible) to stay well under NTES rate limits while still
# tripling+ the row count vs. the original 8.
TRAIN_NUMBERS = [
    # original 8
    "40003", "40045", "40046", "40111", "40149", "40394", "49003", "49004",
    # early morning
    "40501",  # MSB-CGL SUB LOCAL, 03:58
    "40701",  # MSB-CJ SEMI FAST, 05:18
    "40801",  # MSB-TMLP EMU LOCAL, 07:28
    "42502",  # GPD-CGL EMU LOCAL, 07:58
    "40009",  # MSB-TBM EMU LOCAL, 08:18
    # late morning
    "66034",  # TNM-TBM FAST MEMU (Daily), 10:08
    "40527",  # MSB-CGL SUB LOCAL, 10:21
    "40033",  # MSB-TBM EMU LOCAL, 11:06
    # midday
    "40533",  # MSB-CGL SUB LOCAL, 12:10
    "40901",  # MSB-CGL-AJJ EMU, 13:08
    "40051",  # MSB-TBM EMU LOCAL, 14:08
    "40803",  # MSB-TMLP EMU LOCAL, 15:08
    # evening peak
    "40547",  # MSB-CGL SUB LOCAL, 16:23
    "42522",  # GPD-TBM, 16:58
    "40067",  # MSB-TBM EMU LOCAL, 17:53
    "40903",  # MSB-CGL-AJJ EMU FAST, 18:15
    "49007",  # MSB-CGL AC EMU FAST, 18:25
    "40703",  # MSB-CJ EMU LOCAL, 19:28
    "40201",  # MSB-GI EMU LOCAL, 19:38
    # night
    "40203",  # MSB-GI EMU LOCAL, 20:08
    "40081",  # MSB-TBM EMU LOCAL, 21:08
    "40569",  # MSB-CGL SUB LOCAL, 22:08
    "40085",  # MSB-TBM EMU LOCAL, 23:08
]
OUTPUT_FILE = "raw_data.csv"
# ------------------------------------------


def parse_delay_to_minutes(delay_str):
    """
    Converts NTES's delay format to minutes:
    'On Time' -> 0.0
    '00:03'   -> 3.0   (HH:MM format)
    '01:15'   -> 75.0
    ''/None   -> None  (not yet reached / no data)
    """
    if not delay_str:
        return None
    delay_str = delay_str.strip()
    if delay_str.lower() == "on time":
        return 0.0
    match = re.match(r"^(\d+):(\d+)$", delay_str)
    if match:
        hours, minutes = int(match.group(1)), int(match.group(2))
        return float(hours * 60 + minutes)
    return None


def parse_scheduled_time(time_str):
    """Converts '12:24 08-Aug' -> '12:24'. Returns None for 'Source'/'Destination'/''."""
    if not time_str or time_str.strip() in ("Source", "Destination", ""):
        return None
    return time_str.strip().split(" ")[0]


def parse_response_to_rows(data, train_number, date_str):
    rows = []
    if not data or "STNS" not in data:
        print(f"  [WARN] Unexpected response for train {train_number}: {str(data)[:200]}")
        return rows

    for stn in data["STNS"]:
        scheduled_arrival = parse_scheduled_time(stn.get("STA"))
        # Prefer arrival delay; fall back to departure delay if arrival isn't available
        delay = parse_delay_to_minutes(stn.get("DARR")) 
        if delay is None:
            delay = parse_delay_to_minutes(stn.get("DDEP"))

        # Reconstruct an approximate actual arrival time from schedule + delay,
        # since NTES doesn't give a plain actual-time field for non-reporting stations
        actual_arrival = None
        if scheduled_arrival and delay is not None:
            try:
                sched_dt = datetime.strptime(scheduled_arrival, "%H:%M")
                actual_dt = sched_dt.replace(minute=(sched_dt.minute + int(delay)) % 60,
                                              hour=(sched_dt.hour + (sched_dt.minute + int(delay)) // 60) % 24)
                actual_arrival = actual_dt.strftime("%H:%M")
            except (ValueError, TypeError):
                actual_arrival = None

        rows.append({
            "date": date_str,
            "train_id": train_number,
            "station_name": stn.get("SN"),
            "station_order": int(stn.get("Sr", 0)),
            "distance_km": stn.get("DIST"),
            "scheduled_arrival": scheduled_arrival,
            "actual_arrival": actual_arrival,
            "delay_minutes": delay,
        })

    return rows


def reconcile_backups(output_file):
    """
    Finds any raw_data_backup_*.csv files left behind by a previous locked
    write, merges them into output_file (deduping on the same key the rest
    of the script uses), and archives the backups so they're never merged
    in twice. Returns the reconciled dataframe to build on, and whether it
    still needs to be written back to disk.

    This is what was silently missing before: the fallback save wrote a
    backup file but nothing ever merged it back in, so a second full copy
    of a day's data could sit in a backup file and later get pasted/merged
    in by hand without going through drop_duplicates - producing exact
    duplicate rows that looked like a scraping bug but weren't.
    """
    # dtype={"train_id": str} is load-bearing: without it, pandas sees a column
    # that looks purely numeric ("40003") and silently reads it back as int64.
    # Then drop_duplicates() below compares int 40003 against the str "40003"
    # from freshly-scraped data - which is always False - so "identical" rows
    # sail past the dedup check and you get duplicates that look like a
    # scraping bug but are actually a dtype mismatch.
    dtype_map = {"train_id": str}

    backups = sorted(glob.glob("raw_data_backup_*.csv"))
    if not backups:
        return (pd.read_csv(output_file, dtype=dtype_map) if os.path.exists(output_file) else None), False

    print(f"Found {len(backups)} leftover backup file(s) - merging them in: {backups}")
    frames = []
    if os.path.exists(output_file):
        frames.append(pd.read_csv(output_file, dtype=dtype_map))
    for b in backups:
        frames.append(pd.read_csv(b, dtype=dtype_map))

    merged = pd.concat(frames, ignore_index=True)
    merged["train_id"] = merged["train_id"].astype(str)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["date", "train_id", "station_name"], keep="last")
    print(f"  merged {before} rows down to {len(merged)} after deduping backups")

    # Archive the backups instead of deleting outright, so nothing is lost
    # if the merge above ever looks wrong - but rename them so they can
    # never accidentally get merged in a second time.
    archive_dir = "merged_backups"
    os.makedirs(archive_dir, exist_ok=True)
    for b in backups:
        os.replace(b, os.path.join(archive_dir, os.path.basename(b)))
    print(f"  moved backup file(s) into ./{archive_dir}/")

    return merged, True


def main():
    client = NTESClient(timeout=15, retries=2)
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_ntes_format = datetime.now().strftime("%d-%b-%Y")

    all_new_rows = []
    for i, train_number in enumerate(TRAIN_NUMBERS):
        print(f"Fetching train {train_number} ...")
        try:
            data = client.live_status(train_number, today_ntes_format)
        except NTESError as e:
            print(f"  [ERROR] {e}")
            continue

        rows = parse_response_to_rows(data, train_number, today_str)
        # Only keep rows where we actually have a delay value (skip stations
        # the train hasn't reached yet, or with no data at all)
        rows_with_data = [r for r in rows if r["delay_minutes"] is not None]
        print(f"  -> {len(rows_with_data)} of {len(rows)} stations have delay data")
        if rows_with_data:
            print(f"  sample: {rows_with_data[0]}")

        all_new_rows.extend(rows_with_data)

        # Small pause between calls - now that we're hitting 30 trains instead
        # of 8, this keeps the run polite to NTES and avoids back-to-back
        # requests that could trigger throttling. Skip the wait after the
        # last train so we don't delay the save unnecessarily.
        if i < len(TRAIN_NUMBERS) - 1:
            time.sleep(2)

    if not all_new_rows:
        print("No data collected today. Nothing saved.")
        return

    new_df = pd.DataFrame(all_new_rows)

    # Merge in any leftover backup files from a previous locked write
    # BEFORE combining today's scrape, so we always start from one clean,
    # de-duplicated baseline instead of layering new duplicates on old ones.
    old_df, backups_merged = reconcile_backups(OUTPUT_FILE)
    old_row_count = len(old_df) if old_df is not None else 0

    if old_df is not None:
        # avoid duplicate (date, train, station) rows if run twice in a day
        old_df["train_id"] = old_df["train_id"].astype(str)
        new_df["train_id"] = new_df["train_id"].astype(str)
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "train_id", "station_name"], keep="last")
    else:
        combined = new_df

    # --- SAFE SAVE: retry a few times in case the file is locked ---
    # This happens most often when raw_data.csv is open in Excel at the
    # same moment the scheduled task runs. Retrying with a short wait
    # handles brief locks; if it's still locked after retries, we save
    # to a timestamped backup file instead of losing today's data.
    saved = False
    for attempt in range(3):
        try:
            combined.to_csv(OUTPUT_FILE, index=False)
            saved = True
            break
        except PermissionError:
            print(f"  [WARN] {OUTPUT_FILE} is locked (attempt {attempt + 1}/3) - "
                  f"is it open in Excel? Retrying in 10 seconds...")
            time.sleep(10)

    if saved:
        net_new = len(combined) - old_row_count
        print(f"\nScraped {len(new_df)} rows with data this run "
              f"({net_new} were genuinely new, {len(new_df) - net_new} were "
              f"already on file). {OUTPUT_FILE} now has {len(combined)} total rows.")
        if backups_merged:
            print("  (this total also includes rows recovered from a previously "
                  "locked/failed save - see the merge log above)")
    else:
        fallback_name = f"raw_data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        combined.to_csv(fallback_name, index=False)
        print(f"\n[ERROR] Could not write to {OUTPUT_FILE} after 3 attempts - it's "
              f"probably still open elsewhere. Saved today's data to {fallback_name} "
              f"instead so nothing was lost. CLOSE {OUTPUT_FILE} in Excel/Notepad, "
              f"then manually merge {fallback_name} into it.")


if __name__ == "__main__":
    main()

"""
====================================================================
SCHEDULER SETUP (much simpler now - just once a day)
====================================================================
--- MAC / LINUX (cron) ---
crontab -e, then add:
   0 22 * * * /usr/bin/python3 /full/path/to/08_scrape_ntes_client.py >> /full/path/to/scrape_log.txt 2>&1
   (runs once daily at 10 PM, after all trains have finished running)

--- WINDOWS (Task Scheduler) ---
Create Task -> Trigger: Daily at 10:00 PM -> Action: run python.exe
with argument = full path to this script. No repeat interval needed
anymore - just once a day.
====================================================================
"""
