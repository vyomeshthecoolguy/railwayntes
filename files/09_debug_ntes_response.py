"""
DEBUG: print the FULL raw response for one train, so we can see the
real field names NTES sends back - don't guess, just look.

Run this while a train (pick one you've confirmed is actually running
right now on NTES's website) is en route.
"""

from ntes import NTESClient, NTESError
from datetime import datetime

TRAIN_NUMBER = "40046"   # change to a train you've confirmed is running NOW

client = NTESClient(timeout=15, retries=2)
today_ntes_format = datetime.now().strftime("%d-%b-%Y")

print(f"Checking train {TRAIN_NUMBER} for date {today_ntes_format}...")

try:
    status = client.live_status(TRAIN_NUMBER, today_ntes_format)
    print("\n--- FULL RAW RESPONSE ---")
    import json
    print(json.dumps(status, indent=2))
except NTESError as e:
    print(f"NTESError: {e}")
