"""
auto_retrain.py
─────────────────
Automatically triggers a background retrain of growth_model.pkl once enough
NEW real data has accumulated since the last retrain — so nobody has to
remember to run retrain_growth_model.py by hand.

HOW IT DECIDES WHEN TO RUN
────────────────────────────
Every time a real height observation is saved (a DailyGrowthRecord with both
actual_height_cm and a linked SensorData), we check:
    (current count of qualifying real rows) - (count at last retrain) >= AUTO_RETRAIN_EVERY_N
If true, and no retrain is already running, we kick one off.

WHY A BACKGROUND THREAD, NOT INLINE
──────────────────────────────────────
Retraining takes real (if short) time — fitting a few hundred trees on
~2000+ rows. Blocking the person's photo-upload HTTP request until that
finishes would make the app feel slow/broken for no good reason. So we
start it in a daemon thread and let the request return immediately; the
NEXT prediction after the thread finishes will use the updated model.

WHY A JSON STATE FILE, NOT A DB TABLE
────────────────────────────────────────
This only needs to remember two numbers (last retrain's row count and
timestamp) and whether a retrain is currently running. A tiny JSON file
next to growth_model.pkl is simpler than a migration for something this
small, and it's easy to delete/reset by hand if needed.

This is still PERIODIC BATCH RETRAINING under the hood (see
retrain_growth_model.py's docstring for why) — just triggered automatically
by data volume instead of manually by a person. It is NOT true online
learning; the model doesn't update on every single request, only every
AUTO_RETRAIN_EVERY_N new real rows.
"""

import json
import threading
import traceback
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent
STATE_PATH = BASE_DIR / 'ml_models' / 'retrain_state.json'
SYNTHETIC_CSV = BASE_DIR / 'dataset' / 'green_gram_growth_dataset.csv'

AUTO_RETRAIN_EVERY_N = 5   # trigger after every 5 NEW qualifying real rows

_retrain_lock = threading.Lock()
_is_retraining = False


def _load_state():
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    return {'last_retrain_row_count': 0, 'last_retrain_at': None}


def _save_state(state):
    STATE_PATH.parent.mkdir(exist_ok=True)
    STATE_PATH.write_text(json.dumps(state))


def _count_qualifying_rows():
    from agriculture.models import DailyGrowthRecord
    return DailyGrowthRecord.objects.filter(
        actual_height_cm__isnull=False,
        sensor_data__isnull=False,
    ).count()


def _run_retrain_in_background():
    global _is_retraining
    try:
        # Imported here (not at module load time) to avoid circular imports
        # with views.py, and because this only needs to exist while a
        # retrain is actually happening.
        import retrain_growth_model as rg
        rg.retrain(str(SYNTHETIC_CSV))

        state = _load_state()
        state['last_retrain_row_count'] = _count_qualifying_rows()
        import datetime
        state['last_retrain_at'] = datetime.datetime.now().isoformat()
        _save_state(state)
        print("[auto_retrain] Background retrain finished successfully.")
    except Exception:
        print("[auto_retrain] Background retrain FAILED — model left unchanged:")
        traceback.print_exc()
    finally:
        with _retrain_lock:
            _is_retraining = False


def maybe_trigger_retrain():
    """Call this after saving any new real height observation."""
    global _is_retraining

    with _retrain_lock:
        if _is_retraining:
            return  # a retrain is already running, don't stack another

        state = _load_state()
        current_count = _count_qualifying_rows()
        new_rows_since_last_retrain = current_count - state.get('last_retrain_row_count', 0)

        if new_rows_since_last_retrain < AUTO_RETRAIN_EVERY_N:
            return  # not enough new data yet

        _is_retraining = True

    print(f"[auto_retrain] {new_rows_since_last_retrain} new real rows since last "
          f"retrain — starting background retrain now.")
    thread = threading.Thread(target=_run_retrain_in_background, daemon=True)
    thread.start()