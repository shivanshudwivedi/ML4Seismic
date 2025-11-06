import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from gwpy.time import to_gps
import logging

# =========================================================
# 1. USER CONFIGURATION
# =========================================================
IFO = "L1"  # Interferometer (e.g., "L1", "H1")
HOST = "nds.ligo-la.caltech.edu"
EARTHQUAKE_CHANNEL = f"{IFO}:ISI-GND_STS_ETMX_Z_BLRMS_30M_100M.mean,m-trend"

DATE_START = "2024-09-27 06:00:00"
DATE_END = "2024-10-27 16:00:00"
EARTHQUAKE_THRESHOLD = 300  # BLRMS threshold for earthquake
OUT_PLOT = f"./{IFO}_earthquake_segments.png"

# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================
def setup_logging():
    """Configures logging with consistent format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def fetch_timeseries(channel: str, t0: int, t1: int, host: str) -> TimeSeries | None:
    """Fetch a GWPy TimeSeries safely."""
    try:
        logging.info(f"Fetching data for {channel}")
        ts = TimeSeries.get(channel, t0, t1, host=host, allow_tape=True)
        logging.info(f"→ Retrieved {len(ts)} samples at {ts.sample_rate.value:.2f} Hz")
        return ts
    except Exception as e:
        logging.error(f"Failed to fetch {channel}: {e}")
        return None

def find_contiguous_segments(ts: TimeSeries, threshold: float) -> list[tuple[int, int]]:
    """
    Identify contiguous earthquake segments where ts >= threshold.
    Returns a list of (start_time, end_time) GPS pairs.
    """
    times = ts.times.value
    mask = ts.value >= threshold
    change_points = np.where(np.diff(mask.astype(int)))[0] + 1
    segment_indices = np.split(np.arange(len(mask)), change_points)
    
    segments = []
    for segment in segment_indices:
        if len(segment) > 0 and mask[segment[0]]:
            start, end = int(times[segment[0]]), int(times[segment[-1]])
            segments.append((start, end))
    return segments

def plot_earthquake_segments(ts: TimeSeries, segments: list[tuple[int, int]], threshold: float, outfile: str):
    """Plot the time series with earthquake segments highlighted."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ts.times, ts.value, color="steelblue", lw=1, label="BLRMS (30–100 mHz)")
    
    for s, e in segments:
        ax.axvspan(s, e, color="red", alpha=0.3, label="Earthquake segment")

    ax.axhline(threshold, color="black", linestyle="--", lw=1, label=f"Threshold = {threshold}")
    ax.set_title(f"{IFO} Earthquake Segments ({DATE_START} → {DATE_END})")
    ax.set_xlabel("Time [GPS]")
    ax.set_ylabel("BLRMS [counts]")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    logging.info(f"Plot saved to: {outfile}")

# =========================================================
# 3. MAIN
# =========================================================
def main():
    setup_logging()

    gps_start = int(to_gps(DATE_START))
    gps_end = int(to_gps(DATE_END))
    logging.info(f"Analyzing earthquake activity from {DATE_START} to {DATE_END}")

    ts = fetch_timeseries(EARTHQUAKE_CHANNEL, gps_start, gps_end, HOST)
    if ts is None:
        logging.critical("Failed to fetch earthquake channel. Exiting.")
        return

    segments = find_contiguous_segments(ts, EARTHQUAKE_THRESHOLD)
    logging.info(f"Found {len(segments)} earthquake segments exceeding {EARTHQUAKE_THRESHOLD} BLRMS")

    for s, e in segments:
        logging.info(f"  Segment: {s} → {e} ({(e-s)/60:.1f} min)")

    plot_earthquake_segments(ts, segments, EARTHQUAKE_THRESHOLD, OUT_PLOT)

if __name__ == "__main__":
    main()
