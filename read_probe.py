from pathlib import Path
import os
import resource
import pandas as pd

path = Path("data/probe/hmc_violations.csv")

print(f"pid={os.getpid()}")
print(f"file={path}")
print(f"size_mib={path.stat().st_size / 1024 / 1024:,.1f}")

def rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

rows = 0

for i, chunk in enumerate(pd.read_csv(path, chunksize=100_000), start=1):
    rows += len(chunk)
    print(
        f"chunk={i} rows={rows:,} "
        f"chunk_shape={chunk.shape} "
        f"max_rss={rss_mib():,.1f} MiB",
        flush=True,
    )

print("finished")

