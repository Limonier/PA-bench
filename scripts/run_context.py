import json, os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EFFORT=os.getenv("SII_REASONING_EFFORT","none")
RESULTS=Path(os.getenv("BENCH_RESULTS_DIR",str(ROOT/"results"/("nex-n2.5-pro-"+EFFORT))))
RESULTS.mkdir(parents=True,exist_ok=True)
config={"model":os.getenv("SII_MODEL","nex-n2.5-pro"),"reasoning_effort":EFFORT}
manifest=RESULTS/"run_config.json"
if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8"))!=config:
    raise RuntimeError("Output directory belongs to a different experiment")
manifest.write_text(json.dumps(config,indent=2),encoding="utf-8")
