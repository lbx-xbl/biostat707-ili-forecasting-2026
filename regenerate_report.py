"""Re-render saved results without training, changing predictions or selection."""
import os
os.environ['MPLCONFIGDIR']=str(__import__('pathlib').Path('.cache/matplotlib').resolve())
from pathlib import Path
import json
import pandas as pd
from src.evaluate import render_report
out=Path('outputs/demo_v0_1')
render_report(pd.read_csv(out/'predictions.csv'),pd.read_csv(out/'metrics.csv'),pd.read_csv(out/'lstm_training_curve.csv'),json.loads((out/'model_selection.json').read_text()),pd.read_csv(out/'split_summary.csv'),json.loads((out/'data_audit.json').read_text()),out)
print('Report and figures regenerated from saved predictions; no training.')
