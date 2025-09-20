from pathlib import Path

RAW_LABEL_COL = "Label"
MAIN_LABEL_COL = "LabelMain"
SUFFIX_COL = "LabelSuffix"   # Begin/End/Flow 등
ID_COL = "RowID"

# Columns to drop if present (non-feature metadata frequently in CIC-IDS-2017 CSVs)
DROP_IF_PRESENT = {
    "Flow ID","Src IP","Src Port","Dst IP","Dst Port","Timestamp",
    "Protocol","SimillarHTTP","Fwd Header Length","Bwd Header Length"  # keep minimal; adjust later if needed
}

# Features that are non-numeric will be attempted to coerce
NON_NUM_POLICY = "coerce"  # invalid to NaN then fill later
