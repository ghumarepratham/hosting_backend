from fastapi import FastAPI, UploadFile, File, HTTPException
import json
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    contents = await file.read()

    name = (file.filename or "").lower()
    try:
        if name.endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(contents))
            except Exception:
                df = pd.read_csv(io.BytesIO(contents), encoding="latin1")
        elif name.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    desc_json = df.describe(include="all").to_json()
    desc = json.loads(desc_json)
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    missing = df.isna().sum().to_dict()
    columns = list(df.columns)

    outliers = {}
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        series = df[col].dropna()
        if series.empty:
            outliers[col] = {"lower_bound": None, "upper_bound": None, "count": 0}
            continue
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((series < lower) | (series > upper)).sum())
        outliers[col] = {
            "lower_bound": lower,
            "upper_bound": upper,
            "count": count,
        }

    summaries = {}
    def fmt(v):
        try:
            return f"{float(v):.4g}"
        except Exception:
            return str(v)
    for col in columns:
        dtype = dtypes.get(col, "")
        miss = int(missing.get(col, 0))
        oinfo = outliers.get(col, {"count": 0})
        ocount = int(oinfo.get("count", 0))
        lb = oinfo.get("lower_bound", None)
        ub = oinfo.get("upper_bound", None)
        stats = desc.get(col, {})
        if "float" in dtype or "int" in dtype or col in num_cols:
            cnt = fmt(stats.get("count"))
            mean = fmt(stats.get("mean"))
            std = fmt(stats.get("std"))
            minv = fmt(stats.get("min"))
            p25 = fmt(stats.get("25%"))
            med = fmt(stats.get("50%"))
            p75 = fmt(stats.get("75%"))
            maxv = fmt(stats.get("max"))
            lb_s = "-" if lb is None else fmt(lb)
            ub_s = "-" if ub is None else fmt(ub)
            summaries[col] = (
                f"{col}: numeric column with {cnt} values and {miss} missing. "
                f"Mean {mean}, standard deviation {std}. "
                f"Minimum {minv}, 25th percentile {p25}, median {med}, 75th percentile {p75}, maximum {maxv}. "
                f"Detected {ocount} outliers using IQR with bounds [{lb_s}, {ub_s}]."
            )
            try:
                series = df[col].dropna()
                skew = fmt(series.skew())
                kurt = fmt(series.kurt())
                o_pct = "0%" if cnt in [None, "None", "0"] else f"{(ocount / float(cnt)) * 100:.2f}%"
            except Exception:
                skew = "-"
                kurt = "-"
                o_pct = "-"
            summaries_lines = {
                "name": col,
                "dtype": dtype,
                "count": cnt,
                "missing": miss,
                "mean": mean,
                "std": std,
                "min": minv,
                "p25": p25,
                "median": med,
                "p75": p75,
                "max": maxv,
                "skew": skew,
                "kurtosis": kurt,
                "outlier_lower": lb_s,
                "outlier_upper": ub_s,
                "outlier_count": str(ocount),
                "outlier_percent": o_pct,
            }
            lines = [
                f"Column: {summaries_lines['name']}",
                f"Dtype: {summaries_lines['dtype']}",
                f"Count: {summaries_lines['count']}, Missing: {summaries_lines['missing']}",
                f"Mean: {summaries_lines['mean']}, Std: {summaries_lines['std']}",
                f"Min: {summaries_lines['min']}, Max: {summaries_lines['max']}",
                f"25%: {summaries_lines['p25']}, Median: {summaries_lines['median']}, 75%: {summaries_lines['p75']}",
                f"Skewness: {summaries_lines['skew']}, Kurtosis: {summaries_lines['kurtosis']}",
                f"IQR Outlier Bounds: [{summaries_lines['outlier_lower']}, {summaries_lines['outlier_upper']}]",
                f"Outliers: {summaries_lines['outlier_count']} ({summaries_lines['outlier_percent']})",
            ]
            summaries.setdefault("_lines", {})[col] = lines
        else:
            cnt = fmt(stats.get("count"))
            uniq = fmt(stats.get("unique"))
            top = stats.get("top")
            freq = fmt(stats.get("freq"))
            tval = "" if top is None else str(top)
            summaries[col] = (
                f"{col}: categorical column with {cnt} values and {miss} missing. "
                f"Unique values {uniq}. Most frequent value '{tval}' occurs {freq} times."
            )
            try:
                series = df[col].dropna()
                vc = series.value_counts()
                total = float(series.shape[0]) if series.shape[0] else 1.0
                top5 = vc.head(5).to_dict()
                top_items = [f"{k}: {v} ({(v/total)*100:.2f}%)" for k, v in top5.items()]
                distinct_ratio = "-" if cnt in [None, "None", "0"] else f"{(float(uniq) / float(cnt)) * 100:.2f}%"
            except Exception:
                top_items = []
                distinct_ratio = "-"
            lines = [
                f"Column: {col}",
                f"Dtype: {dtype}",
                f"Count: {cnt}, Missing: {miss}",
                f"Unique: {uniq}, Distinctness: {distinct_ratio}",
                f"Mode: '{tval}' with Frequency: {freq}",
            ]
            if top_items:
                lines.append("Top Categories:")
                for item in top_items:
                    lines.append(f"- {item}")
            summaries.setdefault("_lines", {})[col] = lines

    return {
        "shape": df.shape,
        "columns": columns,
        "dtypes": dtypes,
        "missing_values": missing,
        "describe": desc,
        "outliers": outliers,
        "summaries": summaries,
    }
