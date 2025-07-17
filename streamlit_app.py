import streamlit as st
import pandas as pd
import numpy as np
import re

"""GSC Analysis Toolkit – Streamlit App
========================================
Two modes:
1. **Content Opportunity Finder** – highlights high‑impression / low‑CTR queries that rank just off page 1.
2. **YoY Page Analysis** – compares any page’s query metrics year‑over‑year and attributes traffic change to rank, demand, or CTR.

### Expected exports
* **Opportunity Finder** → *Performance → Search results* (no compare) → Export
* **YoY Analysis**      → *Performance → Search results* → **Compare → Year over year** + Page filter → Export (Excel)
"""

st.set_page_config(page_title="GSC Analysis Toolkit", layout="wide")
st.title("Google Search Console – Analysis Toolkit")

###############################################################################
# Helper – load CSV / TSV / XLSX
###############################################################################

def load_gsc_export(uploaded_file):
    """Return a dict of DataFrames keyed by sheet name (upper‑cased)."""
    name = uploaded_file.name.lower()

    if name.endswith((".xls", ".xlsx")):
        try:
            xls = pd.ExcelFile(uploaded_file)
        except ImportError:
            st.error("Excel support requires **openpyxl**. Add it to requirements.txt or `pip install openpyxl`.")
            return {}
        return {s.upper(): xls.parse(s) for s in xls.sheet_names}

    # CSV / TSV – treat as “QUERIES” sheet
    raw_head = uploaded_file.read().decode("utf‑8")
    delim = "\t" if "\t" in raw_head.splitlines()[0] else ","
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file, delimiter=delim)
    return {"QUERIES": df}

###############################################################################
# Content Opportunity Finder
###############################################################################

def run_opportunity_finder():
    st.markdown("Upload a **Search results** export (CSV, TSV, or Excel) then fine‑tune the filters.")

    up = st.file_uploader(
        "Upload GSC export (must include Query, Clicks, Impressions, CTR, Position, Date columns)",
        type=["csv", "tsv", "xls", "xlsx"],
    )
    if not up:
        return

    sheets = load_gsc_export(up)
    if "QUERIES" not in sheets:
        st.error("Could not locate a Queries sheet – double‑check your export type.")
        return

    df = sheets["QUERIES"].copy()
    df.columns = df.columns.str.lower()

    expected = {"query", "clicks", "impressions", "ctr", "position", "date"}
    missing = expected - set(df.columns)
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        return

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["clicks", "impressions", "ctr", "position"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Date range slider
    min_d, max_d = df["date"].min(), df["date"].max()
    default_start = max_d - pd.Timedelta(days=28)
    start_d, end_d = st.slider(
        "Date range", min_value=min_d.date(), max_value=max_d.date(),
        value=(default_start.date(), max_d.date()), format="YYYY‑MM‑DD")
    df = df[df["date"].between(pd.to_datetime(start_d), pd.to_datetime(end_d))]

    # Aggregate
    agg = (
        df.groupby("query")[["clicks", "impressions"]].sum()
        .assign(
            ctr=lambda x: x["clicks"] / x["impressions"],
            position=df.groupby("query")["position"].mean(),
        )
        .reset_index()
    )

    st.sidebar.header("Filters")
    min_impr = st.sidebar.number_input("Min impressions", 0, value=100, step=50)
    max_ctr = st.sidebar.number_input("Max CTR (%)", 0.0, 100.0, value=1.0, step=0.1)
    pos_low, pos_high = st.sidebar.slider("Avg. position band", 1.0, 100.0, (4.0, 20.0), step=0.5)

    opp = agg[(agg["impressions"] >= min_impr) &
              (agg["ctr"] * 100 <= max_ctr) &
              (agg["position"].between(pos_low, pos_high))]
    opp = opp.sort_values("impressions", ascending=False)
    opp["recommended_action"] = "Rewrite meta / beef‑up content"

    st.subheader("⚡ Keyword opportunities")
    st.dataframe(opp[["query", "clicks", "impressions", "ctr", "position", "recommended_action"]])
    st.download_button("Download CSV", opp.to_csv(index=False), "content_opportunities.csv", "text/csv")

###############################################################################
# YoY Page Analysis helpers
###############################################################################

def _normalize_ctr(series):
    """Strip % and convert to float 0‑1."""
    return pd.to_numeric(series.astype(str).str.replace("%", "", regex=False), errors="coerce") / 100.0


def parse_period_columns(df: pd.DataFrame):
    """Add *_new / *_old numeric columns for each metric based on first/second appearance."""
    metrics = ["Clicks", "Impressions", "CTR", "Position"]
    for m in metrics:
        cols = [c for c in df.columns if m.lower() in str(c).lower()]
        if len(cols) < 2:
            continue
        new_col, old_col = cols[0], cols[1]
        if m == "CTR":
            df[f"{m.lower()}_new"] = _normalize_ctr(df[new_col])
            df[f"{m.lower()}_old"] = _normalize_ctr(df[old_col])
        else:
            df[f"{m.lower()}_new"] = pd.to_numeric(df[new_col], errors="coerce")
            df[f"{m.lower()}_old"] = pd.to_numeric(df[old_col], errors="coerce")

###############################################################################
# YoY Page Analysis
###############################################################################

def run_yoy_analysis():
    st.markdown("Upload a *Compare → Year over year* Excel export filtered to a single page.")

    up = st.file_uploader("Upload YoY Excel export", type=["xlsx", "xls"])
    if not up:
        return

    sheets = load_gsc_export(up)
    if {"QUERIES", "PAGES"} - sheets.keys():
        st.error("Missing 'Queries' or 'Pages' tab – did you export the full report?")
        return

    # Page selector (optional if multiple pages)
    pages_df = sheets["PAGES"]
    pages = pages_df.iloc[:, 0].astype(str).tolist()
    st.selectbox("Select page URL", pages, key="page_selector")
    if len(pages) > 1:
        st.warning("Export contains multiple pages; consider exporting a single‑URL report for precise attribution.")

    qdf = sheets["QUERIES"].copy()
    if "Top queries" in qdf.columns and "query" not in qdf.columns:
        qdf = qdf.rename(columns={"Top queries": "query"})

    parse_period_columns(qdf)
    required = [f"{m}_{p}" for m in ("clicks", "impressions", "ctr", "position") for p in ("new", "old")]
    if not all(c in qdf.columns for c in required):
        st.error("Could not identify both periods for each metric – ensure you’re in Compare → YoY mode.")
        return

    q = qdf.copy()
    q["clicks_diff"]       = q["clicks_new"]       - q["clicks_old"]
    q["impressions_diff"]  = q["impressions_new"]  - q["impressions_old"]
    q["ctr_diff"]          = q["ctr_new"]          - q["ctr_old"]
    q["position_diff"]     = q["position_new"]     - q["position_old"]

    # Attribution
    q["impr_effect"]        = q["impressions_diff"] * q["ctr_old"]
    q["ctr_effect"]         = q["clicks_old"]      * q["ctr_diff"]
    q["interaction_effect"] = q["impressions_diff"] * q["ctr_diff"]

    def classify(r):
        if abs(r["position_diff"]) >= 1 and abs(r["impressions_diff"]) < 0.1 * r["impressions_old"]:
            return "Rank change"
        if abs(r["impressions_diff"]) >= 0.1 * r["impressions_old"]:
            return "Demand change"
        if abs(r["ctr_diff"]) >= 0.01:
            return "CTR change"
        return "Other"
    q["reason"] = q.apply(classify, axis=1)

    # KPIs
    new_clicks, old_clicks = q["clicks_new"].sum(), q["clicks_old"].sum()
    delta = new_clicks - old_clicks
    c1, c2 = st.columns(2)
    c1.metric("Clicks (new period)", f"{int(new_clicks):,}", f"{delta:+,}")
    pct = delta / old_clicks * 100 if old_clicks else np.nan
    c2.metric("YoY % change", f"{pct:+.1f}%")

    st.subheader("Query‑level changes")
    movers = q.sort_values("clicks_diff", ascending=False)
    st.dataframe(movers[[
        "query",
        "clicks_new", "clicks_old", "clicks_diff",
        "impressions_new", "impressions_old", "impressions_diff",
        "ctr_new", "ctr_old", "ctr_diff",
        "position_new", "position_old", "position_diff",
        "reason",
    ]])

    st.download_button("Download CSV", movers.to_csv(index=False), "yoy_query_analysis.csv", "text/csv")

###############################################################################
# Sidebar mode switch
###############################################################################
mode = st.sidebar.radio("Select a tool", ["Content Opportunity Finder", "YoY Page Analysis"])

if mode == "Content Opportunity Finder":
    run_opportunity_finder()
else:
    run_yoy_analysis()
