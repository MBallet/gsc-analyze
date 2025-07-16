import streamlit as st
import pandas as pd
import numpy as np
import re

"""GSC Analysis Toolkit – Streamlit App
---------------------------------------
Two analysis modes:
1. Content Opportunity Finder – surfaces high‑impression/low‑CTR queries within a selectable position band.
2. YoY Page Analysis        – compares query performance for a single page year‑over‑year and classifies the driver of change.

Upload either:
* Search results export (CSV/TSV/XLSX) for Opportunity Finder
* Full Performance export (XLSX, with Queries & Pages tabs) in compare‑YoY mode for Page Analysis.
"""

st.set_page_config(page_title="GSC Analysis Toolkit", layout="wide")

st.title("Google Search Console – Analysis Toolkit")

###############################################################################
# Sidebar – analysis mode selector
###############################################################################
mode = st.sidebar.radio(
    "Select a tool",
    ["Content Opportunity Finder", "YoY Page Analysis"],
    help="Switch between prospecting low‑hanging keywords and diagnosing year‑over‑year shifts.",
)

###############################################################################
# Utility: load export (CSV/TSV/XLSX)
###############################################################################

def load_gsc_export(uploaded_file):
    """Return dict of DataFrames keyed by uppercase sheet name."""
    fname = uploaded_file.name.lower()
    if fname.endswith((".xls", ".xlsx")):
        xls = pd.ExcelFile(uploaded_file)
        return {name.upper(): xls.parse(name) for name in xls.sheet_names}

    # CSV / TSV – treat whole file as QUERIES sheet
    raw_head = uploaded_file.read().decode("utf-8")
    delim = "\t" if "\t" in raw_head.splitlines()[0] else ","
    uploaded_file.seek(0)  # reset pointer
    df = pd.read_csv(uploaded_file, delimiter=delim)
    return {"QUERIES": df}

###############################################################################
# Content Opportunity Finder
###############################################################################

def run_opportunity_finder():
    st.markdown(
        """
        **Step 1.** Export *Performance → Search results* (including Date column) as CSV, TSV, or Excel.  
        **Step 2.** Upload below, adjust filters, and download the opportunity list.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload GSC export (Query, Page, Clicks, Impressions, CTR, Position, Date)",
        type=["csv", "tsv", "xls", "xlsx"],
    )
    if not uploaded_file:
        return

    sheets = load_gsc_export(uploaded_file)
    if "QUERIES" not in sheets:
        st.error("Could not find the Queries sheet – ensure you exported *Search results*.")
        return

    df = sheets["QUERIES"].copy()
    df.columns = df.columns.str.lower()
    required_cols = {"query", "clicks", "impressions", "ctr", "position", "date"}
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"Missing expected columns: {', '.join(missing)}")
        return

    # Types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["clicks", "impressions", "ctr", "position"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Date filter
    min_d, max_d = df["date"].min(), df["date"].max()
    default_start = max_d - pd.Timedelta(days=28)
    start_d, end_d = st.slider(
        "Date range",
        min_value=min_d.date(), max_value=max_d.date(),
        value=(default_start.date(), max_d.date()),
        format="YYYY‑MM‑DD",
    )
    mask = df["date"].between(pd.to_datetime(start_d), pd.to_datetime(end_d))
    df = df.loc[mask]

    # Aggregate at query level
    agg = (
        df.groupby("query")[["clicks", "impressions"]].sum()
        .assign(
            ctr=lambda x: x["clicks"] / x["impressions"],
            position=df.groupby("query")["position"].mean(),
        )
        .reset_index()
    )

    # Sidebar filters
    st.sidebar.header("Filters")
    min_impr = st.sidebar.number_input("Min. Impressions", value=100, step=50)
    max_ctr = st.sidebar.number_input("Max. CTR (%)", value=1.0, step=0.1)
    pos_range = st.sidebar.slider("Avg. Position Range", 1.0, 100.0, (4.0, 20.0), step=0.5)

    opp = agg[(agg["impressions"] >= min_impr) &
              (agg["ctr"] * 100 <= max_ctr) &
              (agg["position"].between(*pos_range))]
    opp = opp.sort_values("impressions", ascending=False)
    opp["recommended_action"] = "Rewrite meta / improve content"

    st.subheader("⚡ Content Opportunities")
    st.dataframe(opp[["query", "clicks", "impressions", "ctr", "position", "recommended_action"]])

    st.download_button(
        "Download CSV", opp.to_csv(index=False), "content_opportunities.csv", "text/csv"
    )

###############################################################################
# YoY Page Analysis
###############################################################################

def parse_period_columns(df):
    """Map metrics to newest/oldest columns and add *_new / *_old normalized fields."""
    metrics = ["Clicks", "Impressions", "CTR", "Position"]
    for metric in metrics:
        cols = [c for c in df.columns if str(c).strip().endswith(metric)]
        if len(cols) < 2:
            continue  # not found
        # Sort by year extracted from column name (YYYY)
        cols_sorted = sorted(cols, key=lambda c: int(re.search(r"(\d{4})", str(c)).group(1)), reverse=True)
        new_col, old_col = cols_sorted[:2]
        df[f"{metric.lower()}_new"] = pd.to_numeric(df[new_col], errors="coerce")
        df[f"{metric.lower()}_old"] = pd.to_numeric(df[old_col], errors="coerce")


def run_yoy_analysis():
    st.markdown(
        """
        Export Performance → Search results in **Compare → Year over year** mode, filtered to a single page, and choose Excel.  
        Upload that file below to diagnose why traffic changed.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload YoY Excel export (with Queries & Pages tabs)",
        type=["xlsx", "xls"],
    )
    if not uploaded_file:
        return

    sheets = load_gsc_export(uploaded_file)
    if {"QUERIES", "PAGES"} - sheets.keys():
        st.error("Expected both 'Queries' and 'Pages' tabs – make sure you exported the full report.")
        return

    # Page selector (supports multi‑page export but warns)
    pages_df = sheets["PAGES"]
    page_urls = pages_df.iloc[:, 0].astype(str).tolist()
    page = st.selectbox("Select page URL", page_urls)
    if len(page_urls) > 1:
        st.warning("Export contains multiple pages – results will include all queries unless you filter the export to a single URL.")

    qdf = sheets["QUERIES"].copy()
    parse_period_columns(qdf)

    needed = [f"{m}_new" for m in ("clicks", "impressions", "ctr", "position")] + \
             [f"{m}_old" for m in ("clicks", "impressions", "ctr", "position")]
    if not all(col in qdf.columns for col in needed):
        st.error("Could not detect both period columns for Clicks, Impressions, CTR, and Position.")
        return

    # Compute deltas
    q = qdf.copy()
    q["clicks_diff"] = q["clicks_new"] - q["clicks_old"]
    q["impressions_diff"] = q["impressions_new"] - q["impressions_old"]
    q["ctr_diff"] = q["ctr_new"] - q["ctr_old"]
    q["position_diff"] = q["position_new"] - q["position_old"]

    # Simple attribution
    q["impr_effect"] = q["impressions_diff"] * q["ctr_old"]
    q["ctr_effect"] = q["clicks_old"] * q["ctr_diff"]
    q["interaction_effect"] = q["impressions_diff"] * q["ctr_diff"]
    q["clicks_pred_diff"] = q[["impr_effect", "ctr_effect", "interaction_effect"]].sum(axis=1)

    def classify(row):
        if abs(row["position_diff"]) >= 1 and abs(row["impressions_diff"]) < 0.1 * row["impressions_old"]:
            return "Rank change"
        if abs(row["impressions_diff"]) >= 0.1 * row["impressions_old"]:
            return "Demand change"
        if abs(row["ctr_diff"]) >= 0.01:
            return "CTR change"
        return "Other"

    q["reason"] = q.apply(classify, axis=1)

    # Overview KPIs
    new_clicks, old_clicks = q["clicks_new"].sum(), q["clicks_old"].sum()
    delta_clicks = new_clicks - old_clicks
    k1, k2 = st.columns(2)
    k1.metric("Clicks (New period)", f"{int(new_clicks):,}", f"{delta_clicks:+,}")
    pct = (delta_clicks / old_clicks * 100) if old_clicks else np.nan
    k2.metric("YoY % change", f"{pct:+.1f}%")

    # Table of movers
    movers = q.sort_values("clicks_diff", ascending=False)
    rename_cols = {"Top queries": "query"} if "Top queries" in movers.columns else {}
    movers = movers.rename(columns=rename_cols)

    st.subheader("Query‑level changes")
    st.dataframe(
        movers[[
            movers.columns[0],  # query column (either 'query' or 'Top queries')
            "clicks_new", "clicks_old", "clicks_diff",
            "impressions_new", "impressions_old", "impressions_diff",
            "ctr_new", "ctr_old", "ctr_diff",
            "position_new", "position_old", "position_diff",
            "reason",
        ]]
    )

    st.download_button(
        "Download full query analysis CSV",
        movers.to_csv(index=False),
        "yoy_query_analysis.csv",
        "text/csv",
    )

###############################################################################
# Entrypoint – run selected mode
###############################################################################
if mode == "Content Opportunity Finder":
    run_opportunity_finder()
else:
    run_yoy_analysis()
