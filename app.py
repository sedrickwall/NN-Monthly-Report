import pandas as pd
import streamlit as st
import plotly.express as px
import gspread
from google.oauth2.service_account import Credentials

SHEET_ID = st.secrets["SHEET_ID"]
TAB_NAME = "weekly_age_bucket_snapshots"

KEY_COLS = ["snapshot_date", "age"]
VALUE_COLS = ["active", "croUpdate", "directUpdate", "hold", "count"]
EXPECTED_HEADER = KEY_COLS + VALUE_COLS

st.set_page_config(page_title="Sales Pipeline Dashboard", layout="wide")

WIDE_REQUIRED = [
    "age",
    "active_count","active_amount",
    "cro_count","cro_amount",
    "direct_count","direct_amount",
    "hold_count","hold_amount",
]

def validate_and_convert_wide(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in WIDE_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    num_cols = [c for c in WIDE_REQUIRED if c != "age"]
    for c in num_cols:
        # Handles numbers that come in as text
        df[c] = (
            df[c].astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    out = pd.DataFrame()
    out["age"] = df["age"].astype(str)

    out["active"] = df["active_amount"]
    out["croUpdate"] = df["cro_amount"]
    out["directUpdate"] = df["direct_amount"]
    out["hold"] = df["hold_amount"]

    out["count"] = (
        df["active_count"] + df["cro_count"] + df["direct_count"] + df["hold_count"]
    ).astype(int)

    return out

from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
HISTORY_PATH = DATA_DIR / "weekly_age_bucket_snapshots.parquet"

HISTORY_KEYS = ["snapshot_date", "age"]

def load_history() -> pd.DataFrame:
    if HISTORY_PATH.exists():
        hist = pd.read_parquet(HISTORY_PATH)
        hist["snapshot_date"] = pd.to_datetime(hist["snapshot_date"]).dt.date
        return hist
    return pd.DataFrame(columns=HISTORY_KEYS + VALUE_COLS)

def save_history(hist: pd.DataFrame) -> None:
    hist.to_parquet(HISTORY_PATH, index=False)

def append_snapshot(hist: pd.DataFrame, snapshot_df: pd.DataFrame, snapshot_date) -> pd.DataFrame:
    snap = snapshot_df.copy()
    snap["snapshot_date"] = pd.to_datetime(snapshot_date).date()

    # Ensure types
    for c in VALUE_COLS:
        snap[c] = pd.to_numeric(snap[c], errors="coerce").fillna(0)

    # Upsert: keep latest for same (snapshot_date, age)
    combined = pd.concat([hist, snap[HISTORY_KEYS + VALUE_COLS]], ignore_index=True)
    combined["snapshot_date"] = pd.to_datetime(combined["snapshot_date"]).dt.date
    combined = combined.drop_duplicates(subset=HISTORY_KEYS, keep="last")

    return combined.sort_values(["snapshot_date", "age"])

def latest_snapshot(hist: pd.DataFrame) -> pd.DataFrame:
    """Get the most recent snapshot only."""
    h = hist.copy()
    h["snapshot_date"] = pd.to_datetime(h["snapshot_date"])
    latest = h["snapshot_date"].max()
    return h[h["snapshot_date"] == latest].copy()

def add_month_column(hist: pd.DataFrame) -> pd.DataFrame:
    h = hist.copy()
    d = pd.to_datetime(h["snapshot_date"])
    h["month"] = d.dt.to_period("M").astype(str)  # "2026-01"
    return h

def monthly_rollup_end_of_month(hist: pd.DataFrame) -> pd.DataFrame:
    """Select the last weekly snapshot in each month (per age bucket)."""
    h = hist.copy()
    h["snapshot_date"] = pd.to_datetime(h["snapshot_date"])
    h["month"] = h["snapshot_date"].dt.to_period("M").astype(str)

    # Get last snapshot date per month
    last_dt = h.groupby("month")["snapshot_date"].max().reset_index()
    
    # Keep only rows for that month's last snapshot date
    out = h.merge(last_dt, on=["month", "snapshot_date"], how="inner")
    return out

def monthly_totals(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Monthly totals across all age buckets (using end-of-month snapshots)."""
    m = monthly_df.copy()
    m["totalValue"] = m["active"] + m["croUpdate"] + m["directUpdate"] + m["hold"]
    
    out = m.groupby("month", as_index=False)[
        ["active", "croUpdate", "directUpdate", "hold", "count", "totalValue"]
    ].sum()

    out["updatesNeeded"] = out["croUpdate"] + out["directUpdate"]
    out["activePct"] = (out["active"] / out["totalValue"] * 100).round(2).fillna(0)
    out["updatesPct"] = (out["updatesNeeded"] / out["totalValue"] * 100).round(2).fillna(0)
    out["holdPct"] = (out["hold"] / out["totalValue"] * 100).round(2).fillna(0)
    return out

def weekly_totals(hist: pd.DataFrame) -> pd.DataFrame:
    """Weekly totals across all age buckets."""
    h = hist.copy()
    h["snapshot_date"] = pd.to_datetime(h["snapshot_date"], errors="coerce")
    h = h.dropna(subset=["snapshot_date"])

    h["totalValue"] = h["active"] + h["croUpdate"] + h["directUpdate"] + h["hold"]
    out = h.groupby("snapshot_date", as_index=False)[
        ["active", "croUpdate", "directUpdate", "hold", "count", "totalValue"]
    ].sum()

    out["updatesNeeded"] = out["croUpdate"] + out["directUpdate"]
    out["activePct"] = (out["active"] / out["totalValue"] * 100).round(2).fillna(0)
    return out.sort_values("snapshot_date")


# ----------------------------
# Helpers
# ----------------------------
@st.cache_resource
def get_gsheet_client():
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=scopes
    )
    return gspread.authorize(creds)

def get_or_create_worksheet():
    gc = get_gsheet_client()
    sh = gc.open_by_key(SHEET_ID)

    try:
        ws = sh.worksheet(TAB_NAME)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=TAB_NAME, rows=2000, cols=20)

    return ws

def ensure_header(ws):
    header = ws.row_values(1)
    if not header:
        ws.update(values=[EXPECTED_HEADER], range_name="A1")
        return "Header created"

    # Strip whitespace for comparison
    header_stripped = [c.strip() for c in header]
    if header_stripped != EXPECTED_HEADER:
        return f"Header mismatch. Found: {header}"

    return "Header OK"

def fetch_google_sheet_data() -> pd.DataFrame:
    """Fetch and convert data from Google Sheet 'weekly_age_bucket_snapshots'."""
    try:
        ws = get_or_create_worksheet()
        
        # Get all values (skip header row)
        all_values = ws.get_all_values()
        
        if len(all_values) <= 1:
            return pd.DataFrame(columns=HISTORY_KEYS + VALUE_COLS)
        
        # Convert to DataFrame
        header = [c.strip() for c in all_values[0]]
        data = all_values[1:]
        
        if not data:
            return pd.DataFrame(columns=HISTORY_KEYS + VALUE_COLS)
        
        df = pd.DataFrame(data, columns=header)
        
        # Check if this is wide format (has active_count, etc.)
        if "active_count" in df.columns:
            # Convert from wide format
            df = validate_and_convert_wide(df)
        elif "active" in df.columns:
            # Already in narrow format, validate it
            df = validate_df(df)
        else:
            st.error(f"Unrecognized sheet format. Expected wide or narrow columns. Found: {df.columns.tolist()}")
            return pd.DataFrame(columns=HISTORY_KEYS + VALUE_COLS)
        
        # Extract snapshot_date from the sheet if available, otherwise use today
        if "snapshot_date" in df.columns:
            df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.date
        else:
            df["snapshot_date"] = pd.Timestamp.today().date()
        
        return df[HISTORY_KEYS + VALUE_COLS]
    
    except Exception as e:
        st.error(f"Failed to fetch Google Sheet data: {e}")
        return pd.DataFrame(columns=HISTORY_KEYS + VALUE_COLS)


def format_currency(val: float) -> str:
    # Compact-ish formatting (K/M/B) without relying on locale quirks
    abs_val = abs(val)
    if abs_val >= 1_000_000_000:
        return f"${val/1_000_000_000:.1f}B"
    if abs_val >= 1_000_000:
        return f"${val/1_000_000:.1f}M"
    if abs_val >= 1_000:
        return f"${val/1_000:.1f}K"
    return f"${val:,.0f}"

REQUIRED_COLUMNS = ["age", "active", "croUpdate", "directUpdate", "hold", "count"]

def validate_df(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize column names (so "CRO Update" can become "croUpdate", etc. if you map it)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Expected columns: {REQUIRED_COLUMNS}"
        )
    # Coerce numeric columns safely
    for col in ["active", "croUpdate", "directUpdate", "hold", "count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["age"] = df["age"].astype(str)
    return df

@st.cache_data
def load_sample_data() -> pd.DataFrame:
    return pd.DataFrame([
        {"age": "0-3 Mth", "active": 191112346, "croUpdate": 0, "directUpdate": 0, "hold": 4273890, "count": 76},
        {"age": "3-6 Mth", "active": 152416036, "croUpdate": 0, "directUpdate": 0, "hold": 2772240, "count": 72},
        {"age": "6-9 Mth", "active": 67899204, "croUpdate": 45976277, "directUpdate": 50880339, "hold": 0, "count": 66},
        {"age": "9-12 Mth", "active": 33900228, "croUpdate": 4748908, "directUpdate": 11653335, "hold": 27404017, "count": 38},
        {"age": "12+ Mth", "active": 90336258, "croUpdate": 3771903, "directUpdate": 16124537, "hold": 50950959, "count": 86},
    ])

def compute_stats(df: pd.DataFrame) -> dict:
    total_value = float((df["active"] + df["croUpdate"] + df["directUpdate"] + df["hold"]).sum())
    active_value = float(df["active"].sum())

    # "Updates Needed" = croUpdate + directUpdate (your React logic)
    updates_needed = float((df["croUpdate"] + df["directUpdate"]).sum())

    # Basic % metrics
    if total_value > 0:
        active_pct = active_value / total_value * 100
    else:
        st.warning("Total pipeline value is zero. Check your data.")
        active_pct = 0.0

    # "stalePct" and "hrecExceeded" were hardcoded in React.
    # If you want these computed, you’ll need fields like "hrec_exceeded_flag" or "last_contact_date".
    return {
        "totalValue": total_value,
        "activeValue": active_value,
        "updatesNeededValue": updates_needed,
        "holdValue": float(df["hold"].sum()),
        "activePct": active_pct,
        "totalCount": int(df["count"].sum()),
    }

# ----------------------------
# Sidebar: Data input
# ----------------------------

st.sidebar.divider()
st.sidebar.subheader("Google Sheets")

if st.sidebar.button("Test Connection"):
    try:
        ws = get_or_create_worksheet()
        header_status = ensure_header(ws)

        # Read a tiny range to confirm read access
        sample = ws.get("A1:G2")

        st.sidebar.success("Connected to Google Sheets successfully")
        st.sidebar.write(f"Worksheet: {ws.title}")
        st.sidebar.write(header_status)
        st.sidebar.write("Sample read:")
        st.sidebar.json(sample)

    except KeyError as e:
        st.sidebar.error(f"Missing Streamlit secret: {e}")
    except Exception as e:
        st.sidebar.error("Connection failed")
        st.sidebar.exception(e)

if st.sidebar.button("Sync History from Google Sheet"):
    try:
        sheet_data = fetch_google_sheet_data()
        if not sheet_data.empty:
            hist = load_history()
            # Append all rows from the sheet to history
            for _, row in sheet_data.iterrows():
                hist = append_snapshot(hist, pd.DataFrame([row]), row["snapshot_date"])
            save_history(hist)
            st.sidebar.success(f"✓ Synced {len(sheet_data)} snapshots from Google Sheet. Total history rows: {len(hist):,}")
        else:
            st.sidebar.warning("No data found in Google Sheet or sheet is empty.")
    except Exception as e:
        st.sidebar.error(f"Failed to sync: {e}")
        st.sidebar.exception(e)

st.sidebar.divider()
view_mode = st.sidebar.radio(
    "View",
    ["Current pipeline", "Trend by month", "Trend by week"],
    index=0
)


st.sidebar.title("Data")
mode = st.sidebar.radio("Choose data source", ["Upload CSV/Excel", "Use sample data", "Google Sheet"], index=2)

if mode == "Upload CSV/Excel":
    uploaded = st.sidebar.file_uploader("Upload file", type=["csv", "xlsx"])
    if uploaded is None:
        st.info("Upload a CSV/Excel with columns: age, active, croUpdate, directUpdate, hold, count")
        df = load_sample_data()
    else:
        if uploaded.name.lower().endswith(".csv"):
            raw = pd.read_csv(uploaded)
        else:
            raw = pd.read_excel(uploaded)
        try:
            df = validate_df(raw)
        except Exception as e:
            st.error(str(e))
            st.stop()
elif mode == "Google Sheet":
    try:
        df = fetch_google_sheet_data()
        if df.empty:
            st.warning("No current snapshot data in Google Sheet. Using sample data.")
            df = load_sample_data()
        else:
            # Group by age and get the latest snapshot
            latest_date = df["snapshot_date"].max()
            df = df[df["snapshot_date"] == latest_date].drop(columns=["snapshot_date"])
    except Exception as e:
        st.error(f"Failed to load from Google Sheet: {e}")
        df = load_sample_data()
else:
    df = load_sample_data()

st.sidebar.divider()
st.sidebar.subheader("Snapshot history (compounding)")

snapshot_date = st.sidebar.date_input("Snapshot date (week ending)", value=pd.Timestamp.today().date())

hist = load_history()

# Normalize snapshot_date in history
if not hist.empty:
    hist["snapshot_date"] = pd.to_datetime(hist["snapshot_date"], errors="coerce")
    hist = hist.dropna(subset=["snapshot_date"])
    hist = hist.sort_values("snapshot_date")

# Debug: Show history stats
if not hist.empty:
    with st.sidebar.expander("📊 History Debug Info"):
        st.write("**History rows:**", len(hist))
        st.write("**Unique snapshot dates:**", hist["snapshot_date"].nunique())
        st.dataframe(
            hist.sort_values("snapshot_date")[["snapshot_date", "age", "active", "croUpdate", "directUpdate", "hold", "count"]].tail(10),
            use_container_width=True
        )

if st.sidebar.button("Append this snapshot to history"):
    # IMPORTANT: df must be validated and in canonical age order
    hist2 = append_snapshot(hist, df, snapshot_date)
    save_history(hist2)
    st.sidebar.success(f"Saved snapshot for {snapshot_date}. History rows: {len(hist2):,}")
    hist = hist2  # refresh
    hist = hist2  # refresh in-memory

if st.sidebar.button("Delete history file (reset)"):
    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink()
    st.sidebar.warning("History reset. Upload and append again.")

# Optional: ensure your age buckets are in a consistent order
AGE_ORDER = ["0-3 Mth", "3-6 Mth", "6-9 Mth", "9-12 Mth", "12+ Mth"]
df["age"] = pd.Categorical(df["age"], categories=AGE_ORDER, ordered=True)
df = df.sort_values("age")

# Prepare data for different views
if view_mode == "Current pipeline":
    if len(hist) > 0:
        df_current = latest_snapshot(hist)
        if not df_current.empty:
            df_current["age"] = pd.Categorical(df_current["age"], categories=AGE_ORDER, ordered=True)
            df_current = df_current.sort_values("age")
            df_display = df_current.drop(columns=["snapshot_date"], errors="ignore")
        else:
            df_display = df
    else:
        df_display = df
elif view_mode == "Trend by month":
    if len(hist) > 0:
        monthly_data = monthly_rollup_end_of_month(hist)
        if not monthly_data.empty:
            monthly_data["age"] = pd.Categorical(monthly_data["age"], categories=AGE_ORDER, ordered=True)
            df_display = monthly_data.sort_values(["month", "age"])
        else:
            df_display = df
    else:
        df_display = df
else:  # Trend by week
    if len(hist) > 0:
        hist_sorted = hist.copy()
        hist_sorted["snapshot_date"] = pd.to_datetime(hist_sorted["snapshot_date"])
        hist_sorted = hist_sorted.sort_values("snapshot_date", ascending=False)
        hist_sorted["age"] = pd.Categorical(hist_sorted["age"], categories=AGE_ORDER, ordered=True)
        df_display = hist_sorted.sort_values(["snapshot_date", "age"])
    else:
        df_display = df

stats = compute_stats(df_display)

st.title("Sales Pipeline Dashboard")
st.caption(f"Executive Performance & Aging Analytics — {view_mode}")

# ----------------------------
# KPI Cards
# ----------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Pipeline", format_currency(stats["totalValue"]))
k2.metric("Active %", f"{stats['activePct']:.0f}%")
k3.metric("Updates Needed", format_currency(stats["updatesNeededValue"]))
k4.metric("Accounts (Total)", f"{stats['totalCount']:,}")

st.divider()

# ----------------------------
# Main row: stacked bar + donut
# ----------------------------
left, right = st.columns([2, 1])

with left:
    st.subheader("Pipeline Composition by Age")

    # Stacked bar needs "long" format
    long_df = df_display.melt(
        id_vars=["age"],
        value_vars=["active", "croUpdate", "directUpdate", "hold"],
        var_name="bucket",
        value_name="value",
    )

    bucket_labels = {
        "active": "Active",
        "croUpdate": "CRO Update",
        "directUpdate": "Direct Update",
        "hold": "On Hold",
    }
    long_df["bucket"] = long_df["bucket"].map(bucket_labels)

    fig_bar = px.bar(
        long_df,
        x="age",
        y="value",
        color="bucket",
        barmode="stack",
        labels={"age": "Age Bucket", "value": "Value ($)", "bucket": ""},
    )
    fig_bar.update_yaxes(showticklabels=False)
    fig_bar.update_traces(hovertemplate="%{x}<br>%{legendgroup}: %{y:$,.0f}<extra></extra>")
    st.plotly_chart(fig_bar, use_container_width=True)

with right:
    st.subheader("Pipeline Health Mix")

    pie_df = pd.DataFrame([
        {"name": "Active", "value": stats["activeValue"]},
        {"name": "Updates Needed", "value": stats["updatesNeededValue"]},
        {"name": "On Hold/Other", "value": stats["holdValue"]},
    ])

    fig_pie = px.pie(
        pie_df,
        names="name",
        values="value",
        hole=0.55,
    )
    fig_pie.update_traces(hovertemplate="%{label}: %{value:$,.0f}<extra></extra>")
    st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# ----------------------------
# Bottom row: bubble scatter + action items
# ----------------------------
b1, b2 = st.columns([2, 1])

with b1:
    st.subheader("Risk Cluster (Volume vs. Value)")

    df_scatter = df_display.copy()
    df_scatter["totalValueBucket"] = df_scatter["active"] + df_scatter["croUpdate"] + df_scatter["directUpdate"] + df_scatter["hold"]

    fig_scatter = px.scatter(
        df_scatter,
        x="age",
        y="totalValueBucket",
        size="count",
        hover_name="age",
        labels={"age": "Age Bucket", "totalValueBucket": "Total Value ($)", "count": "Account Count"},
        size_max=60,
    )
    fig_scatter.update_traces(hovertemplate="%{x}<br>Total: %{y:$,.0f}<br>Count: %{marker.size}<extra></extra>")
    st.plotly_chart(fig_scatter, use_container_width=True)

with b2:
    st.subheader("Critical Action Items")
    # These mirror your React narrative, but computed from the data.
    active_12plus = float(df_display.loc[df_display["age"] == "12+ Mth", "active"].sum())
    updates_6_9 = float(df_display.loc[df_display["age"] == "6-9 Mth", ["croUpdate", "directUpdate"]].sum(axis=1).sum())
    core_0_6 = float(df_display.loc[df_display["age"].isin(["0-3 Mth", "3-6 Mth"]), "active"].sum())

    st.error(f'Audit "Active" 12+ Month Deals — {format_currency(active_12plus)} marked Active')
    st.warning(f'Hygiene Sprint: 6–9 Month Bracket — {format_currency(updates_6_9)} in Updates Needed')
    st.success(f"Velocity Opportunity: 0–6 Month Core — {format_currency(core_0_6)} Active")

st.divider()

st.header("Trends")

if len(hist) == 0:
    st.info("No history yet. Upload a week and click 'Append this snapshot to history'.")
else:
    # Show trends based on view mode
    if view_mode == "Trend by week":
        st.subheader("Weekly Trends")
        wt = weekly_totals(hist)
        
        if len(wt) > 1:
            # Total pipeline trend by week
            fig_week_total = px.line(
                wt,
                x="snapshot_date",
                y="totalValue",
                markers=True,
                labels={"totalValue": "Total Pipeline ($)", "snapshot_date": "Week"}
            )
            fig_week_total.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>Total: %{y:$,.0f}<extra></extra>")
            st.plotly_chart(fig_week_total, use_container_width=True)

            # Weekly mix
            mix = wt.melt(
                id_vars=["snapshot_date"],
                value_vars=["active", "updatesNeeded", "hold"],
                var_name="metric",
                value_name="value"
            )
            label_map = {"active": "Active", "updatesNeeded": "Updates Needed", "hold": "Hold"}
            mix["metric"] = mix["metric"].map(label_map)

            fig_week_mix = px.line(
                mix,
                x="snapshot_date",
                y="value",
                color="metric",
                markers=True,
                labels={"value": "Value ($)", "snapshot_date": "Week", "metric": ""}
            )
            fig_week_mix.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>%{legendgroup}: %{y:$,.0f}<extra></extra>")
            st.plotly_chart(fig_week_mix, use_container_width=True)
        else:
            st.info("Need at least 2 weeks of data to show weekly trends.")
        
        with st.expander("Show weekly data"):
            st.dataframe(wt, use_container_width=True)
    
    else:  # Monthly trend views
        st.subheader("Monthly Trends (Jan–Dec)")
        monthly = monthly_rollup_end_of_month(hist)
        mt = monthly_totals(monthly)

        # Total pipeline trend
        fig_total = px.line(mt, x="month", y="totalValue", markers=True, labels={"totalValue":"Total Pipeline ($)"})
        fig_total.update_traces(hovertemplate="<b>%{x}</b><br>Total: %{y:$,.0f}<extra></extra>")
        fig_total.update_xaxes(hoverformat="%B %Y")
        st.plotly_chart(fig_total, use_container_width=True)

        # Mix trend (%)
        pct_long = mt.melt(
            id_vars=["month"],
            value_vars=["activePct", "updatesPct", "holdPct"],
            var_name="metric",
            value_name="pct",
        )
        metric_labels = {"activePct":"Active %", "updatesPct":"Updates Needed %", "holdPct":"Hold %"}
        pct_long["metric"] = pct_long["metric"].map(metric_labels)

        fig_mix = px.line(pct_long, x="month", y="pct", color="metric", markers=True, labels={"pct":"Percent"})
        fig_mix.update_traces(hovertemplate="<b>%{x}</b><br>%{legendgroup}: %{y:.2f}%<extra></extra>")
        fig_mix.update_xaxes(hoverformat="%B %Y")
        st.plotly_chart(fig_mix, use_container_width=True)

        # Age bucket totals heatmap (optional but powerful)
        monthly2 = monthly.copy()
        monthly2["totalValueBucket"] = monthly2["active"] + monthly2["croUpdate"] + monthly2["directUpdate"] + monthly2["hold"]

        pivot = monthly2.pivot_table(index="age", columns="month", values="totalValueBucket", aggfunc="sum").fillna(0)
        
        if not pivot.empty:
            heat_df = pivot.reset_index().melt(id_vars=["age"], var_name="month", value_name="value")
            fig_heat = px.density_heatmap(
                heat_df, x="month", y="age", z="value",
                labels={"value":"Total ($)"},
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        with st.expander("Show monthly data"):
            st.dataframe(add_month_column(hist).sort_values(["snapshot_date","age"]), use_container_width=True)


# ----------------------------
# Raw data preview (useful for debugging uploads)
# ----------------------------
with st.expander("Show data table"):
    st.dataframe(df, use_container_width=True)
