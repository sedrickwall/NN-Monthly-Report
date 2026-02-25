import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.io as pio
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
from fpdf import FPDF
import io

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
    
    # Preserve snapshot_date if it exists
    if "snapshot_date" in df.columns:
        out["snapshot_date"] = df["snapshot_date"]

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
    
    # Sort by month to ensure chronological order
    out = out.sort_values("month")
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

        # Normalize column names: map common variants to canonical names
        col_map = {}
        for c in df.columns:
            key = c.strip().lower().replace(" ", "_")
            if key in ("snapshot_date", "snapshotdate", "date", "snapshot"):
                col_map[c] = "snapshot_date"
            elif "active_count" in key or ("active" in key and "count" in key):
                col_map[c] = "active_count"
            elif "active_amount" in key or ("active" in key and "amount" in key):
                col_map[c] = "active_amount"
            elif "cro_count" in key or ("cro" in key and "count" in key):
                col_map[c] = "cro_count"
            elif "cro_amount" in key or ("cro" in key and "amount" in key) or "croupdate" in key:
                col_map[c] = "cro_amount"
            elif "direct_count" in key or ("direct" in key and "count" in key):
                col_map[c] = "direct_count"
            elif "direct_amount" in key or ("direct" in key and "amount" in key):
                col_map[c] = "direct_amount"
            elif "hold_count" in key or ("hold" in key and "count" in key):
                col_map[c] = "hold_count"
            elif "hold_amount" in key or ("hold" in key and "amount" in key):
                col_map[c] = "hold_amount"
            elif key in ("age",):
                col_map[c] = "age"
            elif key in ("active", "croupdate", "directupdate", "hold", "count"):
                # already narrow-format names
                col_map[c] = key

        if col_map:
            df = df.rename(columns=col_map)

        # Check if this is wide format (has active_count, etc.)
        if "active_count" in df.columns or (set(["active_amount", "cro_amount", "direct_amount", "hold_amount"]) & set(df.columns)):
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
            df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce").dt.date
        else:
            st.warning("No snapshot_date column found. Using today's date for all rows.")
            df["snapshot_date"] = pd.Timestamp.today().date()
        
        # Ensure all required columns exist
        for col in HISTORY_KEYS + VALUE_COLS:
            if col not in df.columns:
                st.warning(f"Missing column: {col}")
                return pd.DataFrame(columns=HISTORY_KEYS + VALUE_COLS)
        return df[HISTORY_KEYS + VALUE_COLS]
    
    except Exception as e:
        st.error(f"Failed to fetch Google Sheet data: {e}")
        return pd.DataFrame(columns=HISTORY_KEYS + VALUE_COLS)


def generate_pdf_report(stats: dict, view_mode: str, df_display: pd.DataFrame, hist: pd.DataFrame = None) -> bytes:
    """Generate a PDF report with metrics and data tables."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 10, "Sales Pipeline Dashboard Report", ln=True, align="C")
    
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 5, f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", ln=True, align="C")
    pdf.cell(0, 5, f"View: {view_mode}", ln=True, align="C")
    pdf.ln(10)
    
    # KPI Summary
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "Key Metrics", ln=True)
    
    pdf.set_font("Helvetica", "", 11)
    pdf.set_fill_color(240, 240, 240)
    
    metrics = [
        ("Total Pipeline", format_currency(stats.get('totalValue', 0))),
        ("Active", f"{format_currency(stats.get('activeValue', 0))} ({stats.get('activePct', 0):.1f}%)"),
        ("Updates Needed", format_currency(stats.get('updatesNeededValue', 0))),
        ("On Hold", format_currency(stats.get('holdValue', 0))),
        ("Total Accounts", f"{stats.get('totalCount', 0):,}"),
    ]
    
    for label, value in metrics:
        pdf.cell(80, 7, label, border=1, fill=True)
        pdf.cell(0, 7, value, border=1, ln=True, fill=False)
    
    pdf.ln(8)
    
    # Age Bucket Summary
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "Pipeline by Age Bucket", ln=True)
    
    pdf.set_font("Helvetica", "", 9)
    try:
        age_summary = df_display.groupby("age")[["active", "croUpdate", "directUpdate", "hold"]].sum()
        
        # Header row
        pdf.set_fill_color(200, 200, 200)
        pdf.cell(40, 6, "Age Bucket", border=1, fill=True)
        pdf.cell(40, 6, "Active", border=1, fill=True)
        pdf.cell(40, 6, "CRO Update", border=1, fill=True)
        pdf.cell(40, 6, "Direct Update", border=1, fill=True)
        pdf.cell(40, 6, "Hold", border=1, ln=True, fill=True)
        
        # Data rows
        pdf.set_fill_color(255, 255, 255)
        for age in age_summary.index:
            row = age_summary.loc[age]
            pdf.cell(40, 6, str(age)[:15], border=1)
            pdf.cell(40, 6, f"${row['active']/1e6:.1f}M", border=1)
            pdf.cell(40, 6, f"${row['croUpdate']/1e6:.1f}M", border=1)
            pdf.cell(40, 6, f"${row['directUpdate']/1e6:.1f}M", border=1)
            pdf.cell(40, 6, f"${row['hold']/1e6:.1f}M", border=1, ln=True)
    except Exception as e:
        pdf.cell(0, 5, f"Age bucket data unavailable: {str(e)[:50]}", ln=True)
    
    pdf.ln(8)
    
    # View-specific summary
    pdf.set_font("Helvetica", "B", 14)
    if view_mode == "Current pipeline":
        pdf.cell(0, 8, "Current Pipeline Summary", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 5, f"This report shows the current state of the sales pipeline as of {datetime.now().strftime('%B %d, %Y')}.")
    
    elif view_mode == "Trend by week" and hist is not None and len(hist) > 1:
        pdf.cell(0, 8, "Weekly Trends Summary", ln=True)
        pdf.set_font("Helvetica", "", 9)
        try:
            wt = weekly_totals(hist)
            if len(wt) > 1:
                pdf.multi_cell(0, 5, f"Weekly data spans from {wt['snapshot_date'].min()} to {wt['snapshot_date'].max()}.")
                pdf.cell(0, 5, f"Total weeks tracked: {len(wt)}", ln=True)
                pdf.ln(5)
                
                # Weekly data table
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 8, "Weekly Pipeline Totals", ln=True)
                
                pdf.set_font("Helvetica", "", 8)
                pdf.set_fill_color(200, 200, 200)
                pdf.cell(30, 6, "Week", border=1, fill=True)
                pdf.cell(30, 6, "Total ($)", border=1, fill=True)
                pdf.cell(30, 6, "Active ($)", border=1, fill=True)
                pdf.cell(30, 6, "Updates ($)", border=1, fill=True)
                pdf.cell(30, 6, "Hold ($)", border=1, ln=True, fill=True)
                
                pdf.set_fill_color(255, 255, 255)
                for _, row in wt.iterrows():
                    pdf.cell(30, 6, str(row['snapshot_date'])[:10], border=1)
                    pdf.cell(30, 6, f"${row['totalValue']/1e6:.1f}M", border=1)
                    pdf.cell(30, 6, f"${row['active']/1e6:.1f}M", border=1)
                    pdf.cell(30, 6, f"${row['updatesNeeded']/1e6:.1f}M", border=1)
                    pdf.cell(30, 6, f"${row['hold']/1e6:.1f}M", border=1, ln=True)
        except Exception as e:
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 5, f"Could not generate weekly trends: {str(e)[:60]}")
    
    elif view_mode == "Trend by month" and hist is not None and len(hist) > 1:
        pdf.cell(0, 8, "Monthly Trends Summary", ln=True)
        pdf.set_font("Helvetica", "", 9)
        try:
            monthly = monthly_rollup_end_of_month(hist)
            mt = monthly_totals(monthly)
            if len(mt) > 1:
                pdf.multi_cell(0, 5, f"Monthly data spans from {mt['month'].min()} to {mt['month'].max()}.")
                pdf.cell(0, 5, f"Total months tracked: {len(mt)}", ln=True)
                pdf.ln(5)
                
                # Monthly data table
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 8, "Monthly Pipeline Totals", ln=True)
                
                pdf.set_font("Helvetica", "", 8)
                pdf.set_fill_color(200, 200, 200)
                pdf.cell(25, 6, "Month", border=1, fill=True)
                pdf.cell(25, 6, "Total ($)", border=1, fill=True)
                pdf.cell(25, 6, "Active ($)", border=1, fill=True)
                pdf.cell(25, 6, "Updates ($)", border=1, fill=True)
                pdf.cell(25, 6, "Hold ($)", border=1, fill=True)
                pdf.cell(20, 6, "Active %", border=1, ln=True, fill=True)
                
                pdf.set_fill_color(255, 255, 255)
                for _, row in mt.iterrows():
                    pdf.cell(25, 6, str(row['month']), border=1)
                    pdf.cell(25, 6, f"${row['totalValue']/1e6:.1f}M", border=1)
                    pdf.cell(25, 6, f"${row['active']/1e6:.1f}M", border=1)
                    pdf.cell(25, 6, f"${row['updatesNeeded']/1e6:.1f}M", border=1)
                    pdf.cell(25, 6, f"${row['hold']/1e6:.1f}M", border=1)
                    pdf.cell(20, 6, f"{row['activePct']:.1f}%", border=1, ln=True)
        except Exception as e:
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 5, f"Could not generate monthly trends: {str(e)[:60]}")
    
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 4, "For interactive charts and detailed analysis, visit the Sales Pipeline Dashboard. Charts are optimized for web viewing where full interactivity is available.")
    
    # Convert to bytes explicitly
    return bytes(pdf.output())


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
    hist["snapshot_dt"] = pd.to_datetime(hist["snapshot_date"], errors="coerce")
    
    # Calculate ISO week label as fallback
    iso = hist["snapshot_dt"].dt.isocalendar()
    hist["week_label"] = iso.year.astype(str) + " W" + iso.week.astype(str).str.zfill(2)
    
    # Check if date parsing failed on too many rows
    bad_rate = hist["snapshot_dt"].isna().mean()
    
    if bad_rate > 0.2:
        # Fallback: use week_label instead of datetime
        st.warning(f"⚠️ {bad_rate*100:.1f}% of snapshot_date values could not be parsed. Using week labels as fallback.")
        use_week_label = True
        hist["snapshot_date"] = hist["week_label"]
    else:
        use_week_label = False
        hist["snapshot_date"] = hist["snapshot_dt"].dt.date
    
    # Clean up and sort
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
    # Warn if we're about to overwrite an existing snapshot_date for these ages
    existing_dates = set(pd.to_datetime(hist["snapshot_date"], errors="coerce").dt.date.dropna().tolist()) if len(hist) else set()
    if snapshot_date in existing_dates:
        st.sidebar.warning("This snapshot date already exists. Appending will update/overwrite that week for matching age buckets.")

    hist2 = append_snapshot(hist, df, snapshot_date)
    save_history(hist2)
    st.sidebar.success(f"Saved snapshot for {snapshot_date}. History rows: {len(hist2):,}")
    # Refresh UI so the new history is visible immediately
    st.experimental_rerun()

if st.sidebar.button("Delete history file (reset)"):
    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink()
    st.sidebar.warning("History reset. Upload and append again.")

# Optional: ensure your age buckets are in a consistent order
AGE_ORDER = ["0-3 Mth", "3-6 Mth", "6-9 Mth", "9-12 Mth", "12+ Mth"]
df["age"] = pd.Categorical(df["age"], categories=AGE_ORDER, ordered=True)
df = df.sort_values("age")

# Prepare data for different views
df_display = df.copy()  # Default to current uploaded data
show_charts = True  # Show main charts by default
show_trends = True  # Show trends section

# Auto-sync from Google Sheets if history is empty and user selected a trend view
if view_mode in ["Trend by month", "Trend by week"] and len(hist) == 0:
    try:
        sheet_data = fetch_google_sheet_data()
        if not sheet_data.empty:
            st.info(f"📊 Auto-syncing {len(sheet_data)} rows from Google Sheet ({sheet_data['snapshot_date'].nunique()} unique dates)...")
            
            # Show what we're loading
            with st.expander("📋 Rows being synced from Google Sheet"):
                st.dataframe(sheet_data.sort_values("snapshot_date"), use_container_width=True)
            
            for _, row in sheet_data.iterrows():
                hist = append_snapshot(hist, pd.DataFrame([row]), row["snapshot_date"])
            save_history(hist)
            # Normalize after loading
            if not hist.empty:
                hist["snapshot_date"] = pd.to_datetime(hist["snapshot_date"], errors="coerce")
                hist = hist.dropna(subset=["snapshot_date"])
                hist = hist.sort_values("snapshot_date")
    except Exception as e:
        st.warning(f"Could not auto-sync from Google Sheet: {e}")

if view_mode == "Current pipeline":
    show_charts = True
    show_trends = False
    if len(hist) > 0:
        df_current = latest_snapshot(hist)
        if not df_current.empty:
            df_current["age"] = pd.Categorical(df_current["age"], categories=AGE_ORDER, ordered=True)
            df_current = df_current.sort_values("age")
            df_display = df_current.drop(columns=["snapshot_date"], errors="ignore")
elif view_mode == "Trend by month":
    show_charts = False
    show_trends = True
elif view_mode == "Trend by week":
    show_charts = False
    show_trends = True

stats = compute_stats(df_display)

st.title("Sales Pipeline Dashboard")

# Add PDF download button in the title area
col1, col2 = st.columns([3, 1])
with col1:
    st.caption(f"Executive Performance & Aging Analytics — {view_mode}")
with col2:
    try:
        pdf_bytes = generate_pdf_report(stats, view_mode, df_display, hist)
        st.download_button(
            label="📥 Download PDF",
            data=pdf_bytes,
            file_name=f"Sales_Pipeline_Report_{datetime.now().strftime('%Y-%m-%d')}.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.warning(f"PDF download unavailable: {e}")

# Show KPI cards only for "Current pipeline" view
if view_mode == "Current pipeline":
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
# Main charts section (only for "Current pipeline" view)
# ----------------------------
if show_charts:
    # KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Pipeline", format_currency(stats["totalValue"]))
    k2.metric("Active %", f"{stats['activePct']:.0f}%")
    k3.metric("Updates Needed", format_currency(stats["updatesNeededValue"]))
    k4.metric("Accounts (Total)", f"{stats['totalCount']:,}")

    st.divider()

# ----------------------------
# Main row: stacked bar + donut
# ----------------------------
if show_charts:
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

if show_charts:
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

# ----------------------------
# Trends section
# ----------------------------
if show_trends:
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
            
            # Debug: Show what we have
            with st.expander("📊 Debug: Monthly Data"):
                st.write("Months found:", monthly["month"].unique() if "month" in monthly.columns else "No month column")
                st.write("Rows in monthly data:", len(monthly))
                st.dataframe(monthly.sort_values("month"), use_container_width=True)
            
            mt = monthly_totals(monthly)
            
            with st.expander("📊 Debug: Monthly Totals"):
                st.write("Months in totals:", mt["month"].unique())
                st.write("Rows in totals:", len(mt))
                st.dataframe(mt, use_container_width=True)

            # Also show ALL weeks (not just end-of-month)
            st.subheader("All Weeks Trend")
            
            # Debug: Check history before weekly_totals
            with st.expander("📊 Debug: Raw History"):
                st.write("History rows:", len(hist))
                st.write("Unique dates in history:", hist["snapshot_date"].nunique() if "snapshot_date" in hist.columns else 0)
                st.dataframe(hist.sort_values("snapshot_date").tail(20), use_container_width=True)
            
            wt_all = weekly_totals(hist)
            
            with st.expander("📊 Debug: Weekly Totals"):
                st.write("Weekly data rows:", len(wt_all))
                st.write("Unique weeks:", wt_all["snapshot_date"].nunique() if len(wt_all) > 0 else 0)
                st.dataframe(wt_all, use_container_width=True)
            
            if len(wt_all) > 1:
                fig_all_total = px.line(
                    wt_all,
                    x="snapshot_date",
                    y="totalValue",
                    markers=True,
                    labels={"totalValue": "Total Pipeline ($)", "snapshot_date": "Week Ending"}
                )
                fig_all_total.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>Total: %{y:$,.0f}<extra></extra>")
                st.plotly_chart(fig_all_total, use_container_width=True)

                # All weeks mix (absolute)
                mix_all = wt_all.melt(
                    id_vars=["snapshot_date"],
                    value_vars=["active", "croUpdate", "directUpdate", "hold"],
                    var_name="status",
                    value_name="value"
                )
                status_labels_all = {
                    "active": "Active",
                    "croUpdate": "CRO Update",
                    "directUpdate": "Direct Update",
                    "hold": "On Hold"
                }
                mix_all["status"] = mix_all["status"].map(status_labels_all)

                fig_all_mix = px.line(
                    mix_all,
                    x="snapshot_date",
                    y="value",
                    color="status",
                    markers=True,
                    labels={"value": "Value ($)", "snapshot_date": "Week Ending", "status": "Status"}
                )
                fig_all_mix.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>%{legendgroup}: %{y:$,.0f}<extra></extra>")
                st.plotly_chart(fig_all_mix, use_container_width=True)

                with st.expander("Show all weeks data"):
                    st.dataframe(wt_all, use_container_width=True)
            
            st.divider()

            # Total pipeline trend (end-of-month only)
            fig_total = px.line(mt, x="month", y="totalValue", markers=True, labels={"totalValue":"Total Pipeline ($) - End of Month"})
            fig_total.update_traces(hovertemplate="<b>%{x}</b><br>Total: %{y:$,.0f}<extra></extra>")
            st.plotly_chart(fig_total, use_container_width=True)
            
            # Mix trend - Absolute values
            st.subheader("Pipeline Mix by Status (Absolute Values)")
            mix_abs = mt.melt(
                id_vars=["month"],
                value_vars=["active", "croUpdate", "directUpdate", "hold"],
                var_name="status",
                value_name="value",
            )
            status_labels = {
                "active": "Active",
                "croUpdate": "CRO Update",
                "directUpdate": "Direct Update",
                "hold": "On Hold"
            }
            mix_abs["status"] = mix_abs["status"].map(status_labels)

            fig_mix_abs = px.line(
                mix_abs,
                x="month",
                y="value",
                color="status",
                markers=True,
                labels={"value": "Value ($)", "month": "Month", "status": "Status"}
            )
            fig_mix_abs.update_traces(hovertemplate="<b>%{x}</b><br>%{legendgroup}: %{y:$,.0f}<extra></extra>")
            st.plotly_chart(fig_mix_abs, use_container_width=True)

            # Active & Updates Needed Trend (easier to see individual changes)
            st.subheader("Active & Updates Needed Trend")
            active_updates = mt.melt(
                id_vars=["month"],
                value_vars=["active", "updatesNeeded"],
                var_name="metric",
                value_name="value",
            )
            metric_labels_au = {"active": "Active", "updatesNeeded": "Updates Needed"}
            active_updates["metric"] = active_updates["metric"].map(metric_labels_au)

            fig_au = px.line(
                active_updates,
                x="month",
                y="value",
                color="metric",
                markers=True,
                labels={"value": "Value ($)", "month": "Month", "metric": ""}
            )
            fig_au.update_traces(hovertemplate="<b>%{x}</b><br>%{legendgroup}: %{y:$,.0f}<extra></extra>")
            st.plotly_chart(fig_au, use_container_width=True)

            # Mix trend (%)
            st.subheader("Pipeline Mix by Status (%)")
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

# ----------------------------
# History & Debug Info (Bottom of page)
# ----------------------------
st.divider()

st.subheader("History check")
st.write("History rows", len(hist))
st.write("Unique snapshot dates", hist["snapshot_date"].nunique() if "snapshot_date" in hist.columns else "missing")
st.dataframe(
    hist.sort_values("snapshot_date").tail(10),
    use_container_width=True
)

# DEBUG: Peek at raw Google Sheet data
with st.expander("🔍 DEBUG: Raw Google Sheet Data"):
    try:
        raw_sheet = fetch_google_sheet_data()
        st.write(f"Rows from sheet: {len(raw_sheet)}")
        if len(raw_sheet) > 0:
            st.write(f"Columns: {raw_sheet.columns.tolist()}")
            st.write(f"Unique snapshot_date values in sheet: {raw_sheet['snapshot_date'].nunique() if 'snapshot_date' in raw_sheet.columns else 'N/A'}")
            st.dataframe(raw_sheet.sort_values("snapshot_date"), use_container_width=True)
        else:
            st.write("Sheet returned empty")
    except Exception as e:
        st.error(f"Could not fetch sheet: {e}")
