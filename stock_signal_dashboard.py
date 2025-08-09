#!/usr/bin/env python3

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

import pandas as pd
import yfinance as yf
# import pandas_ta as ta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

# Optional: fundamentals from NSE if available
try:
    from nsetools import Nse  # type: ignore
    NSE_AVAILABLE = True
except Exception:
    NSE_AVAILABLE = False

# Email
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Plotting
import plotly.graph_objects as go
import plotly.io as pio

# Env vars
from dotenv import load_dotenv


# === CONFIGURATION ===
DEFAULT_STOCK_LIST = ["TCS.NS", "INFY.NS", "HDFCBANK.NS"]
SHORT_WINDOW = 20  # SMA lookback for trend/entry
LONG_TREND_WINDOW = 200  # Higher timeframe trend filter
RSI_LENGTH = 14
RSI_THRESHOLD = 30  # Oversold threshold
VOLUME_WINDOW = 20
MIN_AVG_VOLUME = 100_000  # Skip illiquid names
USE_ADJUSTED = True  # Use Adjusted Close for indicators
PERIOD = "24mo"
INTERVAL = "1d"
MAX_DOWNLOAD_ATTEMPTS = 4

# Output
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "stock_outputs")
COMBINED_DASHBOARD = os.path.join(OUTPUT_DIR, "dashboard_combined.html")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "summary.csv")
OPEN_BROWSER = False  # Set True to open combined HTML after run

# Email (load from env vars)
SEND_EMAIL = os.environ.get("SEND_EMAIL", "false").lower() in {"1", "true", "yes"}
EMAIL_FROM = os.environ.get("EMAIL_FROM", "")
EMAIL_TO = os.environ.get("EMAIL_TO", "")
EMAIL_SUBJECT = os.environ.get("EMAIL_SUBJECT", "Stock Buy Signal Alert")
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
EMAIL_USER = os.environ.get("EMAIL_USER", "")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("stock_signal")


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sanitize_ticker(ticker: str) -> str:
    # Ensure Yahoo Finance NSE suffix .NS for Indian equities if user omitted it
    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return ticker
    # Heuristic: uppercase letters only -> assume NSE symbol
    if ticker.isalpha():
        return f"{ticker}.NS"
    return ticker


class DownloadError(Exception):
    pass


@retry(
    reraise=True,
    stop=stop_after_attempt(MAX_DOWNLOAD_ATTEMPTS),
    wait=wait_exponential(multiplier=0.8, min=1, max=10),
    retry=retry_if_exception_type(DownloadError),
)
def download_ohlcv(ticker: str, period: str, interval: str, auto_adjust: bool) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=auto_adjust, progress=False)
    except Exception as exc:  # network/api failures
        raise DownloadError(str(exc))

    if df is None or df.empty or "Close" not in df.columns:
        raise DownloadError(f"No data returned for {ticker}")
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    price_series = df["Adj Close"] if (USE_ADJUSTED and "Adj Close" in df.columns) else df["Close"]

    df = df.copy()
    # Native SMA
    df["SMA_SHORT"] = price_series.rolling(window=SHORT_WINDOW, min_periods=SHORT_WINDOW).mean()
    df["SMA_LONG"] = price_series.rolling(window=LONG_TREND_WINDOW, min_periods=LONG_TREND_WINDOW).mean()

    # Native RSI (Wilder's smoothing)
    delta = price_series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / RSI_LENGTH, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / RSI_LENGTH, adjust=False).mean()
    rs = roll_up / roll_down
    df["RSI"] = 100 - (100 / (1 + rs))

    # Volume average
    if "Volume" in df.columns:
        df["VOL_SMA"] = df["Volume"].rolling(window=VOLUME_WINDOW, min_periods=VOLUME_WINDOW).mean()
    else:
        df["VOL_SMA"] = pd.NA

    # Slope of short SMA
    lookback_slope = max(3, SHORT_WINDOW // 5)
    df["SMA_SHORT_SLOPE"] = df["SMA_SHORT"] - df["SMA_SHORT"].shift(lookback_slope)

    return df


def generate_buy_signal(df: pd.DataFrame) -> pd.Series:
    # Conditions for higher quality signals:
    # 1) RSI crossing up the threshold (from below to above)
    # 2) Price above short SMA and short SMA slope positive
    # 3) Above long-term trend (SMA_LONG) filter
    # 4) Sufficient liquidity via average volume
    price_series = df["Adj Close"] if (USE_ADJUSTED and "Adj Close" in df.columns) else df["Close"]

    rsi_cross_up = (df["RSI"].shift(1) < RSI_THRESHOLD) & (df["RSI"] >= RSI_THRESHOLD)
    trend_ok = (price_series > df["SMA_SHORT"]) & (df["SMA_SHORT_SLOPE"] > 0)
    long_trend_ok = price_series > df["SMA_LONG"]
    liquid_ok = df["VOL_SMA"].fillna(0) >= MIN_AVG_VOLUME

    return rsi_cross_up & trend_ok & long_trend_ok & liquid_ok


def fetch_fundamentals(ticker: str) -> Dict[str, Optional[Any]]:
    # Try Yahoo first (fast_info / get_info), then NSE as fallback if available
    pe_ratio: Optional[float] = None
    eps: Optional[float] = None
    try:
        t = yf.Ticker(ticker)
        # fast_info is lightweight but may not have PE/EPS
        fi = getattr(t, "fast_info", None)
        if fi:
            pe_ratio = getattr(fi, "trailing_pe", None) or getattr(fi, "pe_ratio", None)
        # get_info can be slow; use as fallback
        if pe_ratio is None or eps is None:
            info = t.get_info()  # noqa: F841
            if isinstance(info, dict):
                pe_ratio = pe_ratio or info.get("trailingPE") or info.get("forwardPE")
                eps = info.get("trailingEps") or info.get("epsTrailingTwelveMonths")
    except Exception:
        pass

    if (pe_ratio is None or eps is None) and NSE_AVAILABLE:
        try:
            base = ticker.split(".")[0].lower()
            nse_client = Nse()
            q = nse_client.get_quote(base)
            if isinstance(q, dict):
                pe_ratio = pe_ratio or q.get("p/e")
                eps = eps or q.get("eps")
        except Exception:
            pass

    return {"P/E Ratio": pe_ratio, "EPS": eps}


def plot_dashboard(ticker: str, df: pd.DataFrame, buy_signal_series: pd.Series, out_dir: str) -> str:
    price_series = df["Adj Close"] if (USE_ADJUSTED and "Adj Close" in df.columns) else df["Close"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=price_series, mode="lines", name="Price"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_SHORT"], mode="lines", name=f"SMA {SHORT_WINDOW}"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_LONG"], mode="lines", name=f"SMA {LONG_TREND_WINDOW}", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines", name=f"RSI {RSI_LENGTH}", yaxis="y2"))

    # Mark buy signals on price chart
    if buy_signal_series.any():
        signal_points = df.loc[buy_signal_series]
        fig.add_trace(
            go.Scatter(
                x=signal_points.index,
                y=price_series.loc[buy_signal_series],
                mode="markers",
                name="Buy Signal",
                marker=dict(size=9, color="green", symbol="triangle-up"),
            )
        )

    fig.update_layout(
        title=f"{ticker} | Price, SMAs, RSI & Buy Signals",
        xaxis_title="Date",
        yaxis_title="Price",
        yaxis2=dict(title="RSI", overlaying="y", side="right", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
    )

    filename = os.path.join(out_dir, f"dashboard_{ticker.replace('.', '_')}.html")
    pio.write_html(fig, file=filename, auto_open=False, include_plotlyjs="cdn")

    # Return body content for combined HTML
    html_str = pio.to_html(fig, include_plotlyjs=False, full_html=False)
    return html_str


def send_email_summary(html_table: str, extra_html: Optional[str] = None) -> None:
    if not (EMAIL_FROM and EMAIL_TO and EMAIL_USER and EMAIL_PASSWORD):
        logger.warning("Email not sent: missing EMAIL_* environment variables")
        return

    message = MIMEMultipart("alternative")
    message["Subject"] = EMAIL_SUBJECT
    message["From"] = EMAIL_FROM
    message["To"] = EMAIL_TO

    plain_text = "Please find the stock buy signal report below."
    message.attach(MIMEText(plain_text, "plain"))
    html_full = f"<html><body>{html_table}{extra_html or ''}</body></html>"
    message.attach(MIMEText(html_full, "html"))

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_FROM, EMAIL_TO.split(","), message.as_string())


def main() -> int:
    load_dotenv()  # load .env if present
    ensure_output_dir(OUTPUT_DIR)

    # Stock list from env or default
    env_list = os.environ.get("STOCK_LIST", "").strip()
    tickers = [sanitize_ticker(t.strip()) for t in (env_list.split(",") if env_list else DEFAULT_STOCK_LIST)]

    results: List[Dict[str, Any]] = []
    combined_divs: List[str] = []

    logger.info("Fetching data and computing signals...")
    for ticker in tqdm(tickers, desc="Tickers"):
        try:
            df = download_ohlcv(ticker, PERIOD, INTERVAL, auto_adjust=USE_ADJUSTED)
            df = compute_indicators(df)
            df = df.dropna(subset=["SMA_SHORT", "SMA_LONG", "RSI"])  # ensure indicators present
            if df.empty:
                logger.warning(f"Indicators not available for {ticker}; skipping")
                continue

            buy_series = generate_buy_signal(df)
            latest = df.iloc[-1]
            fundamentals = fetch_fundamentals(ticker)

            last_close = float((df["Adj Close"] if (USE_ADJUSTED and "Adj Close" in df.columns) else df["Close"]).iloc[-1])
            out_row: Dict[str, Any] = {
                "Stock": ticker,
                "Close": round(last_close, 2),
                "RSI": round(float(latest["RSI"]), 2),
                f"SMA{SHORT_WINDOW}": round(float(latest["SMA_SHORT"]), 2),
                f"SMA{LONG_TREND_WINDOW}": round(float(latest["SMA_LONG"]), 2),
                "AvgVol20": int(float(latest.get("VOL_SMA", 0)) if pd.notna(latest.get("VOL_SMA", pd.NA)) else 0),
                "Buy Signal": "YES" if bool(buy_series.iloc[-1]) else "NO",
            }
            out_row.update(fundamentals)
            results.append(out_row)

            # Plot per-stock dashboard and collect for combined
            try:
                div = plot_dashboard(ticker, df, buy_series, OUTPUT_DIR)
                combined_divs.append(div)
            except Exception as plot_exc:
                logger.warning(f"Plot error for {ticker}: {plot_exc}")

        except Exception as e:
            logger.warning(f"Failed to process {ticker}: {e}")
            continue

    summary_df = pd.DataFrame(results)
    if not summary_df.empty:
        # Sort: buy signals first, then by RSI ascending
        sort_cols = ["Buy Signal", "RSI"]
        summary_df["Buy Signal Rank"] = summary_df["Buy Signal"].eq("YES").astype(int)
        summary_df = summary_df.sort_values(["Buy Signal Rank", "RSI"], ascending=[False, True]).drop(columns=["Buy Signal Rank"])  # type: ignore

        # Save CSV
        summary_df.to_csv(SUMMARY_CSV, index=False)

        # Print to stdout
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 150)
        print("\n=== Summary Report ===")
        print(summary_df.to_string(index=False))

        # Combined HTML
        combined_html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Stock Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial, 'Noto Sans', 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol'; padding: 16px; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 24px; }
    th, td { border: 1px solid #e5e7eb; padding: 8px 10px; text-align: right; }
    th { background: #f9fafb; text-align: right; }
    td:first-child, th:first-child { text-align: left; }
    .chart { margin-bottom: 40px; }
  </style>
</head>
<body>
  <h2>Summary</h2>
  {summary_table}
  <h2>Charts</h2>
  {charts}
</body>
</html>
""".strip()
        html_table = summary_df.to_html(index=False)
        final_html = combined_html.format(summary_table=html_table, charts="\n".join(combined_divs))
        with open(COMBINED_DASHBOARD, "w", encoding="utf-8") as f:
            f.write(final_html)

        # Email
        if SEND_EMAIL:
            try:
                send_email_summary(html_table, extra_html=None)
                logger.info("Email sent successfully")
            except Exception as e:
                logger.warning(f"Failed to send email: {e}")

        # Optionally open in browser
        if OPEN_BROWSER:
            import webbrowser
            webbrowser.open("file://" + os.path.realpath(COMBINED_DASHBOARD))

    else:
        logger.info("No results to report.")

    logger.info(f"Outputs: {SUMMARY_CSV} and {COMBINED_DASHBOARD}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted by user")
        sys.exit(130)