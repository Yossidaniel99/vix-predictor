"""
VIX Market-Timing Signal — Streamlit app

Deploy free on Streamlit Community Cloud:
  https://share.streamlit.io  →  "New app"  →  point at your GitHub repo

Repo layout (commit these two files):
  streamlit_app.py                        ← THIS FILE
  vix_model.py                            ← rename of vix_signal_app_FOR_BASE44.py
  requirements.txt                        ← pandas / numpy / xgboost / scikit-learn
                                            requests / yfinance / streamlit

Streamlit Cloud auto-installs from requirements.txt and runs streamlit_app.py.
The resulting URL works on any phone browser; on Android use Chrome →
"Add to Home Screen" to install as a PWA.

DATA SOURCE
  - Set your Twelve Data API key as a Streamlit secret named TWELVEDATA_KEY
    (Streamlit Cloud → app settings → Secrets).  Free tier 800 calls/day works.
  - Falls back to yfinance if the secret is absent.

PERSISTENCE
  - Free tier has no persistent disk.  History is kept in a small CSV in this
    repo; a daily GitHub Action (workflow file shown at end of this comment)
    is the cleanest way to write it.  For now, the app shows the latest
    computed signal on demand.
"""

import os, json
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np

# Import the modeling code (rename your downloaded
# vix_signal_app_FOR_BASE44.py to vix_model.py and put it next to this file).
from vix_model import (
    get_signal_live,
    get_signal_yfinance,
    format_signal,
    STRICT_RULE,
    LOOSE_RULE,
)


# ── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VIX Signal",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Mobile-friendly compact CSS
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    .big-badge {
        font-size: 2.0rem; font-weight: 700; text-align: center;
        padding: 1.2rem; border-radius: 1rem; margin: 0.7rem 0;
    }
    .badge-long-svix  { background: #1f8a3a; color: white; }
    .badge-long-uvix  { background: #b22222; color: white; }
    .badge-cash       { background: #6b7280; color: white; }
    .small-meta { color: #888; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ── Cached signal generation ──────────────────────────────────────────────

@st.cache_data(ttl=60*60*4, show_spinner=False)   # cache 4 hours
def fetch_signal(use_live: bool, api_key: str | None):
    if use_live and api_key:
        return get_signal_live(twelvedata_api_key=api_key)
    return get_signal_yfinance()


# ── Sidebar / Settings ────────────────────────────────────────────────────

with st.sidebar:
    st.subheader("Settings")
    threshold_choice = st.radio(
        "Threshold preset",
        options=["STRICT (B@0.70/0.030)", "LOOSER (B@0.60/0.020)"],
        index=0,
        help="STRICT = fewer, higher-conviction trades. LOOSER = more frequent."
    )
    api_key_secret = st.secrets.get("TWELVEDATA_KEY", os.environ.get("TWELVEDATA_KEY"))
    use_live = st.toggle(
        "Use Twelve Data (live)",
        value=bool(api_key_secret),
        disabled=not bool(api_key_secret),
        help="Falls back to yfinance if no API key is set."
    )
    if st.button("🔄 Refresh now", use_container_width=True):
        fetch_signal.clear()
        st.rerun()


# ── Page nav (tabs) ───────────────────────────────────────────────────────

st.title("📈 VIX Market-Timing Signal")
st.caption("v4 model — direction + magnitude + range. "
           "**Research prototype, not investment advice.**")

tab_today, tab_history, tab_about = st.tabs(["Today", "History", "About"])


# ── TAB 1: TODAY ──────────────────────────────────────────────────────────

with tab_today:
    with st.spinner("Fetching market data and running model..."):
        try:
            sig = fetch_signal(use_live=use_live, api_key=api_key_secret)
        except Exception as e:
            st.error(f"Failed to compute signal: {e}")
            st.stop()

    action = sig["action_strict"] if threshold_choice.startswith("STRICT") else sig["action_loose"]

    # Big colored badge
    if action == "LONG_SVIX":
        badge_class = "badge-long-svix"; emoji = "🟢"; label = "LONG SVIX"
    elif action == "LONG_UVIX":
        badge_class = "badge-long-uvix"; emoji = "🔴"; label = "LONG UVIX"
    else:
        badge_class = "badge-cash"; emoji = "⚪"; label = "CASH"
    st.markdown(
        f"<div class='big-badge {badge_class}'>{emoji} {label}</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"<p class='small-meta'>Signal for trading day after "
        f"<b>{sig['data_through']}</b> · computed {sig['computed_at']}</p>",
        unsafe_allow_html=True
    )

    # Market state
    col1, col2 = st.columns(2)
    col1.metric("VIX",   f"{sig['vix']:.2f}")
    col2.metric("SPX",   f"{sig['spx']:,.2f}")
    col3, col4 = st.columns(2)
    col3.metric("VIX3M", f"{sig['vix3m']:.2f}" if sig['vix3m'] else "—",
                delta=sig['term_structure'])
    col4.metric("VVIX",  f"{sig['vvix']:.2f}" if sig['vvix'] else "—")

    st.divider()
    st.subheader("Forecast — next trading day VIX move")

    fcol1, fcol2 = st.columns(2)
    fcol1.metric("P(VIX up)",   f"{sig['prob_up']*100:.1f}%")
    fcol2.metric("Expected",    f"{sig['expected_pct']:+.2f}%")
    st.caption(f"Range 10–90%: **{sig['p10_pct']:+.2f}%** to **{sig['p90_pct']:+.2f}%** "
               f"(width {sig['width']:.4f})")

    st.divider()
    with st.expander("Both threshold variants", expanded=False):
        st.write(f"**STRICT (B@0.70/0.030)** → `{sig['action_strict']}`")
        st.write(f"**LOOSER (B@0.60/0.020)** → `{sig['action_loose']}`")
        st.caption("STRICT requires both prob_up≥0.70 AND p50≥+0.030 (long), "
                   "or prob_up≤0.30 AND p50≤−0.030 (short).")

    st.divider()
    st.markdown(
        f"### What to do\n"
        f"Place the trade for **{label}** at today's market close (or first thing tomorrow). "
        f"Hold until the next signal flip — **re-check tomorrow at ~4:30pm ET**. "
        f"No stops, no targets — earlier backtests showed they hurt this model."
    )


# ── TAB 2: HISTORY ────────────────────────────────────────────────────────

with tab_history:
    st.subheader("Signal history")
    log_path = "signal_log.csv"
    if os.path.exists(log_path):
        try:
            df = pd.read_csv(log_path, parse_dates=["data_through"])
            df = df.sort_values("data_through", ascending=False)
            st.dataframe(df.head(60), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Could not read history: {e}")
    else:
        st.info("No history yet — the GitHub Action will start populating "
                "`signal_log.csv` once configured (see About tab).")

    if st.button("Append today's signal to history"):
        new_row = {
            "data_through": sig["data_through"],
            "vix":          sig["vix"],
            "vix3m":        sig["vix3m"],
            "vvix":         sig["vvix"],
            "spx":          sig["spx"],
            "prob_up":      sig["prob_up"],
            "p50_logret":   sig["p50_logret"],
            "expected_pct": sig["expected_pct"],
            "action_strict": sig["action_strict"],
            "action_loose":  sig["action_loose"],
            "computed_at":  sig["computed_at"],
        }
        new_df = pd.DataFrame([new_row])
        if os.path.exists(log_path):
            new_df.to_csv(log_path, mode="a", header=False, index=False)
        else:
            new_df.to_csv(log_path, index=False)
        st.success("Logged. Reload the page.")


# ── TAB 3: ABOUT ──────────────────────────────────────────────────────────

with tab_about:
    st.markdown("""
### How this works
- An XGBoost classifier predicts the probability that VIX rises tomorrow.
- An XGBoost quantile regressor predicts the 10/50/90 percentile of tomorrow's
  log return on VIX (i.e. the expected magnitude and range).
- Strategy B@0.70/0.030 (STRICT) goes long UVIX only when prob_up ≥ 0.70
  AND median expected move ≥ +3%, long SVIX only when prob_up ≤ 0.30 AND
  median ≤ −3%, else cash.

### Backtest summary (UVIX/SVIX, 2022-03 → 2026-04, sealed OOS)
| Strategy            | CAGR  | Sharpe | Max DD |
|---------------------|------:|-------:|-------:|
| Buy & hold UVIX     | −85%  | −0.80  | −100% |
| Buy & hold SVIX     | 17.8% | 0.59   | −67.8% |
| **STRICT B@0.70/0.030** | **26.3%** | **1.11** | **−20.7%** |

Sample size is small (4 years, 73 active trades). Forward expectation:
15-25% CAGR with single-digit drawdowns most years and occasional 20-40%
drawdowns when a regime shift catches the model wrong.

### Daily auto-update via GitHub Actions

Add `.github/workflows/daily.yml`:

```yaml
name: Daily VIX signal
on:
  schedule:
    - cron: "30 20 * * 1-5"   # 4:30 PM ET (20:30 UTC, EDT)
  workflow_dispatch:
jobs:
  signal:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt
      - env:
          TWELVEDATA_KEY: ${{ secrets.TWELVEDATA_KEY }}
        run: |
          python -c "
          import os, json, csv, datetime as dt
          from vix_model import get_signal_live
          sig = get_signal_live(os.environ['TWELVEDATA_KEY'])
          row = {
              'data_through': sig['data_through'],
              'vix': sig['vix'], 'vix3m': sig['vix3m'], 'vvix': sig['vvix'],
              'spx': sig['spx'], 'prob_up': sig['prob_up'],
              'p50_logret': sig['p50_logret'],
              'expected_pct': sig['expected_pct'],
              'action_strict': sig['action_strict'],
              'action_loose': sig['action_loose'],
              'computed_at': sig['computed_at'],
          }
          import pathlib, pandas as pd
          p = pathlib.Path('signal_log.csv')
          new = pd.DataFrame([row])
          if p.exists():
              new.to_csv(p, mode='a', header=False, index=False)
          else:
              new.to_csv(p, index=False)
          "
      - name: commit log
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add signal_log.csv
          git commit -m "daily signal" || true
          git push
```

Add a GitHub repo secret `TWELVEDATA_KEY`.  This appends a row to
`signal_log.csv` every weekday and the Streamlit app picks it up on next
visit.  Free; 2,000 GitHub Action minutes/month is way more than you need.
""")
