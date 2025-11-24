from watchlist_app import Watchlist
import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
import pandas as pd
from pandas import CategoricalDtype
import numpy as np

st.set_page_config(page_title="Options Strategy Dashboard", layout="wide")


### FUNCTIONS ###


def filter_by_moneyness(df, option_type, spot, n):
    if df.empty or n is None or spot is None:
        return df.sort_values("strike")

    df = df.copy()

    if option_type == "Calls":
        df["itm"] = df["strike"] < spot
    elif option_type == "Puts":
        df["itm"] = df["strike"] > spot
    else:
        raise ValueError("option_type must be 'Calls' or 'Puts'")

    itm_df = df[df["itm"]].copy()
    otm_df = df[~df["itm"]].copy()

    # Sort by closeness to spot within each group
    if not itm_df.empty:
        itm_df["dist"] = (itm_df["strike"] - spot).abs()
        itm_df = itm_df.sort_values("dist")
    if not otm_df.empty:
        otm_df["dist"] = (otm_df["strike"] - spot).abs()
        otm_df = otm_df.sort_values("dist")

    n_itm = n // 2
    n_otm = n - n_itm

    itm_sel = itm_df.head(n_itm)
    otm_sel = otm_df.head(n_otm)

    combined = pd.concat([itm_sel, otm_sel])
    for col in ["itm", "dist"]:
        if col in combined.columns:
            combined = combined.drop(columns=[col])

    return combined.sort_values("strike")


def style_strike_blocks(df: pd.DataFrame):
    last_strike = None
    toggle = False
    styles = {}

    for idx in df.index:
        # idx is (strike, side)
        if isinstance(idx, tuple):
            strike = idx[0]
        else:
            strike = idx

        if strike != last_strike:
            toggle = not toggle  # flip color when strike changes
            last_strike = strike

        color = "#1A263D" if toggle else "#10131A"
        styles[idx] = [f"background-color: {color}"] * df.shape[1]

    return pd.DataFrame.from_dict(styles, orient="index", columns=df.columns)


# ---------- Sidebar ----------
st.sidebar.header("Settings")

# ---- Session State for Watchlist ----
if "watchlist_tickers" not in st.session_state:
    st.session_state.watchlist_tickers = ["AAPL", "NVDA", "TSLA"]

if "active_ticker" not in st.session_state:
    st.session_state.active_ticker = st.session_state.watchlist_tickers[0]

# ---- Search / Add Ticker ----
new_ticker = st.sidebar.text_input("Search / add ticker (e.g. AAPL)", value="")


if st.sidebar.button("Add to watchlist"):
    t = new_ticker.upper().strip()
    if t and t not in st.session_state.watchlist_tickers:
        st.session_state.watchlist_tickers.append(t)
        st.session_state.active_ticker = t  # auto-switch to the new one

# ---- Build Watchlist DataFrame using your Watchlist class ----
wl = Watchlist(st.session_state.watchlist_tickers)
# uses yfinance to get last_price :contentReference[oaicite:1]{index=1}
df_wl = wl.load_data()

st.sidebar.write("### Watchlist")
st.sidebar.dataframe(
    df_wl,
    use_container_width=True,
    height=300,
)

tab1, tab2 = st.tabs(["Strategy Builder", "Dashboard"])

with tab2:
    st.subheader("Dashboard")

with tab1:
    # ---- Click-to-switch using radio ----
    selected = st.sidebar.radio(
        "Click a ticker to view:",
        st.session_state.watchlist_tickers,
        index=st.session_state.watchlist_tickers.index(
            st.session_state.active_ticker),
    )

    st.session_state.active_ticker = selected

    # Use this as the main ticker for the rest of the app
    ticker_symbol = st.session_state.active_ticker

    ticker = yf.Ticker(ticker_symbol)

    options_quotes, strat_builder = st.columns(2)
    price_history, visualizer = st.columns(2)

    if ticker:
        # Options Quotes Section
        with options_quotes:
            st.subheader("Options Quotes")
            expirations = list(ticker.options)

            if not expirations:
                st.info("No options data available for this ticker.")
            else:
                select_exp, select_opt, select_strikes = st.columns(3)
                with select_exp:
                    selected_exp = st.selectbox("Expiration", expirations)

                try:
                    chain = ticker.option_chain(selected_exp)
                except Exception:
                    st.error(f"Could not load options for {selected_exp}.")
                    chain = None

                spot_hist = ticker.history("1d")
                if spot_hist.empty:
                    spot_price = None
                else:
                    spot_price = float(spot_hist["Close"].iloc[-1])

                if chain is not None:
                    with select_opt:
                        view_mode = st.radio(
                            "Option Type", ["Calls", "Puts", "Both"], horizontal=True)
                    num_choices = ["6", "10", "20", "All"]
                    with select_strikes:
                        num_label = st.selectbox(
                            "Number of strikes around ATM", num_choices, index=1)
                    num_strikes = None if num_label == "All" else int(
                        num_label)

                    calls = chain.calls.copy()
                    puts = chain.puts.copy()

                    if spot_price is None:
                        calls_filtered = calls.sort_values("strike")
                        puts_filtered = puts.sort_values("strike")
                    else:
                        calls_filtered = (
                            filter_by_moneyness(
                                calls, "Calls", spot_price, num_strikes)
                            if not calls.empty
                            else calls
                        )
                        puts_filtered = (
                            filter_by_moneyness(
                                puts, "Puts", spot_price, num_strikes)
                            if not puts.empty
                            else puts
                        )
                    base_cols_single = [
                        "strike",
                        "lastPrice",
                        "bid",
                        "ask",
                        "change",
                        "percentChange",
                        "volume",
                        "openInterest",
                        "impliedVolatility",
                    ]

                    if view_mode in ["Calls", "Puts"]:
                        df = calls_filtered if view_mode == "Calls" else puts_filtered

                        if df.empty:
                            st.warning(
                                f"No {view_mode.lower()} found for this expiration.")
                        else:
                            cols_to_show = [
                                c for c in base_cols_single if c in df.columns]

                            st.dataframe(
                                df[cols_to_show].sort_values("strike"),
                                use_container_width=True,
                                height=430,
                            )

                    else:  # Both
                        if calls_filtered.empty and puts_filtered.empty:
                            st.warning(
                                "No calls or puts found for this expiration.")
                        else:
                            atm_strikes = sorted(
                                set(calls_filtered["strike"].tolist())
                                | set(puts_filtered["strike"].tolist())
                            )

                            calls_view = calls[calls["strike"].isin(
                                atm_strikes)].copy()
                            puts_view = puts[puts["strike"].isin(
                                atm_strikes)].copy()

                            calls_view["side"] = "Call"
                            puts_view["side"] = "Put"

                            base_cols_both = [
                                "lastPrice",
                                "bid",
                                "ask",
                                "change",
                                "percentChange",
                                "volume",
                                "openInterest",
                                "impliedVolatility",
                            ]

                            cols_for_both = [
                                c
                                for c in base_cols_both
                                if c in calls_view.columns and c in puts_view.columns
                            ]

                            calls_view = calls_view[[
                                "strike", "side"] + cols_for_both]
                            puts_view = puts_view[[
                                "strike", "side"] + cols_for_both]

                            stacked = pd.concat(
                                [calls_view, puts_view], ignore_index=True)

                            side_cat = CategoricalDtype(
                                categories=["Call", "Put"], ordered=True)
                            stacked["side"] = stacked["side"].astype(side_cat)

                            stacked = stacked.sort_values(["strike", "side"])
                            stacked = stacked.set_index(["strike", "side"])

                            df_display = stacked[cols_for_both]
                            styled = df_display.style.apply(
                                style_strike_blocks, axis=None)

                            st.dataframe(
                                styled,
                                use_container_width=True,
                                height=430,
                            )

        # Strategy Builder
        with strat_builder:
            st.subheader("Strategy Builder")
            st.button("Add Leg")

        # Stock Price History
        with price_history:
            st.subheader(f"{ticker_symbol} Price History")
            period = "6mo"
            col1, col2, col3, col4, col5, col6, col7 = st.columns(
                [1, 1, 1, 1, 4, 1, 3])
            with col1:
                if st.button(label="1M"):
                    period = "1mo"
            with col2:
                if st.button(label="3M"):
                    period = "3mo"
            with col3:
                if st.button(label="6M"):
                    period = "6mo"
            with col4:
                if st.button(label="1Y"):
                    period = "1y"
            with col5:
                if st.button(label="2Y"):
                    period = "2y"
            hist = ticker.history(period)
            if hist.empty:
                st.warning("No price data found. Try a different ticker.")
            else:
                fig = go.Figure(data=[go.Candlestick(
                    x=hist.index,
                    open=hist["Open"],
                    high=hist["High"],
                    low=hist["Low"],
                    close=hist["Close"],
                    name=ticker_symbol,
                )])
                with col6:
                    if st.button("Line"):
                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=hist.index,
                                y=hist["Close"],
                                mode="lines",
                                name="Close",

                            )
                        )

                with col7:
                    if st.button("Candles"):
                        fig = go.Figure(data=[go.Candlestick(
                            x=hist.index,
                            open=hist["Open"],
                            high=hist["High"],
                            low=hist["Low"],
                            close=hist["Close"],
                            name=ticker_symbol,
                        )])

                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Price",
                    margin=dict(l=40, r=20, t=40, b=40),
                    xaxis_rangeslider_visible=False,
                )
                st.plotly_chart(fig, use_container_width=True)
        # Visualize Strategy
        with visualizer:
            st.subheader("Strategy Visualizer")
