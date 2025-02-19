import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from alpaca_trade_api import REST
from datetime import datetime, timedelta
import time
import pandas as pd
import numpy as np

# ----------------------------
# Custom CSS for a Modern Look
# ----------------------------
custom_css = """
<style>
/* Overall background and font */
body {
    background-color: #f0f2f6;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    color: #333;
}
/* Title styling */
h1, h2, h3, h4, h5, h6 {
    color: #2c3e50;
}
/* Button styling */
.stButton>button {
    background-color: #2c3e50;
    color: white;
    border-radius: 5px;
    padding: 0.5em 1em;
    border: none;
}
.stButton>button:hover {
    background-color: #34495e;
}
/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #ecf0f1;
    border-right: 1px solid #bdc3c7;
}
/* Input and text styling */
div.stTextInput>div>input {
    border-radius: 4px;
    border: 1px solid #bdc3c7;
    padding: 0.5em;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ----------------------------
# Helper: Rerun the App if Possible
# ----------------------------
def rerun_app():
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.write("Please refresh the page manually to continue.")

# ----------------------------
# FinBERT Model Loading
# ----------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
    return tokenizer, model

tokenizer, model = load_finbert()
labels = ["positive", "negative", "neutral"]

def estimate_sentiment(news_list):
    """
    Given a list of news headlines, tokenize and pass them through FinBERT,
    then return the sentiment (and its probability) that best represents the overall sentiment.
    """
    if news_list:
        tokens = tokenizer(news_list, return_tensors="pt", padding=True).to(device)
        logits = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"]
        summed = torch.sum(logits, dim=0)
        result = torch.nn.functional.softmax(summed, dim=-1)
        best_idx = torch.argmax(result)
        probability = result[best_idx]
        sentiment = labels[best_idx]
        return probability.item(), sentiment
    else:
        return 0, labels[-1]

# ----------------------------
# Alpaca & Trading Bot Setup
# ----------------------------
# Real API credentials for your paper trading account:
API_KEY = "PKCLIIWUQKZIRDLCZJFD"
API_SECRET = "8gnyNqo5FKDcoDjmI36m9oar2KdQeMooQSwDFdTu"
# Note: Do not include '/v2' in BASE_URL.
BASE_URL = "https://paper-api.alpaca.markets"

class MLTrader:
    def __init__(self, symbol="SPY", cash_at_risk=0.5):
        self.symbol = symbol
        self.cash_at_risk = cash_at_risk
        self.last_trade = None
        self.api = REST(API_KEY, API_SECRET, BASE_URL)
    
    def position_sizing(self):
        try:
            account = self.api.get_account()
            cash = float(account.cash)
        except Exception as e:
            st.error(f"Error fetching account info: {e}")
            cash = 0
        try:
            # Get the latest 1-minute bar
            bars = self.api.get_bars(self.symbol, timeframe="1Min", limit=1)
            if bars and len(bars) > 0:
                last_price = bars[0].c
            else:
                last_price = 0
        except Exception as e:
            st.error(f"Error fetching market data: {e}")
            last_price = 0
        
        quantity = round(cash * self.cash_at_risk / last_price, 0) if last_price > 0 else 0
        return cash, last_price, quantity

    def get_dates(self):
        today = datetime.now()
        three_days_prior = today - timedelta(days=3)
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')

    def get_sentiment(self):
        today, three_days_prior = self.get_dates()
        try:
            news = self.api.get_news(self.symbol, start=three_days_prior, end=today)
            headlines = [ev.__dict__["_raw"]["headline"] for ev in news]
        except Exception as e:
            st.error(f"Error fetching news: {e}")
            headlines = []
        probability, sentiment = estimate_sentiment(headlines)
        return probability, sentiment

    def execute_trade(self):
        cash, last_price, quantity = self.position_sizing()
        probability, sentiment = self.get_sentiment()
        order_details = ""
        trade_executed = False

        if cash > last_price and quantity > 0:
            if sentiment == "positive" and probability > 0.999:
                if self.last_trade == "sell":
                    order_details += "Closing short position. "
                    try:
                        st.info("Closing short position...")
                        self.api.close_position(self.symbol)
                    except Exception as e:
                        st.error(f"Error closing short position: {e}")
                order_details += (f"Placing BUY order for {quantity} shares of {self.symbol} at ${last_price:.2f}. "
                                  f"Target Profit: ${last_price*1.20:.2f}, Stop Loss: ${last_price*0.95:.2f}.")
                try:
                    self.api.submit_order(
                        symbol=self.symbol,
                        qty=quantity,
                        side='buy',
                        type='market',
                        time_in_force='gtc'
                    )
                    st.success("Buy order submitted.")
                except Exception as e:
                    st.error(f"Error submitting buy order: {e}")
                self.last_trade = "buy"
                trade_executed = True
            elif sentiment == "negative" and probability > 0.999:
                if self.last_trade == "buy":
                    order_details += "Closing long position. "
                    try:
                        st.info("Closing long position...")
                        self.api.close_position(self.symbol)
                    except Exception as e:
                        st.error(f"Error closing long position: {e}")
                order_details += (f"Placing SELL order for {quantity} shares of {self.symbol} at ${last_price:.2f}. "
                                  f"Target Profit: ${last_price*0.80:.2f}, Stop Loss: ${last_price*1.05:.2f}.")
                try:
                    self.api.submit_order(
                        symbol=self.symbol,
                        qty=quantity,
                        side='sell',
                        type='market',
                        time_in_force='gtc'
                    )
                    st.success("Sell order submitted.")
                except Exception as e:
                    st.error(f"Error submitting sell order: {e}")
                self.last_trade = "sell"
                trade_executed = True
            else:
                order_details = (f"No clear trading signal. Current sentiment: {sentiment} "
                                 f"with probability {probability:.2f}.")
        else:
            order_details = "Insufficient cash or market data unavailable for trade execution."
        
        return trade_executed, order_details

# ----------------------------
# Backtesting Functions
# ----------------------------
def backtest_strategy(symbol, start_date, end_date, initial_cash=10000, cash_at_risk=0.5):
    """
    Backtest a sentiment-based strategy over the specified period.
    For each trading day, fetch historical bars and news data from Alpaca,
    compute sentiment using FinBERT, and simulate trades accordingly.
    """
    alpaca_api = REST(API_KEY, API_SECRET, BASE_URL)
    # Convert dates to string format
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    try:
        bars = alpaca_api.get_bars(symbol, timeframe="1Day", start=start_str, end=end_str).df
    except Exception as e:
        st.error(f"Error fetching historical bars: {e}")
        return None, None

    if bars.empty:
        st.error("No historical data available for this period.")
        return None, None

    bars.sort_index(inplace=True)
    portfolio = []
    cash = initial_cash
    position = 0
    entry_price = 0

    # Iterate over each trading day in the historical period
    for current_date in bars.index:
        day_str = current_date.strftime("%Y-%m-%d")
        day_bar = bars.loc[current_date]
        close_price = day_bar['close']
        
        # Fetch news for the day (from the market open of the day until the next day)
        try:
            news = alpaca_api.get_news(symbol, start=day_str, end=(current_date + timedelta(days=1)).strftime("%Y-%m-%d"))
            headlines = [ev.__dict__["_raw"]["headline"] for ev in news]
        except Exception as e:
            headlines = []
        
        probability, sentiment = estimate_sentiment(headlines)
        
        # Trading logic: if sentiment is strongly positive, buy; if strongly negative, sell.
        if sentiment == "positive" and probability > 0.999 and position == 0:
            invest_amount = cash * cash_at_risk
            if close_price > 0:
                shares = invest_amount / close_price
                position = shares
                cash -= shares * close_price
                entry_price = close_price
        elif sentiment == "negative" and probability > 0.999 and position > 0:
            cash += position * close_price
            position = 0
            entry_price = 0

        portfolio_value = cash + position * close_price
        portfolio.append({
            "date": current_date,
            "portfolio_value": portfolio_value,
            "cash": cash,
            "position": position,
            "close": close_price,
            "sentiment": sentiment,
            "probability": probability
        })
    
    df_portfolio = pd.DataFrame(portfolio).set_index("date")
    return df_portfolio, bars

def compute_performance(df_portfolio, initial_cash=10000):
    final_value = df_portfolio["portfolio_value"].iloc[-1]
    total_return = (final_value - initial_cash) / initial_cash
    daily_returns = df_portfolio["portfolio_value"].pct_change().dropna()
    annualized_return = (final_value / initial_cash) ** (252 / len(daily_returns)) - 1 if len(daily_returns) > 0 else 0
    cumulative = df_portfolio["portfolio_value"].cummax()
    drawdown = (df_portfolio["portfolio_value"] - cumulative) / cumulative
    max_drawdown = drawdown.min()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() != 0 else 0
    return {
        "Total Return": f"{total_return*100:.2f}%",
        "Annualized Return": f"{annualized_return*100:.2f}%",
        "Max Drawdown": f"{max_drawdown*100:.2f}%",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}"
    }

# ----------------------------
# Login Page
# ----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("Trading Bot Application Login")
    st.text("Please enter your credentials to continue.")
    username_input = st.text_input("Username")
    password_input = st.text_input("Password", type="password")
    if st.button("Login"):
        if username_input == "username" and password_input == "idontknow":
            st.session_state.logged_in = True
            st.success("Logged in successfully!")
            rerun_app()
        else:
            st.error("Invalid username or password.")
    st.stop()

# ----------------------------
# Main Navigation & Pages
# ----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Trading", "Backtesting", "Settings"])

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    rerun_app()

if page == "Dashboard":
    st.title("Dashboard")
    st.write("Welcome to your Trading Bot Dashboard.")
    
    st.subheader("Market Sentiment Analysis")
    trader = MLTrader()  # Instantiate the trading bot
    with st.spinner("Fetching news and estimating sentiment..."):
        probability, sentiment = trader.get_sentiment()
        time.sleep(1)
    st.write(f"**Sentiment:** {sentiment}")
    st.write(f"**Confidence:** {probability:.2f}")
    
    st.subheader("Account Information")
    try:
        account = trader.api.get_account()
        st.write(f"**Cash Available:** ${account.cash}")
        st.write(f"**Portfolio Value:** ${account.portfolio_value}")
    except Exception as e:
        st.error(f"Could not retrieve account info: {e}")

elif page == "Trading":
    st.title("Live Trading")
    st.write("Execute trades based on market sentiment analysis.")
    trader = MLTrader()
    if st.button("Execute Trading Iteration"):
        with st.spinner("Evaluating market conditions and executing trade..."):
            trade_executed, order_details = trader.execute_trade()
            time.sleep(2)
        st.write(order_details)
        if trade_executed:
            st.success("Trade executed.")
        else:
            st.info("No trade was executed at this time.")

elif page == "Backtesting":
    st.title("Backtesting")
    st.write("Run a fully functional backtest on historical market data using Alpaca data.")
    start_date = st.date_input("Start Date", datetime(2020, 1, 1))
    end_date = st.date_input("End Date", datetime(2023, 12, 31))
    symbol = st.text_input("Trading Symbol", value="SPY")
    initial_cash = st.number_input("Initial Cash", value=10000)
    cash_at_risk = st.slider("Cash at Risk (fraction)", 0.0, 1.0, 0.5)
    
    if st.button("Run Backtest"):
        with st.spinner("Running backtest... This may take a few moments."):
            df_portfolio, bars = backtest_strategy(symbol, start_date, end_date, initial_cash, cash_at_risk)
            if df_portfolio is not None:
                performance = compute_performance(df_portfolio, initial_cash)
                time.sleep(1)
        st.success("Backtest complete.")
        st.subheader("Performance Metrics")
        st.table(pd.DataFrame(performance.items(), columns=["Metric", "Value"]))
        st.subheader("Equity Curve")
        st.line_chart(df_portfolio["portfolio_value"])

elif page == "Settings":
    st.title("Settings")
    st.write("Configure your trading bot parameters below.")
    symbol = st.text_input("Trading Symbol", value="SPY")
    cash_at_risk = st.slider("Cash at Risk (fraction)", 0.0, 1.0, 0.5)
    st.write("These settings can be used to reinitialize your trading bot instance.")
    if st.button("Update Settings"):
        st.success("Settings updated!")
