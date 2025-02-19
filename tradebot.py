import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from alpaca_trade_api import REST
from datetime import datetime, timedelta
import time

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
        # Sum logits across the list and compute softmax
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
BASE_URL = "https://paper-api.alpaca.markets/v2"

class MLTrader:
    def __init__(self, symbol="SPY", cash_at_risk=0.5):
        self.symbol = symbol
        self.cash_at_risk = cash_at_risk
        self.last_trade = None
        # Initialize Alpaca REST API
        self.api = REST(API_KEY, API_SECRET, BASE_URL)
    
    def position_sizing(self):
        """
        Get available cash and the latest price for position sizing.
        """
        try:
            account = self.api.get_account()
            cash = float(account.cash)
        except Exception as e:
            st.error(f"Error fetching account info: {e}")
            cash = 0
        # Fetch last trade price using the latest minute bar
        try:
            barset = self.api.get_barset(self.symbol, "minute", limit=1)
            last_price = barset[self.symbol][0].c if barset[self.symbol] else 0
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
        """
        Fetch news headlines from Alpaca between three days ago and today,
        then use FinBERT to estimate overall sentiment.
        """
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
        """
        Based on cash, market data, and sentiment, execute a trade.
        This function places real orders on your Alpaca paper account.
        """
        cash, last_price, quantity = self.position_sizing()
        probability, sentiment = self.get_sentiment()
        order_details = ""
        trade_executed = False

        if cash > last_price and quantity > 0:
            if sentiment == "positive" and probability > 0.999:
                # Close short position if necessary
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
                # Close long position if necessary
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
            rerun_app()  # Try to rerun the app automatically.
        else:
            st.error("Invalid username or password.")
    st.stop()

# ----------------------------
# Main Navigation & Pages
# ----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Trading", "Backtesting", "Settings"])

# Logout button
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    rerun_app()

if page == "Dashboard":
    st.title("Dashboard")
    st.write("Welcome to your Trading Bot Dashboard.")
    
    st.subheader("Market Sentiment Analysis")
    trader = MLTrader()  # instantiate the trading bot
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
    st.write("Run a backtesting simulation on historical market data.")
    start_date = st.date_input("Start Date", datetime(2020, 1, 1))
    end_date = st.date_input("End Date", datetime(2023, 12, 31))
    if st.button("Run Backtest"):
        with st.spinner("Running backtest simulation..."):
            # In a full integration you could call lumibot's backtesting here.
            # For demonstration, we simulate a backtest delay.
            time.sleep(3)
        st.success("Backtest complete. (Results would be displayed here.)")
        st.write("**Note:** In production, integrate lumibot's YahooDataBacktesting for detailed analytics.")

elif page == "Settings":
    st.title("Settings")
    st.write("Configure your trading bot parameters below.")
    symbol = st.text_input("Trading Symbol", value="SPY")
    cash_at_risk = st.slider("Cash at Risk (fraction)", 0.0, 1.0, 0.5)
    st.write("These settings can be used to reinitialize your trading bot instance.")
    if st.button("Update Settings"):
        st.success("Settings updated!")
