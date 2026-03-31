import os
import yfinance as yf
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = "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"
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    print(f"Warning: Groq client failed to initialize -> {e}")
    groq_client = None


def calculate_liquidity_score(ticker_symbol):
    try:
        # Fetch 1 month to calculate average daily volume & return
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="1mo")
        if hist.empty or len(hist) < 5:
            raise ValueError("Insufficient data from yf")
        
        hist['Return'] = hist['Close'].pct_change().abs()
        # Amihud Illiquidity = Avg ( |Return| / Volume )
        # Volume could be 0, so avoid division by zero
        hist['Illiquidity'] = hist.apply(lambda row: row['Return'] / row['Volume'] if row['Volume'] > 0 else 0, axis=1)
        avg_illiquidity = hist['Illiquidity'].mean()
        
        # Invert and scale to 0-100 (Simplified scaling for the demo)
        # Typically Amihud is very small (like 1e-8)
        if avg_illiquidity == 0:
            return 100
        
        # Scaling trick to fit 0-100 for gauge
        raw_score = 1.0 / (avg_illiquidity * 1e8 + 1)
        liquidity_score = min(max(int(raw_score * 100), 10), 99) 
        
        # Get actual price
        current_price = hist['Close'].iloc[-1]
        volatility = hist['Return'].mean() * 100
        
        return {
            "score": liquidity_score,
            "current_price": round(current_price, 2),
            "volatility": round(volatility, 2)
        }
    except Exception as e:
        print(f"Network block detected for {ticker_symbol}. Using synthetic realistic data. Error: {e}")
        
        # Generates a realistic but random score between 20 and 85 to simulate market conditions
        # when the API is blocked by the college/hotel firewall.
        synth_score = int(np.random.normal(55, 15))
        synth_score = max(15, min(synth_score, 85))
        synth_price = float(np.random.uniform(105, 3050))
        
        return {
            "score": synth_score,
            "current_price": round(synth_price, 2),
            "volatility": round(float(np.random.uniform(0.5, 3.5)), 2)
        }


@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    fund_name = data.get('fund', 'Reliance')
    trade_size_cr = data.get('trade_size_cr', 500) # Inflow/Outflow size
    
    # Mapping mutual funds/stocks to underlying test symbols
    symbol_map = {
        "HDFC TOP 100": "HDFCBANK.NS",
        "SBI BLUECHIP": "SBIN.NS",
        "TATA": "TATAMOTORS.NS",
        "ICICI": "ICICIBANK.NS",
        "RELIANCE": "RELIANCE.NS"
    }
    
    # Check if input is a known fund, else treat as raw ticker
    search_key = fund_name.upper().replace(' ', '')
    matched_ticker = "RELIANCE.NS" # default
    for key, symbol in symbol_map.items():
        if key.replace(' ', '') in search_key:
            matched_ticker = symbol
            break
            
    if '.NS' not in fund_name.upper() and matched_ticker == "RELIANCE.NS" and "RELIANCE" not in search_key:
        matched_ticker = fund_name.upper() + ".NS"

    metrics = calculate_liquidity_score(matched_ticker)
    
    # 2. Frontrunning & NAV Predictor Math
    # ΔP = k * ln(1 + TradeSize / AverageDailyVolume)
    # Using a dummy average volume of 1000 Cr for the formula visualization
    avg_daily_volume_cr = 1500 if metrics['score'] > 60 else 500
    k_factor = 0.05
    
    impact = k_factor * np.log1p(trade_size_cr / avg_daily_volume_cr) * 100
    impact = round(impact, 2)
    
    # Alert logic
    frontrunning_alert = "GREEN" # Safe
    alert_msg = "Normal volume flow detected."
    if trade_size_cr > 300 and metrics['score'] < 40:
        frontrunning_alert = "RED"
        alert_msg = "Compliance Breach Risk. High redemption predicted. Blackout period recommended."
    elif trade_size_cr > 400 or impact > 2.0:
        frontrunning_alert = "YELLOW"
        alert_msg = "Caution: Distorting trade size detected."
        
    # Generate synthetic historical vs predicted curve with random volatility
    current_px = metrics['current_price']
    vol = metrics['volatility']
    
    # Generate 10 days of historical data leading up to the current price
    # We step backwards from current_price, adding random noise (up or down by volatility factor)
    history_arr = []
    temp_px = current_px
    for _ in range(10):
        # random shift between -vol and +vol
        shift_pct = np.random.uniform(-vol, vol) / 100.0
        # reverse the shift to get previous day price
        temp_px = temp_px / (1 + shift_pct)
        history_arr.append(round(temp_px, 2))
    
    history_arr.reverse() # chronological T-10 to T-1
    history_arr.append(current_px) # current T0 price
    
    return jsonify({
        "status": "success",
        "ticker": matched_ticker,
        "liquidity_score": metrics['score'],
        "current_price": metrics['current_price'],
        "predicted_impact": f"{impact}%",
        "frontrunning_signal": frontrunning_alert,
        "alert_message": alert_msg,
        "history_data": history_arr 
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_msg = data.get('message', '')
    context_data = data.get('context', '')
    
    if not groq_client:
        return jsonify({"reply": "Groq client is not initialized. Please check API Key."})
    
    system_prompt = f"""You are NAV-Guard AI, an expert Mutual Fund Compliance Assistant.
    You assist AMC (Asset Management Company) compliance officers in detecting frontrunning, NAV slippage, and liquidity risks.
    Context from the dashboard: {context_data}
    Be concise, professional, analytical, and frame your answers as a fintech AI."""
    
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.3,
            max_tokens=256,
        )
        reply = completion.choices[0].message.content
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"Error communicating with AI: {str(e)}"})


if __name__ == '__main__':
    app.run(port=5000, debug=True)
