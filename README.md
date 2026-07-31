# Stock Price Prediction in Python with PyTorch using an LSTM-based Architecture

A time-series forecasting project that predicts the next-day closing price of the
**Nifty 50 index (^NSEI)** using an LSTM (Long Short-Term Memory) neural network
built in PyTorch.

## Tech Stack

- **PyTorch** — LSTM model definition and training
- **yfinance** — historical price data download
- **scikit-learn** — feature scaling (StandardScaler) and RMSE evaluation
- **pandas / numpy** — data wrangling and sequence construction
- **matplotlib** — visualization

## How It Works

1. **Data collection** — daily closing prices for `^NSEI` from Yahoo Finance, starting 2020-01-01
2. **Scaling** — prices standardized with `StandardScaler` so the LSTM trains on a well-behaved range
3. **Sequencing** — scaled series converted into overlapping 30-day windows (29 days input, 30th day target)
4. **Train/test split** — chronological 80/20 split
5. **Model** — 2-layer LSTM (hidden_dim=32) + linear output layer
6. **Training** — 200 epochs, Adam optimizer, MSE loss
7. **Evaluation** — predictions inverse-transformed to real price units before computing RMSE

## Results

| Metric | Value |
|---|---|
| Train RMSE | 216.36 |
| Test RMSE | 262.97 |

## Getting Started

### Installation
```bash
git clone https://github.com/vanshb04/stock-price-prediction-pytorch-lstm.git
cd stock-price-prediction-pytorch-lstm
pip install -r requirements.txt
```

### Run
Open `StockPricePrediction.ipynb` in Jupyter and run all cells.

## License
MIT License — see [LICENSE](LICENSE)

## Disclaimer
Educational project only. Not financial advice.
