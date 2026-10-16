"""
Hi Im varun and this script is for time series forcasting using ARIMA, Prophet, and LSTM models.
It fetches stock data from yfinance, trains the models, and evaluates their performance.
It also plots the results for better visualization.
This is optimized to be train on cpu so if you system is taking time to train, you can try running it on a GPU.
or reduce the epochs and lookback to speed up the training.
"""


import os
import argparse
import warnings
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf  # free data; no API key required
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")
plt.style.use("ggplot")

@dataclass
class Config:
    symbol: str = "AAPL"
    start: str = "2018-01-01"
    end: str = "2024-07-31"
    forecast_horizon: int = 30
    lookback: int = 60  
    batch_size: int = 32
    epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()


def fetch_data_yfinance(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetches OHLCV data from yfinance and returns a DataFrame with Date index and 'Close'.
    yfinance is free and doesn't require API keys.
    """
    df = yf.download(symbol, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError("No data returned by yfinance. Check symbol and dates.")
    df = df[["Close"]].rename(columns={"Close": "value"})
    df.index = pd.to_datetime(df.index)
    df = df.asfreq("B")  
    df = df.fillna(method="ffill")
    return df


def train_test_split_ts(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    split = int(n * (1 - test_size))
    return df.iloc[:split].copy(), df.iloc[split:].copy()

def fit_arima(series: pd.Series, order=(5, 1, 0)):
    model = ARIMA(series, order=order)
    res = model.fit()
    return res

def fit_prophet(df: pd.DataFrame):
    prophet_df = df.reset_index()
    prophet_df.columns = ["ds", "y"]  
    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(prophet_df)
    return m


class TimeSeriesDataset(Dataset):
    def __init__(self, series: np.ndarray, lookback: int):
        self.series = series
        self.lookback = lookback

    def __len__(self):
        return max(0, len(self.series) - self.lookback)

    def __getitem__(self, idx):
        x = self.series[idx: idx + self.lookback]
        y = self.series[idx + self.lookback]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(-1)


from tqdm import tqdm

def train_lstm(residuals: np.ndarray, cfg: Config, scaler=None):
    arr = residuals.reshape(-1, 1).astype(np.float32)
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        arr = scaler.fit_transform(arr)
    else:
        arr = scaler.transform(arr)

    seq = arr.flatten()
    dataset = TimeSeriesDataset(seq, cfg.lookback)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = LSTMModel(input_size=1, hidden_size=64, num_layers=2).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    total_batches = len(dataloader) * cfg.epochs
    completed_batches = 0

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0

        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{cfg.epochs}", unit="batch") as pbar:
            for xb, yb in dataloader:
                xb = xb.unsqueeze(-1).to(cfg.device)
                yb = yb.to(cfg.device)
                preds = model(xb)
                loss = loss_fn(preds, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * xb.size(0)

                completed_batches += 1
                percent_done = (completed_batches / total_batches) * 100
                pbar.set_postfix({"Loss": f"{loss.item():.6f}", "Done": f"{percent_done:.1f}%"})
                pbar.update(1)

        avg_loss = total_loss / len(dataset)
        print(f"[LSTM] Epoch {epoch+1}/{cfg.epochs} completed - Avg Loss: {avg_loss:.6f}")

    return model, scaler



def predict_lstm_future(model: nn.Module, last_residuals: np.ndarray, steps: int, cfg: Config, scaler):

    model.eval()
    preds = []
    window = last_residuals.reshape(-1, 1).astype(np.float32)
    window = scaler.transform(window).flatten().tolist() 

    for _ in range(steps):
        x = torch.tensor(window[-cfg.lookback:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(cfg.device)
        with torch.no_grad():
            p = model(x).cpu().numpy().flatten()[0] 
        preds.append(p)
        window.append(p) 

    preds = np.array(preds).reshape(-1, 1)
    preds_unscaled = scaler.inverse_transform(preds).flatten()
    return preds_unscaled

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_predictions(y_true, y_pred, label="model"):
    mae = mean_absolute_error(y_true, y_pred)
    _rmse = rmse(y_true, y_pred)
    print(f"{label} - MAE: {mae:.4f}, RMSE: {_rmse:.4f}")
    return {"mae": mae, "rmse": _rmse}

def pipeline(symbol: str, start: str, end: str, forecast_horizon: int, cfg: Config):
    print(f"Fetching data for {symbol} from {start} to {end} (yfinance, free)...")
    df = fetch_data_yfinance(symbol, start, end)
    print(f"Data length: {len(df)} rows. Date range: {df.index.min().date()} to {df.index.max().date()}")

    train, test = train_test_split_ts(df, test_size=0.2)
    print(f"Train: {len(train)} rows, Test: {len(test)} rows")

    print("Fitting ARIMA (order=(5,1,0)) on train...")
    arima_res = fit_arima(train["value"], order=(5, 1, 0))
    arima_insample = arima_res.predict(start=train.index[0], end=train.index[-1])
    arima_forecast_test = arima_res.predict(start=test.index[0], end=test.index[-1])
    arima_forecast_future = arima_res.forecast(steps=forecast_horizon)

    evaluate_predictions(test["value"].values, arima_forecast_test.values, label="ARIMA")

    print("Computing residuals (actual - arima_in_sample)...")
    arima_full_res = arima_res.predict(start=df.index[0], end=df.index[-1])
    residuals = df["value"].values - arima_full_res.values

    print("Fitting Prophet on train (this can be slower)...")
    prophet_model = fit_prophet(train)
    future_df = prophet_model.make_future_dataframe(periods=len(test) + forecast_horizon, freq="B")
    prophet_preds = prophet_model.predict(future_df)
    prophet_pred_series = pd.Series(prophet_preds["yhat"].values, index=future_df["ds"].values).astype(float)
    prophet_test_pred = prophet_pred_series.loc[test.index].values
    prophet_future_pred = prophet_pred_series.loc[future_df["ds"].values[-forecast_horizon:]].values

    evaluate_predictions(test["value"].values, prophet_test_pred, label="Prophet")

    print("Training LSTM on ARIMA residuals (hybrid)...")
    cfg_local = cfg
    lstm_model, scaler = train_lstm(residuals, cfg_local)

    seed_idx = np.where(df.index == test.index[0])[0][0]  
    last_window = residuals[seed_idx - cfg_local.lookback: seed_idx] 
    lstm_test_residuals_pred = predict_lstm_future(lstm_model, last_window, len(test), cfg_local, scaler)

    hybrid_test_pred = arima_forecast_test.values + lstm_test_residuals_pred

    evaluate_predictions(test["value"].values, hybrid_test_pred, label="Hybrid (ARIMA+LSTM residuals)")

    last_window_full = residuals[-cfg_local.lookback:]
    lstm_future_residuals = predict_lstm_future(lstm_model, last_window_full, forecast_horizon, cfg_local, scaler)
    hybrid_future_forecast = arima_forecast_future.values + lstm_future_residuals

    results_df = pd.DataFrame(index=df.index)
    results_df["actual"] = df["value"]
    results_df["arima_in_sample"] = arima_full_res.values
    future_index = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon)
    arima_future_series = pd.Series(arima_forecast_future.values, index=future_index)
    hybrid_future_series = pd.Series(hybrid_future_forecast, index=future_index)
    prophet_future_series = pd.Series(prophet_future_pred, index=future_index)
    plot_horizon = 200
    plt.figure(figsize=(14, 6))
    to_plot_idx = df.index[-plot_horizon:]
    plt.plot(df.index[-plot_horizon:], df["value"].values[-plot_horizon:], label="actual", linewidth=2)
    plt.plot(to_plot_idx, arima_full_res.values[-plot_horizon:], label="ARIMA in-sample")
    plt.plot(test.index, arima_forecast_test.values, label="ARIMA test forecast", alpha=0.8)
    plt.plot(test.index, hybrid_test_pred, label="Hybrid test forecast (ARIMA + LSTM residuals)", alpha=0.9)
    # future hendrix
    plt.plot(future_index, arima_future_series, label="ARIMA future", linestyle="--")
    plt.plot(future_index, hybrid_future_series, label="Hybrid future", linestyle="--")
    plt.plot(future_index, prophet_future_series, label="Prophet future", linestyle=":")
    plt.legend()
    plt.title(f"{symbol} - Actual vs ARIMA / Prophet / Hybrid (last {plot_horizon} days + {forecast_horizon} day forecast)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.show()

    print("\n--- Summary ---")
    arima_eval = evaluate_predictions(test["value"].values, arima_forecast_test.values, "ARIMA")
    prophet_eval = evaluate_predictions(test["value"].values, prophet_test_pred, "Prophet")
    hybrid_eval = evaluate_predictions(test["value"].values, hybrid_test_pred, "Hybrid")
    return {
        "arima": arima_eval,
        "prophet": prophet_eval,
        "hybrid": hybrid_eval,
        "future_index": future_index,
        "hybrid_future_series": hybrid_future_series,
    }
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=cfg.symbol, help="Ticker symbol (yfinance)")
    parser.add_argument("--start", type=str, default=cfg.start)
    parser.add_argument("--end", type=str, default=cfg.end)
    parser.add_argument("--forecast_horizon", type=int, default=cfg.forecast_horizon)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--lookback", type=int, default=cfg.lookback)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg.symbol = args.symbol
    cfg.start = args.start
    cfg.end = args.end
    cfg.forecast_horizon = args.forecast_horizon
    cfg.epochs = args.epochs
    cfg.lookback = args.lookback

    pipeline(cfg.symbol, cfg.start, cfg.end, cfg.forecast_horizon, cfg)
