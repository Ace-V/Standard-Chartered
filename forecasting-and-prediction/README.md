# Forecasting-and-Prediction
# Time Series Forecasting with ARIMA, Prophet, and LSTM

A comprehensive Python script for time series forecasting that combines three powerful approaches: ARIMA (statistical), Prophet (Facebook's forecasting tool), and LSTM (deep learning) models. The script fetches stock data, trains multiple models, and provides comparative performance analysis with visualization.

## Features

- **Multi-Model Approach**: Implements ARIMA, Prophet, and a Hybrid ARIMA+LSTM model
- **Free Data Source**: Uses yfinance for free stock data (no API key required)
- **Hybrid Modeling**: Combines ARIMA with LSTM to model residuals for improved accuracy
- **Performance Evaluation**: Calculates MAE and RMSE metrics for model comparison
- **Visualization**: Generates comprehensive plots showing actual vs predicted values
- **Flexible Configuration**: Easy-to-modify parameters for different use cases
- **CPU Optimized**: Designed to run efficiently on CPU (with optional GPU support)

## Models Implemented

1. **ARIMA (5,1,0)**: Traditional statistical time series model
2. **Prophet**: Facebook's robust forecasting model with seasonality handling
3. **Hybrid ARIMA+LSTM**: Uses ARIMA for trend and LSTM neural network to model residuals

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: Prophet installation might take some time as it compiles Stan models. If you encounter issues with Prophet installation:

```bash
# For conda users (alternative installation)
conda install -c conda-forge prophet

# For pip users having issues
pip install prophet --no-cache-dir
```

## Usage

### Basic Usage

Run with default settings (AAPL stock, 2018-2024 data):

```bash
python "TIME-Series Forecasting.py"
```

### Custom Parameters

```bash
python "TIME-Series Forecasting.py" --symbol MSFT --start 2020-01-01 --end 2024-01-01 --forecast_horizon 60 --epochs 20
```

### Available Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--symbol` | Stock ticker symbol | AAPL |
| `--start` | Start date (YYYY-MM-DD) | 2018-01-01 |
| `--end` | End date (YYYY-MM-DD) | 2024-07-31 |
| `--forecast_horizon` | Number of days to forecast | 30 |
| `--epochs` | LSTM training epochs | 10 |
| `--lookback` | LSTM lookback window | 60 |

## Configuration

The script uses a `Config` dataclass that can be modified for different use cases:

```python
@dataclass
class Config:
    symbol: str = "AAPL"           # Stock symbol
    start: str = "2018-01-01"      # Start date
    end: str = "2024-07-31"        # End date
    forecast_horizon: int = 30      # Forecast days
    lookback: int = 60             # LSTM lookback window
    batch_size: int = 32           # LSTM batch size
    epochs: int = 10               # LSTM epochs
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
```

## Output

The script provides:

1. **Performance Metrics**: MAE and RMSE for each model on test data
2. **Visualization**: Plot showing:
   - Historical actual prices
   - In-sample ARIMA predictions
   - Test forecasts for all models
   - Future forecasts (30 days by default)
3. **Model Comparison**: Summary of all model performances

### Sample Output

```
AAPL - MAE: 2.4536, RMSE: 3.1242
Prophet - MAE: 3.1847, RMSE: 4.2156
Hybrid (ARIMA+LSTM residuals) - MAE: 2.1834, RMSE: 2.8932
```

## Model Details

### ARIMA Model
- Uses order (5,1,0) - can be modified in the `fit_arima` function
- Good for capturing linear trends and autocorrelations
- Fast training and prediction

### Prophet Model
- Handles seasonality (weekly, yearly) automatically
- Robust to missing data and outliers
- Slower training but very reliable

### Hybrid ARIMA+LSTM
- ARIMA captures the main trend
- LSTM neural network models the residuals (errors)
- Often provides the best accuracy by combining both approaches

## Performance Optimization

### For Faster Training:
- Reduce `epochs` (e.g., 5-10 for quick testing)
- Reduce `lookback` window (e.g., 30 instead of 60)
- Use smaller `batch_size`

### For Better Accuracy:
- Increase `epochs` (20-50)
- Increase `lookback` window (90-120)
- Use more historical data (longer date range)

### GPU Acceleration:
The script automatically detects and uses GPU if available. For GPU usage:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Dependencies

- **numpy**: Numerical computations
- **pandas**: Data manipulation
- **matplotlib**: Plotting and visualization
- **yfinance**: Free stock data fetching
- **statsmodels**: ARIMA implementation
- **prophet**: Facebook's forecasting library
- **scikit-learn**: ML utilities and metrics
- **torch**: PyTorch for LSTM implementation
- **tqdm**: Progress bars during training

## Troubleshooting

### Prophet Installation Issues
```bash
# Try these alternatives if Prophet installation fails:
pip install pystan==2.19.1.1
pip install prophet==1.1

# Or use conda:
conda install -c conda-forge prophet
```

### Memory Issues
- Reduce batch_size (try 16 or 8)
- Reduce lookback window
- Use shorter date ranges

### Poor Performance
- Increase epochs for LSTM training
- Try different ARIMA orders
- Use more historical data
- Check for data quality issues

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## Author

Created by Varun - Optimized for CPU training with GPU support available.
