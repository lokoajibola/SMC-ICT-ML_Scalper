# SMC/ICT Forex Trading Bot

An algorithmic trading bot for MetaTrader 5 that implements **Smart Money Concepts (SMC)**, **Inner Circle Trader (ICT)** methodologies, and liquidity-based strategies. The bot analyzes multiple timeframes (M5, M15, H1, H4, D1) to identify high-probability trade setups, places pending limit orders at key order blocks, and manages positions with dynamic trailing stops.

## Features

- **Multi-Timeframe Analysis** – M5, M15, H1, H4, D1 for confluence.
- **SMC Concepts** – Order blocks, breaker blocks, liquidity grabs.
- **ICT Concepts** – Kill zones (London & NY), fair value gaps (FVG), optimal trade entry (OTE).
- **Dynamic Pending Orders** – Places only the closest buy limit and sell limit orders per symbol.
- **Intelligent Order Management** – Cancels/updates orders only when market structure changes significantly.
- **Trailing Stop** – Moves stop loss to breakeven at 20% profit, then trails with 5% step.
- **Trade Reasoning Logging** – Records detailed reasons for each trade (e.g., FVG, liquidity, SNR levels) in a text file.
- **Kill Zone Filtering** – Only operates during high-probability trading sessions.
- **Multi-Asset Support** – Works with forex, metals, crypto, and Deriv volatility indices.

## Requirements

- **MetaTrader 5** terminal installed (Windows)
- Python 3.7+
- MT5 Python package (`MetaTrader5`)
- Other Python packages: `pandas`, `numpy`, `pytz`, `pandas-ta`

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/smc-ict-trading-bot.git
   cd smc-ict-trading-bot
   ```

2. **Install Python dependencies**
   ```bash
   pip install MetaTrader5 pandas numpy pytz pandas-ta
   ```

3. **Set up MetaTrader 5**
   - Install MT5 terminal from your broker.
   - Enable automated trading in MT5: **Tools → Options → Expert Advisors** → check "Allow automated trading".
   - Note the installation path (e.g., `C:/Program Files/MetaTrader 5 Terminal/terminal64.exe`).

4. **Configure your account**
   - Edit the script with your MT5 login credentials:
     ```python
     account = YOUR_ACCOUNT_NUMBER
     password = "YOUR_PASSWORD"
     server = "YOUR_BROKER_SERVER"
     mt5.initialize("PATH_TO_MT5_TERMINAL/terminal64.exe")
     ```

5. **Customize trading parameters**
   - Adjust `pairs` list to include symbols you want to trade.
   - Set `self.volume` in the strategy class (default 0.1).
   - Modify `magic` number to identify your bot's orders.

## How It Works

The bot continuously scans the market and applies SMC/ICT principles:

1. **Order Block Detection**
   - Identifies strong bullish/bearish candles after trend exhaustion.
   - Uses H4 timeframe for major order blocks.

2. **Fair Value Gaps (FVG)**
   - Looks for three-candle patterns where price leaves an imbalance.
   - FVGs act as magnets for price to return.

3. **Liquidity Zones**
   - Detects recent swing highs/lows where stop orders cluster.
   - Price often sweeps these levels before reversing.

4. **Kill Zones**
   - London: 6:00–11:00 (server time)
   - New York: 10:00–16:00 (server time)
   - Only places orders during these high-activity windows.

5. **Setup Filtering**
   - Selects only the closest buy limit (below price) and sell limit (above price) setups.
   - Requires minimum 1:2 risk-reward ratio.

6. **Trade Execution**
   - Places pending limit orders at order block levels.
   - When triggered, the bot manages the position with trailing stop.

7. **Trailing Stop Logic**
   - Moves SL to breakeven when profit reaches 20% of target.
   - Thereafter, trails SL with 5% step to lock in profits.

8. **Trade Logging**
   - Each trade reason is saved in `trading_reasons.txt` with timestamp, symbol, direction, and detailed analysis (e.g., "FVG@1.0850-1.0865 on H1", "Liquidity@1.0800 on M15").

## Usage

Run the bot from your terminal:
```bash
python smc_ict_bot.py
```

The bot will start scanning and display:
- Full analysis every 10 minutes.
- Order placement scans every minute.
- Active pending orders.
- Trailing stop updates.

Press `Ctrl+C` to stop the bot gracefully; it will cancel all pending orders.

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `pairs` | List of symbols to trade | Forex majors + Volatility indices |
| `magic` | Unique identifier for bot orders | 234000 |
| `volume` | Lot size per trade | 0.1 |
| `min_order_distance` | Minimum distance from open trades to avoid duplicates | 0.00010 |
| `trailing_activation` | Profit % to activate breakeven | 0.2 (20%) |
| `trailing_step` | Step % for trailing stop | 0.05 (5%) |

## Trade Logging Example

When a trade is placed, the reason is saved in `trading_reasons.txt`:
```
[2025-03-15 14:30:22] EURUSD BUY at 1.08250
  • BullishOB@1.08120 on H4
  • FVG@1.08300-1.08450 on H1
  • Support@1.08000 on M15
  • Liquidity below 1.07950 taken
  • London Kill Zone active
```

## Risk Disclaimer

**Trading forex and CFDs carries a high level of risk and may not be suitable for all investors.** This bot is for educational and research purposes only. Past performance does not guarantee future results. Use at your own risk.

## Future Improvements

- Machine learning integration for setup scoring (XGBoost/LSTM).
- Web dashboard for real-time monitoring.
- Telegram alerts for trade executions.
- Backtesting engine using historical data.
- Support for additional brokers via API.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)

---

**Happy Trading!** 🚀
