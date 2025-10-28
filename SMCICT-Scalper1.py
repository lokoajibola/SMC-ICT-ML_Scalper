# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 22:01:00 2025

SMC/ICT Trading Bot with ML & Non-ML Modes + Scalping 
@author: jbriz
""" 

import MetaTrader5 as mt5
import time
import datetime
import pandas as pd
import pytz
import pandas_ta as ta
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

# INPUTS
pairs = ['XAUUSD', 'EURJPY', 'EURGBP', 'EURCAD', 'GBPUSD', 'EURUSD', 'USDJPY', 
         'AUDUSD', 'USDCAD', 'BTCUSD', 'Volatility 150 (1s) Index', 'Boom 500 Index', 'Crash 500 Index']
tz = pytz.timezone('Europe/Nicosia') 

# TRADING MODES
USE_ML = True  # Set to False for pure SMC/ICT trading, True for ML-enhanced
ML_CONFIDENCE_THRESHOLD = 0.65  # Minimum ML confidence to execute trade
SCALPING_MODE = True
HTF_TIMEFRAME = 'H4'  # Analysis timeframe
LTF_TIMEFRAME = 'M5'  # Entry timeframe
MIN_RR_RATIO = 2.0    # Higher RR for scalping

# 1. LOGIN TO MT5
account = 5889667
mt5.initialize("C:/Program Files/MetaTrader 5 Terminal/terminal64.exe")
authorized = mt5.login(account, password=";lkPOI098", server="Deriv-Demo") 

if authorized:
    print("Connected: Connecting to MT5 Client")
else:
    print("Failed to connect at account #{}, error code: {}".format(account, mt5.last_error())) 

if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
    mt5.shutdown()

class MLFeatureEngineer:
    """Machine Learning Feature Engineering for SMC/ICT Trading"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = []
        self.model_trained = False
        self.entry_triggers = {}  # Track HTF setups waiting for LTF entry
        
    def calculate_technical_indicators(self, df):
        """Calculate comprehensive technical indicators"""
        if not isinstance(df, pd.DataFrame):
            return df
            
        df = df.copy()
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df['momentum'] = df['close'] - df['close'].shift(5)
        
        # RSI
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['rsi_sma'] = df['rsi'].rolling(5).mean()
        
        # MACD
        macd = ta.macd(df['close'])
        if macd is not None:
            df['macd'] = macd['MACD_12_26_9']
            df['macd_signal'] = macd['MACDs_12_26_9']
            df['macd_hist'] = macd['MACDh_12_26_9']
        
        # Bollinger Bands
        bb = ta.bbands(df['close'], length=20)
        if bb is not None:
            df['bb_upper'] = bb['BBU_20_2.0']
            df['bb_lower'] = bb['BBL_20_2.0']
            df['bb_middle'] = bb['BBM_20_2.0']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR for volatility
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr_ratio'] = df['atr'] / df['close']
        
        # Volume features (if available)
        if 'tick_volume' in df.columns:
            df['volume_sma'] = df['tick_volume'].rolling(20).mean()
            df['volume_ratio'] = df['tick_volume'] / df['volume_sma']
        
        # Price position features
        df['high_20'] = df['high'].rolling(20).max()
        df['low_20'] = df['low'].rolling(20).min()
        if len(df) > 0 and (df['high_20'] - df['low_20']).iloc[-1] != 0:
            df['price_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'])
        else:
            df['price_position'] = 0.5
        
        return df
    
    def find_swing_points(self, df, lookback=5):
        """Find swing highs and lows"""
        if not isinstance(df, pd.DataFrame) or len(df) < lookback * 2:
            if isinstance(df, pd.DataFrame):
                df = df.copy()
                df['is_swing_high'] = False
                df['is_swing_low'] = False
            return df
            
        df = df.copy()
        df['is_swing_high'] = False
        df['is_swing_low'] = False
        
        for i in range(lookback, len(df)-lookback):
            # Swing high
            if all(df['high'].iloc[i] > df['high'].iloc[i-lookback:i]) and \
               all(df['high'].iloc[i] > df['high'].iloc[i+1:i+lookback+1]):
                df.loc[df.index[i], 'is_swing_high'] = True
            
            # Swing low
            if all(df['low'].iloc[i] < df['low'].iloc[i-lookback:i]) and \
               all(df['low'].iloc[i] < df['low'].iloc[i+1:i+lookback+1]):
                df.loc[df.index[i], 'is_swing_low'] = True
        
        return df
    
    def detect_fvg(self, df):
        """Detect Fair Value Gaps"""
        if not isinstance(df, pd.DataFrame):
            return []
            
        fvgs = []
        for i in range(2, len(df)):
            # Bullish FVG
            if df['low'].iloc[i] > df['high'].iloc[i-1]:
                fvgs.append({
                    'type': 'BULLISH_FVG',
                    'high': df['low'].iloc[i],
                    'low': df['high'].iloc[i-1],
                    'index': i
                })
            # Bearish FVG
            if df['high'].iloc[i] < df['low'].iloc[i-1]:
                fvgs.append({
                    'type': 'BEARISH_FVG',
                    'high': df['low'].iloc[i-1],
                    'low': df['high'].iloc[i],
                    'index': i
                })
        return fvgs
    
    def detect_liquidity_zones(self, df, threshold=0.0002):
        """Detect liquidity zones"""
        if not isinstance(df, pd.DataFrame):
            return []
            
        zones = []
        for i in range(1, len(df)):
            # Equal highs
            if abs(df['high'].iloc[i] - df['high'].iloc[i-1]) / df['high'].iloc[i] < threshold:
                zones.append({
                    'type': 'EQUAL_HIGHS',
                    'price': df['high'].iloc[i],
                    'index': i
                })
            # Equal lows
            if abs(df['low'].iloc[i] - df['low'].iloc[i-1]) / df['low'].iloc[i] < threshold:
                zones.append({
                    'type': 'EQUAL_LOWS',
                    'price': df['low'].iloc[i],
                    'index': i
                })
        return zones
    
    def detect_order_blocks(self, df):
        """Detect order blocks"""
        if not isinstance(df, pd.DataFrame):
            return df
            
        df = df.copy()
        df['bullish_ob'] = False
        df['bearish_ob'] = False
        
        for i in range(1, len(df)):
            # Bullish order block: strong bearish candle followed by strong bullish candle
            if (df['close'].iloc[i-1] < df['open'].iloc[i-1] and  # Bearish candle
                df['close'].iloc[i] > df['open'].iloc[i] and      # Bullish candle
                (df['open'].iloc[i-1] - df['close'].iloc[i-1]) > (df['high'].iloc[i-1] - df['low'].iloc[i-1]) * 0.6):
                df.loc[df.index[i], 'bullish_ob'] = True
            
            # Bearish order block: strong bullish candle followed by strong bearish candle
            if (df['close'].iloc[i-1] > df['open'].iloc[i-1] and  # Bullish candle
                df['close'].iloc[i] < df['open'].iloc[i] and      # Bearish candle
                (df['close'].iloc[i-1] - df['open'].iloc[i-1]) > (df['high'].iloc[i-1] - df['low'].iloc[i-1]) * 0.6):
                df.loc[df.index[i], 'bearish_ob'] = True
        
        return df
    
    def calculate_smc_ict_features(self, df):
        """Calculate SMC/ICT specific features"""
        if not isinstance(df, pd.DataFrame):
            return df
            
        df = df.copy()
        
        # Swing point detection
        df = self.find_swing_points(df)
        
        # FVG detection
        fvgs = self.detect_fvg(df)
        df['bullish_fvg'] = 0
        df['bearish_fvg'] = 0
        
        for fvg in fvgs:
            idx = fvg['index']
            if idx < len(df):
                if fvg['type'] == 'BULLISH_FVG':
                    df.loc[df.index[idx], 'bullish_fvg'] = 1
                else:
                    df.loc[df.index[idx], 'bearish_fvg'] = 1
        
        # Order block detection
        df = self.detect_order_blocks(df)
        
        # Market structure features
        df['higher_high'] = df['high'] > df['high'].shift(1)
        df['higher_low'] = df['low'] > df['low'].shift(1)
        df['lower_high'] = df['high'] < df['high'].shift(1)
        df['lower_low'] = df['low'] < df['low'].shift(1)
        
        return df
    
    def create_ml_features(self, symbol_data):
        """Create ML features from multiple timeframes"""
        features = {}
        
        for tf_name, df in symbol_data.items():
            if tf_name == 'symbol':  # Skip the symbol entry
                continue
                
            if not isinstance(df, pd.DataFrame):
                continue
                
            # Calculate technical indicators
            df_tech = self.calculate_technical_indicators(df)
            df_full = self.calculate_smc_ict_features(df_tech)
            
            # Recent features (last 5 candles)
            for col in ['rsi', 'macd', 'bb_position', 'atr_ratio', 'price_position']:
                if col in df_full.columns and len(df_full) > 0:
                    features[f'{tf_name}_{col}'] = df_full[col].iloc[-1] if not pd.isna(df_full[col].iloc[-1]) else 0
                    features[f'{tf_name}_{col}_mean5'] = df_full[col].tail(5).mean() if len(df_full) >= 5 else 0
                    features[f'{tf_name}_{col}_std5'] = df_full[col].tail(5).std() if len(df_full) >= 5 else 0
            
            # SMC/ICT features
            features[f'{tf_name}_bullish_fvg'] = df_full['bullish_fvg'].tail(5).sum() if len(df_full) >= 5 else 0
            features[f'{tf_name}_bearish_fvg'] = df_full['bearish_fvg'].tail(5).sum() if len(df_full) >= 5 else 0
            features[f'{tf_name}_bullish_ob'] = df_full['bullish_ob'].tail(10).sum() if len(df_full) >= 10 else 0
            features[f'{tf_name}_bearish_ob'] = df_full['bearish_ob'].tail(10).sum() if len(df_full) >= 10 else 0
            features[f'{tf_name}_swing_highs'] = df_full['is_swing_high'].tail(20).sum() if len(df_full) >= 20 else 0
            features[f'{tf_name}_swing_lows'] = df_full['is_swing_low'].tail(20).sum() if len(df_full) >= 20 else 0
        
        # Multi-timeframe alignment features
        if 'H1' in symbol_data and 'H4' in symbol_data:
            h1_df = symbol_data['H1']
            h4_df = symbol_data['H4']
            if isinstance(h1_df, pd.DataFrame) and isinstance(h4_df, pd.DataFrame):
                h1_rsi = h1_df['rsi'].iloc[-1] if 'rsi' in h1_df.columns and len(h1_df) > 0 else 50
                h4_rsi = h4_df['rsi'].iloc[-1] if 'rsi' in h4_df.columns and len(h4_df) > 0 else 50
                features['rsi_alignment'] = 1 if (h1_rsi > 50 and h4_rsi > 50) or (h1_rsi < 50 and h4_rsi < 50) else 0
        
        # Ensure we have at least some features
        if not features:
            # Create default features if none were generated
            for tf in ['M15', 'H1', 'H4']:
                features[f'{tf}_rsi'] = 50
                features[f'{tf}_rsi_mean5'] = 50
                features[f'{tf}_bullish_fvg'] = 0
                features[f'{tf}_bearish_fvg'] = 0
        
        return features
    
    def prepare_training_data(self, historical_data):
        """Prepare training data from historical trades"""
        X = []
        y = []
        
        for trade in historical_data:
            features = self.create_ml_features(trade['market_data'])
            if features:
                X.append(list(features.values()))
                y.append(1 if trade['profit'] > 0 else 0)
        
        if X:
            self.feature_names = list(features.keys())
        return np.array(X), np.array(y)
    
    def train_model(self, historical_data):
        """Train XGBoost model on historical data"""
        print("🤖 Training XGBoost Model...")
        
        X, y = self.prepare_training_data(historical_data)
        
        if len(X) < 50:
            print(f"⚠️ Insufficient training data. Need at least 50 samples, got {len(X)}")
            return False
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model trained with accuracy: {accuracy:.3f}")
        print(classification_report(y_test, y_pred))
        
        # Show feature importance
        feature_importance = self.model.feature_importances_
        top_features = sorted(zip(self.feature_names, feature_importance), 
                            key=lambda x: x[1], reverse=True)[:10]
        print("🔝 Top 10 Features:")
        for feat, imp in top_features:
            print(f"   {feat}: {imp:.4f}")
        
        self.model_trained = True
        return True
    
    def predict_setup_quality(self, symbol_data):
        """Predict quality of trading setup using ML model"""
        if not self.model_trained:
            return 0.5, "Model not trained - using default confidence"
        
        try:
            features = self.create_ml_features(symbol_data)
            if not features:
                return 0.5, "No features generated"
                
            X = np.array([list(features.values())])
            X_scaled = self.scaler.transform(X)
            
            probability = self.model.predict_proba(X_scaled)[0][1]
            
            # Get feature importance for reasoning
            feature_importance = self.model.feature_importances_
            top_features_idx = np.argsort(feature_importance)[-3:][::-1]  # Top 3 features
            top_features = [self.feature_names[i] for i in top_features_idx if i < len(self.feature_names)]
            top_scores = [feature_importance[i] for i in top_features_idx if i < len(self.feature_names)]
            
            reason = "Top features: " + ", ".join([f"{feat}({score:.3f})" 
                                                 for feat, score in zip(top_features, top_scores)])
            
            return probability, reason
            
        except Exception as e:
            print(f"❌ ML prediction error: {e}")
            return 0.5, f"Prediction error: {e}"
    
    def save_model(self, filename):
        """Save trained model and scaler"""
        if self.model_trained:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, filename)
            print(f"💾 Model saved to {filename}")
    
    def load_model(self, filename):
        """Load trained model and scaler"""
        try:
            loaded = joblib.load(filename)
            self.model = loaded['model']
            self.scaler = loaded['scaler']
            self.feature_names = loaded['feature_names']
            self.model_trained = True
            print(f"📂 Model loaded from {filename}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

class HybridSMCStrategy:
    def __init__(self):
        self.magic = 234000
        self.volume = 0.1
        self.trailing_active = {}
        self.pending_orders = {}
        self.last_order_levels = {}
        self.min_order_distance_percentage = 0.01  # 1% dynamic distance
        self.trade_reasons = {}
        self.ml_engine = MLFeatureEngineer()
        self.trade_log_file = "hybrid_trading_decisions.txt"
        
        # Initialize trade log file
        with open(self.trade_log_file, 'w', encoding='utf-8') as f:
            f.write("HYBRID SMC/ICT TRADING DECISIONS LOG\n")
            f.write("=" * 80 + "\n")
            f.write(f"ML Mode: {USE_ML}\n")
            f.write(f"ML Confidence Threshold: {ML_CONFIDENCE_THRESHOLD}\n")
            f.write("=" * 80 + "\n")
        
        # Try to load pre-trained model if ML mode is enabled
        if USE_ML:
            self.ml_engine.load_model("smc_ml_model.pkl")
        
    def get_multiple_timeframes(self, symbol):
        """Get data for multiple timeframes"""
        timeframes = {
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }
        
        data = {'symbol': symbol}
        for tf_name, tf in timeframes.items():
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, 200)
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df['hour'] = df['time'].dt.hour
                data[tf_name] = df
        return data
    
    def detect_fvg(self, df, timeframe):
        """Detect Fair Value Gaps across different timeframes"""
        fvgs = []
        if not isinstance(df, pd.DataFrame):
            return fvgs
            
        df = df.copy()
        
        # Calculate FVGs (three-bar pattern)
        for i in range(2, len(df)):
            # Bullish FVG: current low > previous high
            if df['low'].iloc[i] > df['high'].iloc[i-1]:
                fvg_high = df['low'].iloc[i]
                fvg_low = df['high'].iloc[i-1]
                fvgs.append({
                    'type': 'BULLISH_FVG',
                    'high': fvg_high,
                    'low': fvg_low,
                    'timeframe': timeframe
                })
            
            # Bearish FVG: current high < previous low
            if df['high'].iloc[i] < df['low'].iloc[i-1]:
                fvg_high = df['low'].iloc[i-1]
                fvg_low = df['high'].iloc[i]
                fvgs.append({
                    'type': 'BEARISH_FVG',
                    'high': fvg_high,
                    'low': fvg_low,
                    'timeframe': timeframe
                })
        
        return fvgs
    
    def detect_liquidity_zones(self, df, timeframe):
        """Detect liquidity zones (equal highs/lows)"""
        liquidity_zones = []
        if not isinstance(df, pd.DataFrame):
            return liquidity_zones
            
        lookback = 20
        
        # Detect equal highs (liquidity above)
        for i in range(lookback, len(df)-1):
            if abs(df['high'].iloc[i] - df['high'].iloc[i-1]) / df['high'].iloc[i] < 0.0002:  # Within 0.2 pips
                liquidity_zones.append({
                    'type': 'EQUAL_HIGHS',
                    'price': df['high'].iloc[i],
                    'timeframe': timeframe
                })
            
            # Detect equal lows (liquidity below)
            if abs(df['low'].iloc[i] - df['low'].iloc[i-1]) / df['low'].iloc[i] < 0.0002:  # Within 0.2 pips
                liquidity_zones.append({
                    'type': 'EQUAL_LOWS',
                    'price': df['low'].iloc[i],
                    'timeframe': timeframe
                })
        
        return liquidity_zones
    
    def detect_support_resistance(self, df, timeframe):
        """Detect key support and resistance levels"""
        levels = []
        if not isinstance(df, pd.DataFrame) or len(df) == 0:
            return levels
            
        try:
            df = self.find_swing_points(df)
            
            # Get recent swing highs (resistance)
            if 'is_swing_high' in df.columns:
                swing_highs = df[df['is_swing_high']]['high'].tail(10)
                for level in swing_highs:
                    levels.append({
                        'type': 'RESISTANCE',
                        'price': level,
                        'timeframe': timeframe
                    })
            
            # Get recent swing lows (support)
            if 'is_swing_low' in df.columns:
                swing_lows = df[df['is_swing_low']]['low'].tail(10)
                for level in swing_lows:
                    levels.append({
                        'type': 'SUPPORT',
                        'price': level,
                        'timeframe': timeframe
                    })
        except Exception as e:
            # If analysis fails, return empty levels
            pass
        
        return levels
    
    def generate_trade_reasons(self, symbol, setup_type, entry_price):
        """Generate detailed reasons for the trade"""
        reasons = []
        data = self.get_multiple_timeframes(symbol)
        
        if not data:
            return "No data available for analysis"
        
        current_price = mt5.symbol_info_tick(symbol).bid
        
        # Analyze each timeframe for trade reasons
        for tf in ['M15', 'H1', 'H4', 'D1']:
            if tf in data:
                df = data[tf]
                
                # Safety check
                if not isinstance(df, pd.DataFrame) or len(df) == 0:
                    continue
                    
                # FVG Analysis
                fvgs = self.detect_fvg(df, tf)
                for fvg in fvgs[-5:]:  # Last 5 FVGs
                    if (setup_type == 'BUY' and fvg['type'] == 'BULLISH_FVG' and 
                        fvg['low'] <= entry_price <= fvg['high']):
                        reasons.append(f"FVG@{fvg['low']:.5f}-{fvg['high']:.5f} on {tf}")
                    
                    if (setup_type == 'SELL' and fvg['type'] == 'BEARISH_FVG' and 
                        fvg['low'] <= entry_price <= fvg['high']):
                        reasons.append(f"FVG@{fvg['low']:.5f}-{fvg['high']:.5f} on {tf}")
                
                # Liquidity Analysis
                liquidity_zones = self.detect_liquidity_zones(df, tf)
                for zone in liquidity_zones[-10:]:  # Last 10 zones
                    distance_pips = abs(zone['price'] - entry_price) * 10000
                    if distance_pips < 50:  # Within 5 pips
                        if zone['type'] == 'EQUAL_HIGHS':
                            reasons.append(f"Liquidity observed at {zone['price']:.5f} (Equal Highs) on {tf}")
                        else:
                            reasons.append(f"Liquidity observed at {zone['price']:.5f} (Equal Lows) on {tf}")
                
                # Support/Resistance Analysis
                try:
                    sr_levels = self.detect_support_resistance(df, tf)
                    for level in sr_levels[-5:]:  # Last 5 levels
                        distance_pips = abs(level['price'] - entry_price) * 10000
                        if distance_pips < 30:  # Within 3 pips
                            if level['type'] == 'SUPPORT':
                                reasons.append(f"Support @ {level['price']:.5f} on {tf}")
                            else:
                                reasons.append(f"Resistance @ {level['price']:.5f} on {tf}")
                except Exception as e:
                    # If support/resistance analysis fails, skip it
                    continue
        
        # If no specific reasons found, provide general analysis
        if not reasons:
            reasons.append(f"Order block identified at {entry_price:.5f}")
            reasons.append("Market structure alignment")
        
        return " | ".join(reasons)
    
    def find_best_setups(self, setups, current_price):
        """Find only the best buy and sell setups - maximum 1 each"""
        best_buy = None
        best_sell = None
        
        # Filter out setups too close to existing positions
        valid_setups = []
        for setup in setups:
            if self.is_too_close_to_open_trades(setup['symbol'], setup['entry']):
                continue
            valid_setups.append(setup)
        
        for setup in valid_setups:
            if setup['type'] == 'BUY' and setup['entry'] < current_price:
                # Choose the HIGHEST buy limit (closest to current price)
                if best_buy is None or setup['entry'] > best_buy['entry']:
                    if setup['rr_ratio'] >= 1.2:  # Only consider good RR
                        best_buy = setup
            elif setup['type'] == 'SELL' and setup['entry'] > current_price:
                # Choose the LOWEST sell limit (closest to current price)
                if best_sell is None or setup['entry'] < best_sell['entry']:
                    if setup['rr_ratio'] >= 1.2:  # Only consider good RR
                        best_sell = setup
        
        # Return only the best ones
        result = []
        if best_buy:
            result.append(best_buy)
        if best_sell:
            result.append(best_sell)
            
        return result
    
    def is_too_close_to_open_trades(self, symbol, entry_price):
        """Check if entry is too close to existing open trades"""
        try:
            positions = mt5.positions_get(symbol=symbol)
            if positions:
                for position in positions:
                    # Check distance from open price
                    distance = abs(entry_price - position.price_open)
                    min_distance = entry_price * self.min_order_distance_percentage
                    if distance < min_distance:
                        print(f"⚠️ Order too close to open trade: {symbol} at {entry_price:.5f} (within {min_distance:.5f})")
                        return True
            return False
        except Exception as e:
            print(f"Error checking open trades for {symbol}: {e}")
            return False

    def find_swing_points(self, df, lookback=10):
        """Find swing highs and lows dynamically"""
        if not isinstance(df, pd.DataFrame) or len(df) < lookback * 2:
            # Return original dataframe with default columns if insufficient data
            if isinstance(df, pd.DataFrame):
                df = df.copy()
                df['is_swing_high'] = False
                df['is_swing_low'] = False
            return df
            
        df = df.copy()
        
        # Find swing highs
        df['is_swing_high'] = False
        for i in range(lookback, len(df)-lookback):
            if all(df['high'].iloc[i] > df['high'].iloc[i-lookback:i]) and \
               all(df['high'].iloc[i] > df['high'].iloc[i+1:i+lookback+1]):
                df.loc[df.index[i], 'is_swing_high'] = True
        
        # Find swing lows
        df['is_swing_low'] = False
        for i in range(lookback, len(df)-lookback):
            if all(df['low'].iloc[i] < df['low'].iloc[i-lookback:i]) and \
               all(df['low'].iloc[i] < df['low'].iloc[i+1:i+lookback+1]):
                df.loc[df.index[i], 'is_swing_low'] = True
        
        return df
    
    def calculate_dynamic_levels(self, symbol, entry_price, direction):
        """Calculate dynamic TP and SL based on market structure"""
        try:
            data = self.get_multiple_timeframes(symbol)
            if not data:
                return None, None
                
            # ADD DATA VALIDATION:
            h1_data = data.get('H1')
            h4_data = data.get('H4')
            
            if h1_data is None or not isinstance(h1_data, pd.DataFrame) or len(h1_data) == 0:
                # Fallback levels
                if direction == "BUY":
                    return entry_price * 0.998, entry_price * 1.004
                else:
                    return entry_price * 1.002, entry_price * 0.996
                
            # ENSURE SWING POINTS ARE CALCULATED:
            h1_data = self.find_swing_points(h1_data)
            h4_data = self.find_swing_points(h4_data) if h4_data is not None else h1_data
            
            if direction == "BUY":
                # For BUY: SL below recent swing low, TP at next resistance
                if 'is_swing_low' in h1_data.columns:
                    recent_lows = h1_data[h1_data['is_swing_low']]['low'].tail(5)
                else:
                    recent_lows = pd.Series([h1_data['low'].min()])
                    
                if 'is_swing_high' in h1_data.columns:
                    recent_highs = h1_data[h1_data['is_swing_high']]['high'].tail(5)
                else:
                    recent_highs = pd.Series([h1_data['high'].max()])
                
                if len(recent_lows) > 0:
                    # SL below the most recent significant swing low
                    sl = recent_lows.min() - (recent_lows.min() * 0.0001)  # 1 pip below
                else:
                    # Fallback: Use recent low with buffer
                    sl = h1_data['low'].min() - (h1_data['low'].min() * 0.0001)
                
                if len(recent_highs) > 0:
                    # TP at next swing high resistance
                    potential_tp_levels = recent_highs[recent_highs > entry_price]
                    if len(potential_tp_levels) > 0:
                        tp = potential_tp_levels.min()  # Nearest resistance
                    else:
                        # No resistance above, use 1.5x risk
                        risk = abs(entry_price - sl)
                        tp = entry_price + (risk * 1.5)
                else:
                    risk = abs(entry_price - sl)
                    tp = entry_price + (risk * 2.0)
                    
            else:  # SELL
                # For SELL: SL above recent swing high, TP at next support
                if 'is_swing_high' in h1_data.columns:
                    recent_highs = h1_data[h1_data['is_swing_high']]['high'].tail(5)
                else:
                    recent_highs = pd.Series([h1_data['high'].max()])
                    
                if 'is_swing_low' in h1_data.columns:
                    recent_lows = h1_data[h1_data['is_swing_low']]['low'].tail(5)
                else:
                    recent_lows = pd.Series([h1_data['low'].min()])
                
                if len(recent_highs) > 0:
                    # SL above the most recent significant swing high
                    sl = recent_highs.max() + (recent_highs.max() * 0.0001)  # 1 pip above
                else:
                    # Fallback: Use recent high with buffer
                    sl = h1_data['high'].max() + (h1_data['high'].max() * 0.0001)
                
                if len(recent_lows) > 0:
                    # TP at next swing low support
                    potential_tp_levels = recent_lows[recent_lows < entry_price]
                    if len(potential_tp_levels) > 0:
                        tp = potential_tp_levels.max()  # Nearest support
                    else:
                        # No support below, use 1.5x risk
                        risk = abs(sl - entry_price)
                        tp = entry_price - (risk * 1.5)
                else:
                    risk = abs(sl - entry_price)
                    tp = entry_price - (risk * 2.0)
            
            # Validate RR ratio (minimum 1:1)
            if direction == "BUY":
                risk = abs(entry_price - sl)
                reward = abs(tp - entry_price)
            else:
                risk = abs(sl - entry_price)
                reward = abs(entry_price - tp)
                
            rr_ratio = reward / risk if risk > 0 else 0
            
            # Adjust TP if RR is too low
            if rr_ratio < 1.0:
                if direction == "BUY":
                    tp = entry_price + risk * 1.2  # Minimum 1:1.2 RR
                else:
                    tp = entry_price - risk * 1.2
            
            print(f"🎯 {direction} Levels for {symbol}: Entry={entry_price:.5f}, SL={sl:.5f}, TP={tp:.5f}, RR={reward/risk:.2f}")
            
            return sl, tp
            
        except Exception as e:
            print(f"Error calculating dynamic levels for {symbol}: {e}")
            # Fallback to conservative fixed levels
            if direction == "BUY":
                return entry_price * 0.998, entry_price * 1.004
            else:
                return entry_price * 1.002, entry_price * 0.996
    
    def find_order_blocks_with_targets(self, symbol):
        """Find order blocks with dynamic TP/SL levels"""
        try:
            data = self.get_multiple_timeframes(symbol)
            if not data:
                return []
                
            # ADD DATA VALIDATION:
            h4_data = data.get('H4')
            h1_data = data.get('H1')
            
            if h4_data is None or not isinstance(h4_data, pd.DataFrame) or len(h4_data) == 0:
                return []
                
            # ENSURE SWING POINTS ARE CALCULATED:
            h4_data = self.find_swing_points(h4_data)
            h1_data = self.find_swing_points(h1_data) if h1_data is not None else h4_data
            
            # ADD COLUMN EXISTENCE CHECKS:
            if 'is_swing_high' not in h4_data.columns:
                h4_data['is_swing_high'] = False
                h4_data['is_swing_low'] = False
                
            # Detect order blocks more effectively
            h4_data['strong_bullish'] = (h4_data['close'] > h4_data['open']) & \
                                       ((h4_data['close'] - h4_data['open']) > (h4_data['high'] - h4_data['low']) * 0.6)
            h4_data['strong_bearish'] = (h4_data['close'] < h4_data['open']) & \
                                       ((h4_data['open'] - h4_data['close']) > (h4_data['high'] - h4_data['low']) * 0.6)
            
            h4_data['bullish_OB'] = h4_data['strong_bullish'] & h4_data['strong_bearish'].shift(1)
            h4_data['bearish_OB'] = h4_data['strong_bearish'] & h4_data['strong_bullish'].shift(1)
            
            current_price = mt5.symbol_info_tick(symbol).bid
            setups = []
            
            # Find bullish order blocks below current price
            recent_bullish_obs = h4_data[h4_data['bullish_OB']].tail(10)
            for idx, row in recent_bullish_obs.iterrows():
                ob_price = row['low']
                if ob_price < current_price and abs(current_price - ob_price) / current_price < 0.005:  # Within 0.5%
                    sl, tp = self.calculate_dynamic_levels(symbol, ob_price, "BUY")
                    if sl and tp:
                        rr_ratio = abs(tp - ob_price) / abs(ob_price - sl)
                        if rr_ratio >= 1.2:  # Only consider good RR setups
                            setups.append({
                                'symbol': symbol,
                                'type': 'BUY',
                                'entry': ob_price,
                                'sl': sl,
                                'tp': tp,
                                'rr_ratio': rr_ratio,
                                'distance_pips': abs(current_price - ob_price) * 10000
                            })
            
            # Find bearish order blocks above current price
            recent_bearish_obs = h4_data[h4_data['bearish_OB']].tail(10)
            for idx, row in recent_bearish_obs.iterrows():
                ob_price = row['high']
                if ob_price > current_price and abs(ob_price - current_price) / current_price < 0.005:  # Within 0.5%
                    sl, tp = self.calculate_dynamic_levels(symbol, ob_price, "SELL")
                    if sl and tp:
                        rr_ratio = abs(ob_price - tp) / abs(sl - ob_price)
                        if rr_ratio >= 1.2:  # Only consider good RR setups
                            setups.append({
                                'symbol': symbol,
                                'type': 'SELL',
                                'entry': ob_price,
                                'sl': sl,
                                'tp': tp,
                                'rr_ratio': rr_ratio,
                                'distance_pips': abs(ob_price - current_price) * 10000
                            })
            
            return setups
            
        except Exception as e:
            print(f"Error finding order blocks for {symbol}: {e}")
            return []
    
    def orders_need_update(self, symbol, new_setups):
        """Check if orders need to be updated - more tolerant"""
        if symbol not in self.last_order_levels:
            return True
            
        current_orders = self.get_pending_orders(symbol)
        if len(current_orders) == 0 and len(new_setups) > 0:
            return True
            
        # Get current order levels
        current_levels = {order.type: order.price_open for order in current_orders}
        
        # Get new setup levels
        new_levels = {}
        for setup in new_setups:
            if setup['type'] == 'BUY':
                new_levels[mt5.ORDER_TYPE_BUY_LIMIT] = setup['entry']
            else:
                new_levels[mt5.ORDER_TYPE_SELL_LIMIT] = setup['entry']
        
        # Check if levels changed significantly (more than 2 pips)
        for order_type, new_level in new_levels.items():
            if order_type in current_levels:
                current_level = current_levels[order_type]
                if abs(new_level - current_level) > 0.00020:  # 2 pips difference
                    return True
            else:
                return True
                
        return False

    def ml_trade_decision(self, symbol, setup):
        """Make ML-enhanced trade decision"""
        if not USE_ML:
            # If ML is disabled, approve all valid setups
            return {
                'symbol': symbol,
                'setup': setup,
                'ml_confidence': 1.0,
                'traditional_score': 1.0,
                'combined_score': 1.0,
                'reason': "ML DISABLED - Pure SMC/ICT Trading",
                'approved': True
            }
        
        try:
            market_data = self.get_multiple_timeframes(symbol)
            ml_confidence, ml_reason = self.ml_engine.predict_setup_quality(market_data)
            
            # Traditional scoring
            traditional_score = min(setup['rr_ratio'] / 3.0, 1.0) * 0.4 + \
                              (1.0 - min(setup['distance_pips'] / 100.0, 1.0)) * 0.3 + 0.3
            
            combined_score = 0.7 * ml_confidence + 0.3 * traditional_score
            
            # Generate comprehensive reason
            trade_reason = f"ML Confidence: {ml_confidence:.3f} | Traditional Score: {traditional_score:.3f} | Combined: {combined_score:.3f} | {ml_reason}"
            
            decision = {
                'symbol': symbol,
                'setup': setup,
                'ml_confidence': ml_confidence,
                'traditional_score': traditional_score,
                'combined_score': combined_score,
                'reason': trade_reason,
                'approved': combined_score > ML_CONFIDENCE_THRESHOLD
            }
            
            self.log_trade_decision(decision)
            return decision
            
        except Exception as e:
            print(f"ML decision error for {symbol}: {e}")
            # Fallback to traditional trading if ML fails
            return {
                'symbol': symbol, 'setup': setup, 'ml_confidence': 0.5,
                'traditional_score': 0.5, 'combined_score': 0.5,
                'reason': f"ML Error - Fallback to Traditional: {e}", 
                'approved': True  # Approve anyway as fallback
            }
    
    def log_trade_decision(self, decision):
        """Log trade decision to file"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"""
{'='*80}
Timestamp: {timestamp}
Symbol: {decision['symbol']}
Decision: {'✅ APPROVED' if decision['approved'] else '❌ REJECTED'}
ML Confidence: {decision['ml_confidence']:.3f}
Traditional Score: {decision['traditional_score']:.3f}
Combined Score: {decision['combined_score']:.3f}
Setup Type: {decision['setup']['type']}
Entry: {decision['setup']['entry']:.5f}
RR Ratio: {decision['setup']['rr_ratio']:.2f}
Reasons: {decision['reason']}
{'='*80}
"""
            
            with open(self.trade_log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
            
        except Exception as e:
            print(f"Error logging trade decision: {e}")

    def place_hybrid_orders(self, symbol):
        """Place orders with hybrid ML + SMC/ICT logic"""
        try:
            # Get current price
            current_price = mt5.symbol_info_tick(symbol).bid
            
            # Check if we're in kill zone (optional - can be removed)
            data = self.get_multiple_timeframes(symbol)
            if not data:
                return
                
            m15_data = data.get('M15')
            if m15_data is not None and isinstance(m15_data, pd.DataFrame) and len(m15_data) > 0:
                current_hour = pd.to_datetime(m15_data['time'].iloc[-1]).hour
                in_kill_zone = (1 <= current_hour <= 11) or (10 <= current_hour <= 20)
                
                if not in_kill_zone:
                    # Cancel orders outside kill zone (optional)
                    self.cancel_pending_orders(symbol)
                    self.last_order_levels.pop(symbol, None)
                    return
            
            # Find traditional SMC/ICT setups
            setups = self.find_order_blocks_with_targets(symbol)
            
            # Get only best setups (max 1 buy, 1 sell)
            best_setups = self.find_best_setups(setups, current_price)
            
            # ML-enhanced decision making
            approved_setups = []
            for setup in best_setups:
                decision = self.ml_trade_decision(symbol, setup)
                if decision['approved']:
                    approved_setups.append(setup)
                    mode = "ML" if USE_ML else "SMC/ICT"
                    print(f"✅ {mode} APPROVED: {symbol} {setup['type']} at {setup['entry']:.5f}")
                    if USE_ML:
                        print(f"   Confidence: {decision['ml_confidence']:.3f}, Combined: {decision['combined_score']:.3f}")
                else:
                    print(f"❌ ML REJECTED: {symbol} {setup['type']} - Confidence: {decision['ml_confidence']:.3f}")
            
            # Check if we need to update orders
            if not self.orders_need_update(symbol, approved_setups):
                return  # No changes needed
            
            # Cancel existing orders for this symbol
            self.cancel_pending_orders(symbol)
            
            # Place new orders for approved setups only
            for setup in approved_setups:
                if setup['type'] == 'BUY':
                    self.place_buy_limit_dynamic(setup)
                else:
                    self.place_sell_limit_dynamic(setup)
            
            # Store current setups to avoid duplicates
            if approved_setups:
                self.last_order_levels[symbol] = {setup['type']: setup['entry'] for setup in approved_setups}
            else:
                self.last_order_levels.pop(symbol, None)
                        
        except Exception as e:
            print(f"Error placing hybrid orders for {symbol}: {e}")

    def get_pending_orders(self, symbol=None):
        """Get current pending orders"""
        try:
            orders = mt5.orders_get(symbol=symbol) if symbol else mt5.orders_get()
            return [order for order in orders] if orders else []
        except:
            return []

    def place_buy_limit_dynamic(self, setup):
        """Place dynamic BUY LIMIT order with detailed reasoning"""
        # Generate trade reasons
        trade_reasons = self.generate_trade_reasons(setup['symbol'], 'BUY', setup['entry'])
        
        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": setup['symbol'],
            "volume": self.volume,
            "type": mt5.ORDER_TYPE_BUY_LIMIT,
            "price": setup['entry'],
            "sl": setup['sl'],
            "tp": setup['tp'],
            "deviation": 10,
            "magic": self.magic,
            "comment": f"HYBRID_BUY_RR{setup['rr_ratio']:.1f}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            mode = "ML" if USE_ML else "SMC/ICT"
            print(f"✅ {mode} BUY LIMIT: {setup['symbol']} at {setup['entry']:.5f}")
            print(f"   SL: {setup['sl']:.5f}, TP: {setup['tp']:.5f}, RR: {setup['rr_ratio']:.1f}")
            print(f"   📝 REASONS: {trade_reasons}")
            print("-" * 80)
            
            # Store trade reasons
            self.trade_reasons[result.order] = trade_reasons
            self.pending_orders[result.order] = setup['symbol']
        else:
            print(f"❌ Failed BUY LIMIT for {setup['symbol']}: {result.comment}")

    def place_sell_limit_dynamic(self, setup):
        """Place dynamic SELL LIMIT order with detailed reasoning"""
        # Generate trade reasons
        trade_reasons = self.generate_trade_reasons(setup['symbol'], 'SELL', setup['entry'])
        
        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": setup['symbol'],
            "volume": self.volume,
            "type": mt5.ORDER_TYPE_SELL_LIMIT,
            "price": setup['entry'],
            "sl": setup['sl'],
            "tp": setup['tp'],
            "deviation": 10,
            "magic": self.magic,
            "comment": f"HYBRID_SELL_RR{setup['rr_ratio']:.1f}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            mode = "ML" if USE_ML else "SMC/ICT"
            print(f"✅ {mode} SELL LIMIT: {setup['symbol']} at {setup['entry']:.5f}")
            print(f"   SL: {setup['sl']:.5f}, TP: {setup['tp']:.5f}, RR: {setup['rr_ratio']:.1f}")
            print(f"   📝 REASONS: {trade_reasons}")
            print("-" * 80)
            
            # Store trade reasons
            self.trade_reasons[result.order] = trade_reasons
            self.pending_orders[result.order] = setup['symbol']
        else:
            print(f"❌ Failed SELL LIMIT for {setup['symbol']}: {result.comment}")

    def cancel_pending_orders(self, symbol=None):
        """Cancel pending orders only when necessary"""
        try:
            orders = mt5.orders_get(symbol=symbol) if symbol else mt5.orders_get()
            if orders:
                cancelled_count = 0
                for order in orders:
                    if order.magic == self.magic:
                        result = mt5.order_send({
                            "action": mt5.TRADE_ACTION_REMOVE,
                            "order": order.ticket
                        })
                        if result.retcode == mt5.TRADE_RETCODE_DONE:
                            cancelled_count += 1
                        if order.ticket in self.pending_orders:
                            del self.pending_orders[order.ticket]
                        if order.ticket in self.trade_reasons:
                            del self.trade_reasons[order.ticket]
                if cancelled_count > 0:
                    print(f"📝 Cancelled {cancelled_count} orders for {symbol if symbol else 'all symbols'}")
        except Exception as e:
            print(f"Error cancelling pending orders: {e}")

    def check_trailing_stop(self, symbol, position):
        """Advanced trailing stop management - breakeven at 20% profit"""
        try:
            if symbol not in self.trailing_active:
                self.trailing_active[symbol] = False
            
            current_price = mt5.symbol_info_tick(symbol).bid if position.type == 0 else mt5.symbol_info_tick(symbol).ask
            open_price = position.price_open
            sl = position.sl
            tp = position.tp
            
            if position.type == 0:  # Buy position
                expected_profit = tp - open_price
                current_profit = current_price - open_price
                
                # Move to breakeven at 20% of expected profit
                if current_profit > expected_profit * 0.2 and not self.trailing_active[symbol]:
                    new_sl = open_price
                    self.mod_trade(position, new_sl, tp)
                    self.trailing_active[symbol] = True
                    print(f"🔒 Trailing stop activated for {symbol} - SL moved to breakeven at 20% progress")
                
                # Continue trailing beyond breakeven
                elif self.trailing_active[symbol] and current_price > sl + (tp - open_price) * 0.05:
                    new_sl = current_price - (tp - open_price) * 0.05
                    if new_sl > sl:
                        self.mod_trade(position, new_sl, tp)
                        print(f"📈 Trailing stop updated for {symbol} - New SL: {new_sl:.5f}")
                        
            else:  # Sell position
                expected_profit = open_price - tp
                current_profit = open_price - current_price
                
                # Move to breakeven at 20% of expected profit
                if current_profit > expected_profit * 0.2 and not self.trailing_active[symbol]:
                    new_sl = open_price
                    self.mod_trade(position, new_sl, tp)
                    self.trailing_active[symbol] = True
                    print(f"🔒 Trailing stop activated for {symbol} - SL moved to breakeven at 20% progress")
                
                # Continue trailing beyond breakeven
                elif self.trailing_active[symbol] and current_price < sl - (open_price - tp) * 0.05:
                    new_sl = current_price + (open_price - tp) * 0.05
                    if new_sl < sl:
                        self.mod_trade(position, new_sl, tp)
                        print(f"📈 Trailing stop updated for {symbol} - New SL: {new_sl:.5f}")
                        
        except Exception as e:
            print(f"Error in trailing stop for {symbol}: {e}")

    def mod_trade(self, position, sl, tp):
        """Modify trade SL and TP"""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": position.ticket,
            "sl": sl,
            "tp": tp,
            "symbol": position.symbol,
            "type_time": mt5.ORDER_TIME_GTC,
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"⚡ SL/TP Updated {position.symbol} - SL: {sl:.5f}, TP: {tp:.5f}")
        else:
            print(f"❌ Failed to update stop for ticket {position.ticket}")
        return result

    def collect_training_data(self):
        """Collect historical data for training"""
        print("📊 Collecting training data...")
        training_data = []
        
        for symbol in pairs[:5]:  # Use first 5 pairs for training
            try:
                # Get historical data
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 1000)
                if rates is None:
                    continue
                    
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                
                # Simulate historical trades (in real implementation, use actual trade history)
                for i in range(50, len(df)-10):
                    market_data = self.get_multiple_timeframes(symbol)
                    entry_price = df['close'].iloc[i]
                    exit_price = df['close'].iloc[i+10]
                    profit = exit_price - entry_price
                    
                    training_data.append({
                        'symbol': symbol,
                        'market_data': market_data,
                        'profit': profit,
                        'timestamp': df['time'].iloc[i]
                    })
                    
            except Exception as e:
                print(f"Error collecting training data for {symbol}: {e}")
        
        print(f"✅ Collected {len(training_data)} training samples")
        return training_data

    def train_ml_model(self):
        """Train the ML model on collected data"""
        if not USE_ML:
            print("🤖 ML Mode is disabled - skipping training")
            return
            
        print("🚀 Starting ML Model Training...")
        training_data = self.collect_training_data()
        
        if len(training_data) >= 50:
            success = self.ml_engine.train_model(training_data)
            if success:
                self.ml_engine.save_model("smc_ml_model.pkl")
                print("🎉 ML Model training completed successfully!")
            else:
                print("❌ ML Model training failed!")
        else:
            print(f"⚠️ Not enough training data: {len(training_data)} samples (need 50+)")


    def find_ltf_entry_signals(self, symbol, htf_setup):
        """Find precise entry signals on lower timeframe"""
        try:
            # Get LTF data (M5 for precise entry)
            ltf_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 50)
            if ltf_rates is None or len(ltf_rates) == 0:
                return None
                
            ltf_df = pd.DataFrame(ltf_rates)
            ltf_df['time'] = pd.to_datetime(ltf_df['time'], unit='s')
            
            current_price = mt5.symbol_info_tick(symbol).bid
            setup_type = htf_setup['type']
            htf_entry = htf_setup['entry']
            
            # LTF Entry Conditions for BUY
            if setup_type == 'BUY':
                # Look for LTF bullish confirmation
                ltf_df = self.find_swing_points(ltf_df, lookback=3)
                
                recent_low = ltf_df['low'].tail(5).min()
                recent_high = ltf_df['high'].tail(5).max()
                
                # Check if price is approaching HTF level from above
                if current_price <= htf_entry * 1.001 and current_price >= htf_entry * 0.999:
                    # Price is at HTF level - look for LTF confirmation
                    
                    ltf_fvgs = self.detect_fvg(ltf_df, 'M5')
                    recent_bullish_fvg = any(fvg['type'] == 'BULLISH_FVG' for fvg in ltf_fvgs[-3:])
                    
                    # Check for bullish candle patterns
                    if len(ltf_df) >= 2:
                        last_candle = ltf_df.iloc[-1]
                        prev_candle = ltf_df.iloc[-2]
                        
                        bullish_engulfing = (last_candle['close'] > last_candle['open'] and 
                                           prev_candle['close'] < prev_candle['open'] and
                                           last_candle['close'] > prev_candle['open'] and
                                           last_candle['open'] < prev_candle['close'])
                        
                        if recent_bullish_fvg or bullish_engulfing:
                            # LTF confirmation found - use tighter SL for better RR
                            ltf_sl = ltf_df['low'].tail(10).min()
                            ltf_tp = htf_setup['tp']  # Keep HTF TP for higher RR
                            
                            # Calculate improved RR
                            risk = abs(htf_entry - ltf_sl)
                            reward = abs(ltf_tp - htf_entry)
                            rr_ratio = reward / risk if risk > 0 else 0
                            
                            if rr_ratio >= MIN_RR_RATIO:
                                return {
                                    'symbol': symbol,
                                    'type': 'BUY',
                                    'entry': htf_entry,
                                    'sl': ltf_sl,
                                    'tp': ltf_tp,
                                    'rr_ratio': rr_ratio,
                                    'htf_source': 'H4',
                                    'ltf_confirmation': 'M5',
                                    'reason': f"HTF OB + LTF Bullish Confirmation (RR: {rr_ratio:.1f})"
                                }
            
            # LTF Entry Conditions for SELL
            elif setup_type == 'SELL':
                ltf_df = self.find_swing_points(ltf_df, lookback=3)
                
                if current_price >= htf_entry * 0.999 and current_price <= htf_entry * 1.001:
                    # Price at HTF level - look for LTF bearish confirmation
                    
                    ltf_fvgs = self.detect_fvg(ltf_df, 'M5')
                    recent_bearish_fvg = any(fvg['type'] == 'BEARISH_FVG' for fvg in ltf_fvgs[-3:])
                    
                    if len(ltf_df) >= 2:
                        last_candle = ltf_df.iloc[-1]
                        prev_candle = ltf_df.iloc[-2]
                        
                        bearish_engulfing = (last_candle['close'] < last_candle['open'] and 
                                           prev_candle['close'] > prev_candle['open'] and
                                           last_candle['close'] < prev_candle['open'] and
                                           last_candle['open'] > prev_candle['close'])
                        
                        if recent_bearish_fvg or bearish_engulfing:
                            ltf_sl = ltf_df['high'].tail(10).max()
                            ltf_tp = htf_setup['tp']
                            
                            risk = abs(ltf_sl - htf_entry)
                            reward = abs(htf_entry - ltf_tp)
                            rr_ratio = reward / risk if risk > 0 else 0
                            
                            if rr_ratio >= MIN_RR_RATIO:
                                return {
                                    'symbol': symbol,
                                    'type': 'SELL',
                                    'entry': htf_entry,
                                    'sl': ltf_sl,
                                    'tp': ltf_tp,
                                    'rr_ratio': rr_ratio,
                                    'htf_source': 'H4',
                                    'ltf_confirmation': 'M5',
                                    'reason': f"HTF OB + LTF Bearish Confirmation (RR: {rr_ratio:.1f})"
                                }
            
            return None
            
        except Exception as e:
            print(f"Error finding LTF entry for {symbol}: {e}")
            return None
    
    def find_scalping_setups(self, symbol):
        """Find HTF setups with LTF entry confirmation"""
        # First, find HTF order blocks (existing logic)
        htf_setups = self.find_order_blocks_with_targets(symbol)
        
        scalping_setups = []
        
        for htf_setup in htf_setups:
            # For each HTF setup, look for LTF entry confirmation
            ltf_entry = self.find_ltf_entry_signals(symbol, htf_setup)
            
            if ltf_entry:
                scalping_setups.append(ltf_entry)
        
        return scalping_setups
    
    def place_scalping_orders(self, symbol):
        """New HTF+LTF scalping logic"""
        try:
            current_price = mt5.symbol_info_tick(symbol).bid
            
            # Get fresh HTF+LTF setups
            scalping_setups = self.find_scalping_setups(symbol)
            
            # Filter best setups
            best_setups = self.find_best_setups(scalping_setups, current_price)
            
            # ML decision making (if enabled)
            approved_setups = []
            for setup in best_setups:
                if USE_ML:
                    decision = self.ml_trade_decision(symbol, setup)
                    if decision['approved']:
                        approved_setups.append(setup)
                        print(f"✅ SCALPING ML APPROVED: {symbol} {setup['type']} at {setup['entry']:.5f}")
                        print(f"   RR: {setup['rr_ratio']:.1f}, LTF Confirmed: {setup.get('ltf_confirmation', 'M5')}")
                else:
                    approved_setups.append(setup)
                    print(f"✅ SCALPING APPROVED: {symbol} {setup['type']} at {setup['entry']:.5f}")
                    print(f"   RR: {setup['rr_ratio']:.1f}, LTF Confirmed: {setup.get('ltf_confirmation', 'M5')}")
            
            # Place orders
            if self.orders_need_update(symbol, approved_setups):
                self.cancel_pending_orders(symbol)
                
                for setup in approved_setups:
                    if setup['type'] == 'BUY':
                        self.place_buy_limit_dynamic(setup)
                    else:
                        self.place_sell_limit_dynamic(setup)
                        
        except Exception as e:
            print(f"Error in scalping order placement for {symbol}: {e}")



# COMPLETE MAIN TRADING LOOP
def main_trading_loop():
    strategy = HybridSMCStrategy()
    
    mode = "ML-ENHANCED" if USE_ML else "PURE SMC/ICT"
    print(f"🚀 Starting {mode} Trading Bot...")
    print("=" * 80)
    
    # Train model if not loaded and ML mode is enabled
    if USE_ML and not strategy.ml_engine.model_trained:
        print("🤖 No pre-trained model found. Starting training...")
        strategy.train_ml_model()
    
    scan_count = 0
    
    while True:
        try:
            current_time = datetime.datetime.now(tz)
            
            # Full analysis every 20 scans
            if scan_count % 20 == 0:
                print(f"\n⏰ {current_time.strftime('%Y-%m-%d %H:%M:%S')} - {mode} Analysis Scan...")
                print("-" * 80)
                
                for pair in pairs:
                    try:
                        setups = strategy.find_order_blocks_with_targets(pair)
                        current_price = mt5.symbol_info_tick(pair).bid
                        
                        if setups:
                            best_setups = strategy.find_best_setups(setups, current_price)
                            print(f"📊 {pair}: {len(best_setups)} traditional setups")
                            
                            for setup in best_setups:
                                if USE_ML:
                                    decision = strategy.ml_trade_decision(pair, setup)
                                    status = "APPROVED" if decision['approved'] else "REJECTED"
                                    print(f"   {setup['type']} at {setup['entry']:.5f} - {status} (ML: {decision['ml_confidence']:.3f})")
                                else:
                                    print(f"   {setup['type']} at {setup['entry']:.5f} (RR: {setup['rr_ratio']:.1f})")
                        else:
                            print(f"📊 {pair}: No traditional setups")
                    
                    except Exception as e:
                        print(f"❌ Error analyzing {pair}: {e}")
                        continue
            
            # Order placement every minute
            if scan_count % 2 == 0:
                print(f"\n⏰ {current_time.strftime('%H:%M:%S')} - {mode} Order Placement...")
                for pair in pairs:
                    try:
                        if SCALPING_MODE:
                            strategy.place_scalping_orders(pair)
                        else:
                            strategy.place_hybrid_orders(pair)
                    except Exception as e:
                        print(f"❌ Error in order placement for {pair}: {e}")
            
            # Retrain model periodically (only in ML mode)
            if USE_ML and scan_count % 200 == 0 and scan_count > 0:
                print("🔄 Retraining ML model...")
                strategy.train_ml_model()
            
            # Manage existing positions (every scan)
            positions = mt5.positions_get()
            if positions:
                active_bot_positions = [p for p in positions if p.magic == strategy.magic]
                if active_bot_positions:
                    print(f"📈 Managing {len(active_bot_positions)} positions...")
                    for position in active_bot_positions:
                        strategy.check_trailing_stop(position.symbol, position)
            
            scan_count += 1
            time.sleep(30)
            
        except KeyboardInterrupt:
            print(f"\n🛑 Stopping {mode} trading bot...")
            break
        except Exception as e:
            print(f"💥 Error in main loop: {e}")
            time.sleep(60)    


if __name__ == "__main__":
    main_trading_loop()