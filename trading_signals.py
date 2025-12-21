"""
Trading Signal Generation Module

This module provides functionality for generating trading signals based on technical
analysis indicators and market data analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """Enumeration for trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"


class Signal_Strength(Enum):
    """Enumeration for signal strength levels"""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4


@dataclass
class TradingSignal:
    """Data class representing a trading signal"""
    timestamp: datetime
    signal_type: SignalType
    strength: Signal_Strength
    confidence: float  # 0.0 to 1.0
    price: float
    indicators_used: List[str]
    reason: str


class TradingSignalGenerator:
    """
    Generates trading signals based on technical analysis indicators
    """

    def __init__(self, lookback_period: int = 20):
        """
        Initialize the trading signal generator
        
        Args:
            lookback_period: Number of candles to look back for analysis
        """
        self.lookback_period = lookback_period

    def calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average
        
        Args:
            prices: Series of prices
            period: SMA period
            
        Returns:
            Series of SMA values
        """
        return prices.rolling(window=period).mean()

    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average
        
        Args:
            prices: Series of prices
            period: EMA period
            
        Returns:
            Series of EMA values
        """
        return prices.ewm(span=period, adjust=False).mean()

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index
        
        Args:
            prices: Series of prices
            period: RSI period (default 14)
            
        Returns:
            Series of RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices: pd.Series, fast: int = 12, 
                      slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            prices: Series of prices
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            Tuple of (MACD, Signal, Histogram)
        """
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        return macd, signal_line, histogram

    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, 
                                  std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Args:
            prices: Series of prices
            period: SMA period for middle band
            std_dev: Standard deviation multiplier
            
        Returns:
            Tuple of (Upper Band, Middle Band, Lower Band)
        """
        middle_band = self.calculate_sma(prices, period)
        std = prices.rolling(window=period).std()
        
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        
        return upper_band, middle_band, lower_band

    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                            period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            period: Period for calculation
            
        Returns:
            Tuple of (%K, %D)
        """
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=3).mean()
        
        return k_percent, d_percent

    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                     period: int = 14) -> pd.Series:
        """
        Calculate Average True Range
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            period: ATR period
            
        Returns:
            Series of ATR values
        """
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr

    def generate_signal(self, df: pd.DataFrame) -> Optional[TradingSignal]:
        """
        Generate a trading signal based on multiple technical indicators
        
        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
            
        Returns:
            TradingSignal object or None if insufficient data
        """
        if len(df) < self.lookback_period:
            return None

        current_price = df['close'].iloc[-1]
        timestamp = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else datetime.utcnow()

        # Initialize scoring system
        buy_signals = 0
        sell_signals = 0
        indicators_used = []

        # 1. Moving Average Analysis
        sma_20 = self.calculate_sma(df['close'], 20).iloc[-1]
        sma_50 = self.calculate_sma(df['close'], 50).iloc[-1] if len(df) >= 50 else None
        ema_12 = self.calculate_ema(df['close'], 12).iloc[-1]

        if current_price > sma_20:
            buy_signals += 1
        else:
            sell_signals += 1
        indicators_used.append("SMA_20")

        if sma_50 and current_price > sma_50:
            buy_signals += 1
            indicators_used.append("SMA_50")

        # 2. RSI Analysis
        rsi = self.calculate_rsi(df['close']).iloc[-1]
        indicators_used.append("RSI_14")
        if rsi < 30:
            buy_signals += 2  # Oversold
        elif rsi > 70:
            sell_signals += 2  # Overbought
        elif 40 < rsi < 60:
            pass  # Neutral

        # 3. MACD Analysis
        macd, signal_line, histogram = self.calculate_macd(df['close'])
        macd_val = macd.iloc[-1]
        signal_val = signal_line.iloc[-1]
        hist_val = histogram.iloc[-1]
        indicators_used.append("MACD")

        if macd_val > signal_val and hist_val > 0:
            buy_signals += 1
        elif macd_val < signal_val and hist_val < 0:
            sell_signals += 1

        # 4. Bollinger Bands Analysis
        upper_bb, middle_bb, lower_bb = self.calculate_bollinger_bands(df['close'])
        upper_val = upper_bb.iloc[-1]
        lower_val = lower_bb.iloc[-1]
        indicators_used.append("Bollinger_Bands")

        if current_price < lower_val:
            buy_signals += 1
        elif current_price > upper_val:
            sell_signals += 1

        # 5. Stochastic Analysis
        if len(df) >= 14:
            k_percent, d_percent = self.calculate_stochastic(df['high'], df['low'], df['close'])
            k_val = k_percent.iloc[-1]
            d_val = d_percent.iloc[-1]
            indicators_used.append("Stochastic")

            if k_val < 20:
                buy_signals += 1
            elif k_val > 80:
                sell_signals += 1

        # Determine signal type and strength
        total_signals = buy_signals + sell_signals
        
        if total_signals == 0:
            signal_type = SignalType.HOLD
            strength = Signal_Strength.WEAK
            confidence = 0.5
            reason = "No clear signal from technical indicators"
        elif buy_signals > sell_signals * 1.5:
            if buy_signals >= 5:
                signal_type = SignalType.STRONG_BUY
                strength = Signal_Strength.VERY_STRONG
                confidence = min(0.95, buy_signals / total_signals)
            else:
                signal_type = SignalType.BUY
                strength = Signal_Strength.STRONG if buy_signals >= 3 else Signal_Strength.MODERATE
                confidence = buy_signals / total_signals
            reason = f"Multiple buy indicators aligned: {buy_signals} positive signals"
        elif sell_signals > buy_signals * 1.5:
            if sell_signals >= 5:
                signal_type = SignalType.STRONG_SELL
                strength = Signal_Strength.VERY_STRONG
                confidence = min(0.95, sell_signals / total_signals)
            else:
                signal_type = SignalType.SELL
                strength = Signal_Strength.STRONG if sell_signals >= 3 else Signal_Strength.MODERATE
                confidence = sell_signals / total_signals
            reason = f"Multiple sell indicators aligned: {sell_signals} negative signals"
        else:
            signal_type = SignalType.HOLD
            strength = Signal_Strength.MODERATE
            confidence = 0.5
            reason = "Mixed signals from indicators"

        return TradingSignal(
            timestamp=timestamp,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            price=current_price,
            indicators_used=indicators_used,
            reason=reason
        )

    def generate_signals_batch(self, df: pd.DataFrame) -> List[TradingSignal]:
        """
        Generate trading signals for a batch of data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            List of TradingSignal objects
        """
        signals = []
        for i in range(self.lookback_period, len(df)):
            subset = df.iloc[:i+1]
            signal = self.generate_signal(subset)
            if signal:
                signals.append(signal)
        return signals


# Utility functions
def filter_signals_by_strength(signals: List[TradingSignal], 
                              min_strength: Signal_Strength) -> List[TradingSignal]:
    """Filter trading signals by minimum strength level"""
    return [s for s in signals if s.strength.value >= min_strength.value]


def filter_signals_by_confidence(signals: List[TradingSignal], 
                                min_confidence: float) -> List[TradingSignal]:
    """Filter trading signals by minimum confidence level"""
    return [s for s in signals if s.confidence >= min_confidence]


def get_signal_statistics(signals: List[TradingSignal]) -> Dict:
    """Calculate statistics about a list of signals"""
    if not signals:
        return {}
    
    signal_types = {}
    for signal in signals:
        signal_type = signal.signal_type.value
        signal_types[signal_type] = signal_types.get(signal_type, 0) + 1
    
    return {
        "total_signals": len(signals),
        "signal_distribution": signal_types,
        "average_confidence": np.mean([s.confidence for s in signals]),
        "max_confidence": max([s.confidence for s in signals]),
        "min_confidence": min([s.confidence for s in signals])
    }


if __name__ == "__main__":
    # Example usage
    print("Trading Signal Generation Module")
    print("=" * 50)
    print("Module loaded successfully")
