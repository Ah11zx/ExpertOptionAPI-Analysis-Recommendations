"""
Technical Analysis Module

This module provides technical analysis tools and indicators for analyzing
financial market data and generating trading recommendations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class TechnicalAnalyzer:
    """
    A class for performing technical analysis on financial data.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the TechnicalAnalyzer with market data.

        Args:
            data (pd.DataFrame): DataFrame containing OHLCV (Open, High, Low, Close, Volume) data
        """
        self.data = data.copy()
        self.indicators = {}

    def calculate_sma(self, column: str = 'close', period: int = 20) -> pd.Series:
        """
        Calculate Simple Moving Average (SMA).

        Args:
            column (str): Column name to calculate SMA on
            period (int): Period for SMA calculation

        Returns:
            pd.Series: Series containing SMA values
        """
        return self.data[column].rolling(window=period).mean()

    def calculate_ema(self, column: str = 'close', period: int = 20) -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA).

        Args:
            column (str): Column name to calculate EMA on
            period (int): Period for EMA calculation

        Returns:
            pd.Series: Series containing EMA values
        """
        return self.data[column].ewm(span=period, adjust=False).mean()

    def calculate_rsi(self, column: str = 'close', period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        Args:
            column (str): Column name to calculate RSI on
            period (int): Period for RSI calculation

        Returns:
            pd.Series: Series containing RSI values
        """
        delta = self.data[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_macd(self, column: str = 'close', fast: int = 12, 
                       slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            column (str): Column name to calculate MACD on
            fast (int): Fast EMA period
            slow (int): Slow EMA period
            signal (int): Signal line period

        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: MACD line, Signal line, and Histogram
        """
        ema_fast = self.data[column].ewm(span=fast, adjust=False).mean()
        ema_slow = self.data[column].ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def calculate_bollinger_bands(self, column: str = 'close', period: int = 20, 
                                   std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Args:
            column (str): Column name to calculate Bollinger Bands on
            period (int): Period for moving average
            std_dev (float): Number of standard deviations

        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: Upper band, Middle band, Lower band
        """
        middle_band = self.data[column].rolling(window=period).mean()
        std = self.data[column].rolling(window=period).std()

        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)

        return upper_band, middle_band, lower_band

    def calculate_atr(self, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).

        Args:
            period (int): Period for ATR calculation

        Returns:
            pd.Series: Series containing ATR values
        """
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def calculate_stochastic(self, period: int = 14, 
                            smooth_k: int = 3, smooth_d: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.

        Args:
            period (int): Period for calculation
            smooth_k (int): Smoothing period for %K
            smooth_d (int): Smoothing period for %D

        Returns:
            Tuple[pd.Series, pd.Series]: %K and %D values
        """
        low_min = self.data['low'].rolling(window=period).min()
        high_max = self.data['high'].rolling(window=period).max()

        k_percent = 100 * ((self.data['close'] - low_min) / (high_max - low_min))
        k_percent_smooth = k_percent.rolling(window=smooth_k).mean()
        d_percent = k_percent_smooth.rolling(window=smooth_d).mean()

        return k_percent_smooth, d_percent

    def generate_signals(self) -> Dict[str, str]:
        """
        Generate trading signals based on technical indicators.

        Returns:
            Dict[str, str]: Dictionary containing signal recommendations
        """
        signals = {}

        # RSI Signal
        rsi = self.calculate_rsi()
        current_rsi = rsi.iloc[-1]
        if current_rsi > 70:
            signals['RSI'] = 'OVERBOUGHT'
        elif current_rsi < 30:
            signals['RSI'] = 'OVERSOLD'
        else:
            signals['RSI'] = 'NEUTRAL'

        # MACD Signal
        macd, signal_line, hist = self.calculate_macd()
        if macd.iloc[-1] > signal_line.iloc[-1]:
            signals['MACD'] = 'BULLISH'
        else:
            signals['MACD'] = 'BEARISH'

        # Bollinger Bands Signal
        upper, middle, lower = self.calculate_bollinger_bands()
        if self.data['close'].iloc[-1] > upper.iloc[-1]:
            signals['BB'] = 'OVERBOUGHT'
        elif self.data['close'].iloc[-1] < lower.iloc[-1]:
            signals['BB'] = 'OVERSOLD'
        else:
            signals['BB'] = 'NEUTRAL'

        return signals


def analyze_market(data: pd.DataFrame, symbol: str = 'Unknown') -> Dict:
    """
    Perform comprehensive technical analysis on market data.

    Args:
        data (pd.DataFrame): Market data with OHLCV columns
        symbol (str): Symbol identifier for the data

    Returns:
        Dict: Dictionary containing analysis results and recommendations
    """
    analyzer = TechnicalAnalyzer(data)

    analysis_result = {
        'symbol': symbol,
        'timestamp': pd.Timestamp.now(),
        'signals': analyzer.generate_signals(),
        'indicators': {}
    }

    # Add latest indicator values
    analysis_result['indicators']['RSI'] = analyzer.calculate_rsi().iloc[-1]
    analysis_result['indicators']['SMA_20'] = analyzer.calculate_sma().iloc[-1]
    analysis_result['indicators']['EMA_20'] = analyzer.calculate_ema().iloc[-1]
    analysis_result['indicators']['ATR'] = analyzer.calculate_atr().iloc[-1]

    return analysis_result


if __name__ == '__main__':
    # Example usage
    print("Technical Analysis Module loaded successfully")
