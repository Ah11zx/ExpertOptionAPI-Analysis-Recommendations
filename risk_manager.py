"""
Risk Management Module for Trading
Provides comprehensive risk management functionality including:
- Position sizing calculations
- Stop loss and take profit calculations
- Risk metrics and analysis
- Portfolio risk assessment
- Value at Risk (VaR) calculations
- Risk-adjusted performance metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskMetric(Enum):
    """Enumeration for risk metrics types"""
    VAR = "Value at Risk"
    SHARPE = "Sharpe Ratio"
    SORTINO = "Sortino Ratio"
    MAX_DRAWDOWN = "Maximum Drawdown"
    CALMAR = "Calmar Ratio"
    RETURN_DD = "Return to Drawdown"


@dataclass
class PositionSize:
    """Data class for position sizing results"""
    units: float
    cost: float
    risk_amount: float
    reward_amount: float
    risk_reward_ratio: float
    percentage_of_capital: float


@dataclass
class RiskMetrics:
    """Data class for risk metrics"""
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    win_rate: float
    profit_factor: float
    recovery_factor: float


@dataclass
class PortfolioRisk:
    """Data class for portfolio risk assessment"""
    total_risk: float
    diversification_ratio: float
    correlation_avg: float
    concentration_risk: float
    var_portfolio: float
    marginal_var: Dict[str, float]
    component_var: Dict[str, float]


class RiskManager:
    """
    Comprehensive risk management system for trading strategies
    Handles position sizing, risk calculations, and portfolio analysis
    """

    def __init__(self, initial_capital: float, risk_per_trade: float = 0.02):
        """
        Initialize Risk Manager
        
        Args:
            initial_capital: Starting capital in currency
            risk_per_trade: Risk percentage per trade (default 2%)
        """
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.positions = {}
        self.trade_history = []
        self.equity_curve = [initial_capital]
        
        logger.info(f"RiskManager initialized with capital: ${initial_capital:,.2f}")

    # ==================== POSITION SIZING ====================
    
    def calculate_position_size_fixed_risk(
        self,
        entry_price: float,
        stop_loss_price: float,
        current_capital: Optional[float] = None,
        risk_amount: Optional[float] = None
    ) -> PositionSize:
        """
        Calculate position size based on fixed risk amount
        
        Args:
            entry_price: Entry price for the position
            stop_loss_price: Stop loss price
            current_capital: Current trading capital (uses initial if None)
            risk_amount: Fixed risk amount (uses % of capital if None)
            
        Returns:
            PositionSize object with detailed sizing information
        """
        capital = current_capital or self.initial_capital
        
        if risk_amount is None:
            risk_amount = capital * self.risk_per_trade
        
        price_difference = abs(entry_price - stop_loss_price)
        
        if price_difference == 0:
            logger.warning("Entry and stop loss prices are the same")
            return PositionSize(0, 0, risk_amount, 0, 0, 0)
        
        units = risk_amount / price_difference
        position_cost = units * entry_price
        percentage = (position_cost / capital) * 100
        
        logger.info(f"Position size calculated: {units:.2f} units (${position_cost:,.2f})")
        
        return PositionSize(
            units=units,
            cost=position_cost,
            risk_amount=risk_amount,
            reward_amount=0,  # Will be set if take profit is provided
            risk_reward_ratio=0,
            percentage_of_capital=percentage
        )

    def calculate_position_size_kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        current_capital: Optional[float] = None,
        safety_factor: float = 0.25
    ) -> float:
        """
        Calculate position size using Kelly Criterion
        Kelly % = (bp - q) / b
        where b=odds, p=win probability, q=loss probability
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average win amount in currency
            avg_loss: Average loss amount in currency
            current_capital: Current trading capital
            safety_factor: Safety factor for fractional kelly (0-1)
            
        Returns:
            Optimal position size percentage of capital
        """
        if win_rate < 0 or win_rate > 1:
            logger.error("Win rate must be between 0 and 1")
            return 0
        
        if avg_loss == 0:
            logger.error("Average loss cannot be zero")
            return 0
        
        loss_rate = 1 - win_rate
        b = avg_win / avg_loss
        
        kelly_percent = (win_rate * b - loss_rate) / b
        kelly_percent = max(0, kelly_percent)  # Ensure non-negative
        kelly_percent *= safety_factor  # Apply safety factor
        
        logger.info(f"Kelly Criterion optimal allocation: {kelly_percent*100:.2f}%")
        return min(kelly_percent, 0.25)  # Cap at 25% for safety

    def calculate_optimal_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        win_rate: float,
        current_capital: Optional[float] = None
    ) -> PositionSize:
        """
        Calculate optimal position size considering risk-reward ratio
        
        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss level
            take_profit_price: Take profit target
            win_rate: Expected win rate
            current_capital: Current capital
            
        Returns:
            PositionSize with optimal allocation
        """
        capital = current_capital or self.initial_capital
        
        risk_per_unit = abs(entry_price - stop_loss_price)
        reward_per_unit = abs(take_profit_price - entry_price)
        
        if risk_per_unit == 0:
            return PositionSize(0, 0, 0, 0, 0, 0)
        
        # Expected value calculation
        expected_value = (win_rate * reward_per_unit) - ((1 - win_rate) * risk_per_unit)
        
        if expected_value <= 0:
            logger.warning("Negative expected value - trade not recommended")
            return PositionSize(0, 0, 0, 0, 0, 0)
        
        # Kelly criterion based sizing
        kelly_fraction = self.calculate_position_size_kelly_criterion(
            win_rate, reward_per_unit, risk_per_unit, capital
        )
        
        position_amount = capital * kelly_fraction
        units = position_amount / entry_price
        risk_amount = units * risk_per_unit
        reward_amount = units * reward_per_unit
        risk_reward_ratio = reward_per_unit / risk_per_unit if risk_per_unit > 0 else 0
        
        return PositionSize(
            units=units,
            cost=position_amount,
            risk_amount=risk_amount,
            reward_amount=reward_amount,
            risk_reward_ratio=risk_reward_ratio,
            percentage_of_capital=(position_amount / capital) * 100
        )

    # ==================== STOP LOSS & TAKE PROFIT ====================
    
    def calculate_atr_stop_loss(
        self,
        current_price: float,
        atr: float,
        atr_multiplier: float = 2.0,
        position_type: str = "long"
    ) -> float:
        """
        Calculate stop loss based on ATR (Average True Range)
        
        Args:
            current_price: Current price
            atr: Average True Range value
            atr_multiplier: Multiplier for ATR (default 2.0)
            position_type: "long" or "short"
            
        Returns:
            Stop loss price
        """
        if position_type.lower() == "long":
            stop_loss = current_price - (atr * atr_multiplier)
        elif position_type.lower() == "short":
            stop_loss = current_price + (atr * atr_multiplier)
        else:
            raise ValueError("Position type must be 'long' or 'short'")
        
        return stop_loss

    def calculate_risk_reward_levels(
        self,
        entry_price: float,
        atr: float,
        target_ratio: float = 2.0,
        position_type: str = "long"
    ) -> Dict[str, float]:
        """
        Calculate stop loss and take profit with target risk-reward ratio
        
        Args:
            entry_price: Entry price
            atr: Average True Range
            target_ratio: Target risk-reward ratio (e.g., 2.0 = 1:2)
            position_type: "long" or "short"
            
        Returns:
            Dictionary with stop_loss and take_profit levels
        """
        stop_loss = self.calculate_atr_stop_loss(entry_price, atr, 2.0, position_type)
        risk = abs(entry_price - stop_loss)
        
        if position_type.lower() == "long":
            take_profit = entry_price + (risk * target_ratio)
        else:
            take_profit = entry_price - (risk * target_ratio)
        
        return {
            "entry": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk": risk,
            "reward": abs(take_profit - entry_price),
            "ratio": abs(take_profit - entry_price) / risk if risk > 0 else 0
        }

    def calculate_trailing_stop(
        self,
        current_price: float,
        highest_price: float,
        trailing_percent: float = 0.05,
        position_type: str = "long"
    ) -> float:
        """
        Calculate trailing stop loss
        
        Args:
            current_price: Current price
            highest_price: Highest price reached in trade
            trailing_percent: Trailing percentage (default 5%)
            position_type: "long" or "short"
            
        Returns:
            Updated stop loss level
        """
        if position_type.lower() == "long":
            trailing_stop = highest_price * (1 - trailing_percent)
        elif position_type.lower() == "short":
            trailing_stop = highest_price * (1 + trailing_percent)
        else:
            raise ValueError("Position type must be 'long' or 'short'")
        
        return trailing_stop

    # ==================== RISK METRICS ====================
    
    def calculate_sharpe_ratio(
        self,
        returns: Union[List[float], np.ndarray, pd.Series],
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate Sharpe Ratio
        Sharpe = (Return - Risk Free Rate) / Std Dev of Returns
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate (default 2%)
            
        Returns:
            Sharpe Ratio value
        """
        returns = np.array(returns)
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        
        if len(returns) < 2:
            return 0
        
        sharpe = (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252)
        return sharpe

    def calculate_sortino_ratio(
        self,
        returns: Union[List[float], np.ndarray, pd.Series],
        target_return: float = 0.0,
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate Sortino Ratio (downside deviation only)
        
        Args:
            returns: Series of returns
            target_return: Target return level
            risk_free_rate: Risk-free rate
            
        Returns:
            Sortino Ratio value
        """
        returns = np.array(returns)
        excess_returns = returns - target_return
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
        
        if downside_deviation == 0:
            return 0
        
        sortino = (np.mean(returns) - risk_free_rate) / downside_deviation * np.sqrt(252)
        return sortino

    def calculate_max_drawdown(
        self,
        returns: Union[List[float], np.ndarray, pd.Series]
    ) -> Tuple[float, float]:
        """
        Calculate maximum drawdown
        
        Args:
            returns: Series of returns or equity values
            
        Returns:
            Tuple of (max_drawdown_percentage, duration_in_periods)
        """
        returns = np.array(returns)
        
        # If returns are prices/equity values, calculate returns
        if np.all(returns > 0) and np.mean(returns) > 1:
            cumulative = np.cumprod(1 + returns)
        else:
            cumulative = np.cumprod(1 + returns)
        
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_dd = np.min(drawdown)
        
        return max_dd, np.sum(drawdown == max_dd)

    def calculate_calmar_ratio(
        self,
        returns: Union[List[float], np.ndarray, pd.Series],
        periods: int = 252
    ) -> float:
        """
        Calculate Calmar Ratio (Return / Maximum Drawdown)
        
        Args:
            returns: Series of returns
            periods: Periods per year (default 252 for daily)
            
        Returns:
            Calmar Ratio value
        """
        returns = np.array(returns)
        annual_return = np.mean(returns) * periods
        max_dd, _ = self.calculate_max_drawdown(returns)
        
        if max_dd == 0:
            return float('inf') if annual_return > 0 else 0
        
        calmar = annual_return / abs(max_dd)
        return calmar

    def calculate_value_at_risk(
        self,
        returns: Union[List[float], np.ndarray, pd.Series],
        confidence_level: float = 0.95,
        method: str = "historical"
    ) -> float:
        """
        Calculate Value at Risk (VaR)
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (default 95%)
            method: "historical" or "parametric"
            
        Returns:
            VaR value (negative = potential loss)
        """
        returns = np.array(returns)
        
        if method == "historical":
            var = np.percentile(returns, (1 - confidence_level) * 100)
        elif method == "parametric":
            mean = np.mean(returns)
            std = np.std(returns)
            z_score = np.abs(np.percentile(np.random.standard_normal(10000), (1 - confidence_level) * 100))
            var = mean - (std * z_score)
        else:
            raise ValueError("Method must be 'historical' or 'parametric'")
        
        return var

    def calculate_cvar(
        self,
        returns: Union[List[float], np.ndarray, pd.Series],
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall
        Average of losses beyond VaR
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level
            
        Returns:
            CVaR value
        """
        returns = np.array(returns)
        var = self.calculate_value_at_risk(returns, confidence_level)
        cvar = np.mean(returns[returns <= var])
        
        return cvar if not np.isnan(cvar) else var

    def calculate_win_rate(
        self,
        trades: List[Dict[str, float]]
    ) -> float:
        """
        Calculate win rate from trades
        
        Args:
            trades: List of trade dicts with 'pnl' key
            
        Returns:
            Win rate (0-1)
        """
        if not trades:
            return 0
        
        wins = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
        return wins / len(trades)

    def calculate_profit_factor(
        self,
        trades: List[Dict[str, float]]
    ) -> float:
        """
        Calculate profit factor (Gross Profit / Gross Loss)
        
        Args:
            trades: List of trade dicts with 'pnl' key
            
        Returns:
            Profit factor ratio
        """
        if not trades:
            return 0
        
        gross_profit = sum(max(0, trade.get('pnl', 0)) for trade in trades)
        gross_loss = abs(sum(min(0, trade.get('pnl', 0)) for trade in trades))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0
        
        return gross_profit / gross_loss

    def calculate_recovery_factor(
        self,
        trades: List[Dict[str, float]],
        initial_capital: float
    ) -> float:
        """
        Calculate recovery factor (Net Profit / Max Drawdown)
        
        Args:
            trades: List of trades
            initial_capital: Starting capital
            
        Returns:
            Recovery factor
        """
        if not trades:
            return 0
        
        total_pnl = sum(trade.get('pnl', 0) for trade in trades)
        
        # Calculate max drawdown from equity curve
        equity = [initial_capital]
        for trade in trades:
            equity.append(equity[-1] + trade.get('pnl', 0))
        
        equity = np.array(equity)
        running_max = np.maximum.accumulate(equity)
        max_dd = np.min((equity - running_max) / running_max)
        
        if max_dd == 0:
            return float('inf') if total_pnl > 0 else 0
        
        return total_pnl / abs(max_dd)

    def calculate_comprehensive_metrics(
        self,
        returns: Union[List[float], np.ndarray, pd.Series],
        trades: Optional[List[Dict[str, float]]] = None
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics
        
        Args:
            returns: Series of returns
            trades: List of trades with pnl
            
        Returns:
            RiskMetrics dataclass
        """
        trades = trades or []
        returns = np.array(returns)
        
        sharpe = self.calculate_sharpe_ratio(returns)
        sortino = self.calculate_sortino_ratio(returns)
        max_dd, _ = self.calculate_max_drawdown(returns)
        calmar = self.calculate_calmar_ratio(returns)
        var_95 = self.calculate_value_at_risk(returns, 0.95)
        var_99 = self.calculate_value_at_risk(returns, 0.99)
        cvar_95 = self.calculate_cvar(returns, 0.95)
        cvar_99 = self.calculate_cvar(returns, 0.99)
        win_rate = self.calculate_win_rate(trades)
        profit_factor = self.calculate_profit_factor(trades)
        recovery_factor = self.calculate_recovery_factor(trades, self.initial_capital)
        
        return RiskMetrics(
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            win_rate=win_rate,
            profit_factor=profit_factor,
            recovery_factor=recovery_factor
        )

    # ==================== PORTFOLIO RISK ASSESSMENT ====================
    
    def calculate_portfolio_var(
        self,
        positions: Dict[str, float],
        returns: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate portfolio Value at Risk
        
        Args:
            positions: Dict of position sizes {asset: size}
            returns: Dict of return series {asset: returns}
            weights: Dict of portfolio weights {asset: weight}
            confidence_level: Confidence level
            
        Returns:
            Portfolio VaR
        """
        if weights is None:
            total_value = sum(positions.values())
            weights = {k: v / total_value for k, v in positions.items()}
        
        portfolio_returns = np.zeros(len(next(iter(returns.values()))))
        
        for asset, asset_returns in returns.items():
            portfolio_returns += weights.get(asset, 0) * np.array(asset_returns)
        
        var = self.calculate_value_at_risk(portfolio_returns, confidence_level)
        return var

    def calculate_correlation_matrix(
        self,
        returns: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix between assets
        
        Args:
            returns: Dict of return series
            
        Returns:
            Correlation DataFrame
        """
        df = pd.DataFrame(returns)
        return df.corr()

    def calculate_diversification_ratio(
        self,
        positions: Dict[str, float],
        volatilities: Dict[str, float],
        correlation_matrix: pd.DataFrame
    ) -> float:
        """
        Calculate diversification ratio
        
        Args:
            positions: Position sizes
            volatilities: Individual asset volatilities
            correlation_matrix: Correlation matrix
            
        Returns:
            Diversification ratio
        """
        total_value = sum(positions.values())
        weights = {k: v / total_value for k, v in positions.items()}
        
        # Weighted average volatility
        weighted_vol = sum(
            weights.get(asset, 0) * volatilities.get(asset, 0)
            for asset in weights
        )
        
        # Portfolio volatility
        assets = list(weights.keys())
        w_array = np.array([weights.get(a, 0) for a in assets])
        vol_array = np.array([volatilities.get(a, 0) for a in assets])
        
        # Correlation indices
        corr_matrix = correlation_matrix.loc[assets, assets].values
        portfolio_vol = np.sqrt(w_array @ corr_matrix @ vol_array ** 2)
        
        if portfolio_vol == 0:
            return 0
        
        return weighted_vol / portfolio_vol

    def calculate_concentration_risk(
        self,
        positions: Dict[str, float]
    ) -> float:
        """
        Calculate concentration risk using Herfindahl index
        
        Args:
            positions: Position sizes
            
        Returns:
            Concentration risk (0-1, higher = more concentrated)
        """
        total = sum(positions.values())
        if total == 0:
            return 0
        
        weights = np.array([v / total for v in positions.values()])
        herfindahl = np.sum(weights ** 2)
        
        return herfindahl

    def calculate_marginal_var(
        self,
        positions: Dict[str, float],
        returns: Dict[str, np.ndarray],
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate marginal VaR for each position
        
        Args:
            positions: Position sizes
            returns: Return series
            confidence_level: Confidence level
            
        Returns:
            Dict of marginal VaR by asset
        """
        portfolio_var = self.calculate_portfolio_var(positions, returns, None, confidence_level)
        marginal_vars = {}
        
        small_change = 1
        total_value = sum(positions.values())
        
        for asset in positions:
            # Increase position slightly
            positions_plus = positions.copy()
            positions_plus[asset] += small_change
            
            var_plus = self.calculate_portfolio_var(
                positions_plus, returns, None, confidence_level
            )
            marginal_vars[asset] = (var_plus - portfolio_var) / small_change
        
        return marginal_vars

    def assess_portfolio_risk(
        self,
        positions: Dict[str, float],
        returns: Dict[str, np.ndarray],
        volatilities: Dict[str, float],
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> PortfolioRisk:
        """
        Comprehensive portfolio risk assessment
        
        Args:
            positions: Position sizes
            returns: Return series
            volatilities: Asset volatilities
            correlation_matrix: Correlation matrix
            
        Returns:
            PortfolioRisk dataclass
        """
        if correlation_matrix is None:
            correlation_matrix = self.calculate_correlation_matrix(returns)
        
        # Calculate metrics
        portfolio_var = self.calculate_portfolio_var(positions, returns)
        diversification_ratio = self.calculate_diversification_ratio(
            positions, volatilities, correlation_matrix
        )
        concentration_risk = self.calculate_concentration_risk(positions)
        marginal_vars = self.calculate_marginal_var(positions, returns)
        
        # Calculate average correlation
        corr_values = correlation_matrix.values[np.triu_indices_from(
            correlation_matrix.values, k=1
        )]
        avg_correlation = np.mean(corr_values) if len(corr_values) > 0 else 0
        
        # Component VaR
        total_value = sum(positions.values())
        weights = {k: v / total_value for k, v in positions.items()}
        component_vars = {
            asset: weights.get(asset, 0) * marginal_vars.get(asset, 0)
            for asset in positions
        }
        
        return PortfolioRisk(
            total_risk=portfolio_var,
            diversification_ratio=diversification_ratio,
            correlation_avg=avg_correlation,
            concentration_risk=concentration_risk,
            var_portfolio=portfolio_var,
            marginal_var=marginal_vars,
            component_var=component_vars
        )

    # ==================== UTILITY METHODS ====================
    
    def update_capital(self, new_capital: float):
        """Update current capital"""
        self.initial_capital = new_capital
        logger.info(f"Capital updated to: ${new_capital:,.2f}")

    def log_trade(self, trade: Dict[str, any]):
        """Log a completed trade"""
        self.trade_history.append({
            **trade,
            'timestamp': datetime.now()
        })
        logger.info(f"Trade logged: {trade}")

    def get_risk_summary(self) -> Dict[str, any]:
        """Get current risk summary"""
        if not self.trade_history:
            return {"message": "No trades recorded"}
        
        trades = self.trade_history
        returns = [t.get('pnl', 0) / self.initial_capital for t in trades]
        
        metrics = self.calculate_comprehensive_metrics(returns, trades)
        
        return {
            "total_trades": len(trades),
            "win_rate": metrics.win_rate,
            "profit_factor": metrics.profit_factor,
            "sharpe_ratio": metrics.sharpe_ratio,
            "max_drawdown": metrics.max_drawdown,
            "var_95": metrics.var_95,
            "recovery_factor": metrics.recovery_factor
        }


# ==================== HELPER FUNCTIONS ====================

def create_risk_manager(initial_capital: float, risk_per_trade: float = 0.02) -> RiskManager:
    """Factory function to create RiskManager instance"""
    return RiskManager(initial_capital, risk_per_trade)


def analyze_trade_performance(
    trades: List[Dict[str, float]]
) -> Dict[str, float]:
    """
    Analyze trade performance metrics
    
    Args:
        trades: List of trades with pnl
        
    Returns:
        Performance metrics dictionary
    """
    if not trades:
        return {}
    
    manager = RiskManager(1000)
    pnls = [t.get('pnl', 0) for t in trades]
    returns = [pnl / 1000 for pnl in pnls]
    
    return {
        "total_pnl": sum(pnls),
        "avg_pnl": np.mean(pnls),
        "win_rate": manager.calculate_win_rate(trades),
        "profit_factor": manager.calculate_profit_factor(trades),
        "sharpe_ratio": manager.calculate_sharpe_ratio(returns),
        "max_drawdown": manager.calculate_max_drawdown(returns)[0],
        "recovery_factor": manager.calculate_recovery_factor(trades, 1000)
    }


if __name__ == "__main__":
    # Example usage
    print("Risk Manager Module Loaded Successfully")
    
    # Initialize risk manager
    rm = RiskManager(initial_capital=10000, risk_per_trade=0.02)
    
    # Example: Calculate position size
    pos_size = rm.calculate_position_size_fixed_risk(
        entry_price=100,
        stop_loss_price=95,
        current_capital=10000
    )
    print(f"\nPosition Size: {pos_size.units:.2f} units")
    print(f"Position Cost: ${pos_size.cost:,.2f}")
    print(f"Risk Amount: ${pos_size.risk_amount:,.2f}")
    
    # Example: Calculate risk-reward levels
    levels = rm.calculate_risk_reward_levels(
        entry_price=100,
        atr=2.5,
        target_ratio=2.0,
        position_type="long"
    )
    print(f"\nRisk-Reward Levels:")
    print(f"Entry: ${levels['entry']:.2f}")
    print(f"Stop Loss: ${levels['stop_loss']:.2f}")
    print(f"Take Profit: ${levels['take_profit']:.2f}")
    print(f"Risk-Reward Ratio: {levels['ratio']:.2f}:1")
    
    # Example: Calculate risk metrics
    sample_returns = np.random.normal(0.001, 0.02, 252)
    metrics = rm.calculate_comprehensive_metrics(sample_returns)
    print(f"\nRisk Metrics:")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"VaR (95%): {metrics.var_95:.2%}")
