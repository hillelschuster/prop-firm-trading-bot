"""
Market Regime Classifier

Implements ADX-based market regime classification to enable intelligent strategy
activation based on market conditions. Designed per ADR-015 specifications.

Author: Prop Firm Trading Bot
Date: 2025-06-09
"""

import logging
import pandas as pd
import pandas_ta as ta
from typing import Optional


class MarketRegimeClassifier:
    """
    Classifies market regimes using Average Directional Index (ADX) analysis.
    
    Provides binary classification (Trending/Ranging) with ambiguous zone for
    risk management. Designed to enable selective strategy activation based on
    market suitability assessment.
    
    Classification Rules (per ADR-015):
    - ADX(14) > 25: "Trending" → Activates TrendFollowingSMA strategy
    - ADX(14) < 20: "Ranging" → Activates MeanReversionRSI strategy  
    - ADX(14) between 20-25: "Ambiguous" → No strategy activation
    """
    
    def __init__(self, adx_period: int = 14, trending_threshold: float = 25.0, 
                 ranging_threshold: float = 20.0):
        """
        Initialize MarketRegimeClassifier with configurable parameters.
        
        Args:
            adx_period: Period for ADX calculation (default: 14)
            trending_threshold: ADX threshold for trending market (default: 25.0)
            ranging_threshold: ADX threshold for ranging market (default: 20.0)
        """
        self.adx_period = adx_period
        self.trending_threshold = trending_threshold
        self.ranging_threshold = ranging_threshold
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Validate thresholds
        if ranging_threshold >= trending_threshold:
            raise ValueError(f"Ranging threshold ({ranging_threshold}) must be less than trending threshold ({trending_threshold})")
    
    def classify_market_regime(self, symbol: str, h4_data: pd.DataFrame) -> str:
        """
        Classify current market regime based on ADX analysis of H4 data.
        
        Args:
            symbol: Trading symbol for logging purposes
            h4_data: H4 timeframe OHLC data with minimum required periods
            
        Returns:
            str: Market regime classification ("Trending", "Ranging", or "Ambiguous")
            
        Raises:
            ValueError: If data is insufficient or missing required columns
            RuntimeError: If ADX calculation fails
        """
        try:
            # Validate input data
            self._validate_input_data(h4_data, symbol)
            
            # Calculate ADX
            adx_value = self._calculate_adx(h4_data)
            
            # Classify regime based on ADX value
            regime = self._classify_regime(adx_value)
            
            # Log classification result
            self.logger.info(
                f"Market regime classification for {symbol}: {regime} "
                f"(ADX: {adx_value:.2f}, Trending>{self.trending_threshold}, "
                f"Ranging<{self.ranging_threshold})"
            )
            
            return regime
            
        except Exception as e:
            self.logger.error(f"Failed to classify market regime for {symbol}: {str(e)}")
            # Return "Ambiguous" as safe fallback to prevent strategy activation
            return "Ambiguous"
    
    def _validate_input_data(self, data: pd.DataFrame, symbol: str) -> None:
        """
        Validate that input data contains required columns and sufficient periods.
        
        Args:
            data: H4 OHLC DataFrame to validate
            symbol: Symbol name for error messages
            
        Raises:
            ValueError: If data validation fails
        """
        required_columns = ['high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(
                f"Missing required columns for {symbol}: {missing_columns}. "
                f"Required: {required_columns}"
            )
        
        min_periods = self.adx_period * 2  # Need extra periods for smoothing
        if len(data) < min_periods:
            raise ValueError(
                f"Insufficient data for {symbol}: {len(data)} periods. "
                f"Minimum required: {min_periods} for ADX({self.adx_period}) calculation"
            )
        
        # Check for NaN values in recent data
        recent_data = data.tail(self.adx_period)
        if recent_data[required_columns].isnull().sum().sum() > 0:
            raise ValueError(f"NaN values found in recent H4 data for {symbol}")
    
    def _calculate_adx(self, data: pd.DataFrame) -> float:
        """
        Calculates the Average Directional Index (ADX) using the pandas-ta library
        for a robust and optimized implementation.
        
        Args:
            data: H4 OHLC DataFrame.
            
        Returns:
            float: The most recent ADX value.
            
        Raises:
            RuntimeError: If the ADX calculation fails or returns an invalid value.
        """
        try:
            # Use pandas-ta to calculate ADX. It returns a DataFrame with ADX, DMP, and DMN columns.
            adx_df = data.ta.adx(length=self.adx_period, append=False)
            
            if adx_df is None or adx_df.empty:
                raise RuntimeError("ADX calculation returned no data.")

            # The column name for ADX is typically 'ADX_d', where d is the period.
            adx_column_name = f"ADX_{self.adx_period}"
            if adx_column_name not in adx_df.columns:
                raise RuntimeError(f"Could not find expected ADX column '{adx_column_name}' in results.")

            # Get the last (most recent) ADX value
            current_adx = adx_df[adx_column_name].iloc[-1]

            if pd.isna(current_adx):
                raise RuntimeError("ADX calculation resulted in a NaN value.")
            
            return float(current_adx)
            
        except Exception as e:
            # Catch any exception from the library or our checks and wrap it.
            raise RuntimeError(f"ADX calculation failed: {e}")
    
    def _classify_regime(self, adx_value: float) -> str:
        """
        Classify market regime based on ADX value and configured thresholds.
        
        Args:
            adx_value: Current ADX value
            
        Returns:
            str: Market regime classification
        """
        if adx_value > self.trending_threshold:
            return "Trending"
        elif adx_value < self.ranging_threshold:
            return "Ranging"
        else:
            return "Ambiguous"
    
    def get_classification_info(self) -> dict:
        """
        Get current classifier configuration information.
        
        Returns:
            dict: Configuration parameters and thresholds
        """
        return {
            "adx_period": self.adx_period,
            "trending_threshold": self.trending_threshold,
            "ranging_threshold": self.ranging_threshold,
            "ambiguous_zone": f"{self.ranging_threshold}-{self.trending_threshold}"
        }


  
