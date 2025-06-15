#!/usr/bin/env python3
"""
Generate comprehensive performance analysis report for the regime-filtered backtest.

This script analyzes the results from the multi-timeframe backtest with MarketRegimeClassifier
and compares it against the Phase 7 baseline performance.

Author: Prop Firm Trading Bot
Date: 2025-01-27
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime, timezone

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config_manager import load_and_validate_config
from src.backtesting.backtest_engine import BacktestEngine
from src.utils.performance_reporter import PerformanceReporter


def run_regime_filtered_backtest():
    """
    Run the regime-filtered backtest and extract performance data.
    """
    
    # Setup logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise for clean output
        format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
    )
    logger = logging.getLogger("PerformanceAnalysis")
    
    try:
        print("🔄 Running Regime-Filtered Backtest...")
        print("=" * 60)
        
        # 1. Load configuration
        config_dir = project_root / "config"
        config = load_and_validate_config(str(config_dir))
        
        # 2. Initialize BacktestEngine
        backtest_engine = BacktestEngine(config, logger)
        
        # 3. Run backtest for EURUSD_RSI_M15 strategy
        strategy_profile_key = "EURUSD_RSI_M15"
        m15_data_path = project_root / "data" / "EURUSD_M15_2023_05_01_to_2023_05_31.csv"
        
        if not m15_data_path.exists():
            print(f"❌ Data file not found: {m15_data_path}")
            return None
            
        # Run the backtest
        results = backtest_engine.run_single_strategy_backtest(
            strategy_profile_key=strategy_profile_key,
            csv_file_path=str(m15_data_path),
            initial_balance=10000.0
        )
        
        if not results:
            print("❌ Backtest failed to produce results")
            return None
            
        print("✅ Backtest completed successfully")
        return results, backtest_engine

    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        return None, None


def generate_performance_report(backtest_results, backtest_engine):
    """
    Generate comprehensive performance analysis report.
    """

    if not backtest_results or not backtest_results.success:
        print("❌ No backtest results to analyze")
        return

    # Extract performance data from BacktestResults and PaperTradingAdapter
    trade_history = backtest_engine.paper_adapter.trade_history if backtest_engine.paper_adapter else []
    equity_history = backtest_engine.paper_adapter.equity_history if backtest_engine.paper_adapter else []
    initial_balance = backtest_results.initial_balance
    final_balance = backtest_results.final_balance
    
    # Create PerformanceReporter
    reporter = PerformanceReporter(
        trade_history=trade_history,
        equity_history=equity_history,
        initial_balance=initial_balance,
        risk_free_rate=0.0
    )
    
    # Get all metrics
    metrics = reporter.get_all_metrics()
    
    print("\n" + "=" * 80)
    print("📊 DEFINITIVE PERFORMANCE ANALYSIS REPORT")
    print("🎯 Regime-Filtered Portfolio (MarketRegimeClassifier Active)")
    print("=" * 80)
    
    # PART 1: NEW PERFORMANCE METRICS
    print("\n📈 PART 1: NEW PERFORMANCE METRICS")
    print("-" * 50)
    
    net_pnl = metrics.get('total_net_profit', 0.0)
    total_return = ((final_balance - initial_balance) / initial_balance) * 100
    max_drawdown_currency = metrics.get('max_drawdown_currency', 0.0)
    max_drawdown_percentage = metrics.get('max_drawdown_percentage', 0.0)
    win_rate = metrics.get('win_rate_percentage', 0.0)
    total_trades = metrics.get('total_trades', 0)
    profit_factor = metrics.get('profit_factor', 0.0)
    
    print(f"💰 Net P&L:              ${net_pnl:.2f}")
    print(f"📊 Total Return:         {total_return:.2f}%")
    print(f"📉 Maximum Drawdown:     ${abs(max_drawdown_currency):.2f} ({max_drawdown_percentage:.2f}%)")
    print(f"🎯 Win Rate:             {win_rate:.1f}%")
    print(f"🔢 Total Trades:         {total_trades}")
    print(f"⚖️  Profit Factor:        {profit_factor:.2f}" if not pd.isna(profit_factor) else "⚖️  Profit Factor:        N/A")
    
    # Trade List
    print(f"\n📋 EXECUTED TRADES:")
    print("-" * 30)
    if trade_history:
        for i, trade in enumerate(trade_history, 1):
            pnl = getattr(trade, 'pnl', 0.0) or 0.0
            timestamp = getattr(trade, 'timestamp', 'N/A')
            symbol = getattr(trade, 'symbol', 'N/A')
            action = getattr(trade, 'action', 'N/A')
            price = getattr(trade, 'price', 0.0)
            print(f"  {i}. {timestamp} | {symbol} {action} @ {price:.5f} | P&L: ${pnl:.2f}")
    else:
        print("  No trades executed")
    
    # PART 2: COMPARATIVE ANALYSIS
    print(f"\n📊 PART 2: COMPARATIVE ANALYSIS")
    print("-" * 50)
    
    # Phase 7 baseline metrics (from previous analysis)
    baseline_net_pnl = 1.00
    baseline_max_drawdown = 69.04
    baseline_total_trades = 11
    baseline_win_rate = 9.0
    baseline_total_return = 0.01
    
    # Calculate changes
    pnl_change = net_pnl - baseline_net_pnl
    drawdown_change = abs(max_drawdown_currency) - baseline_max_drawdown
    trades_change = total_trades - baseline_total_trades
    win_rate_change = win_rate - baseline_win_rate
    return_change = total_return - baseline_total_return
    
    print("| Metric                | New Filtered Portfolio | Old Unfiltered Portfolio | Change        |")
    print("|----------------------|------------------------|---------------------------|---------------|")
    print(f"| Net P&L              | ${net_pnl:.2f}                | ~${baseline_net_pnl:.2f}                    | {pnl_change:+.2f}        |")
    print(f"| Total Return         | {total_return:.2f}%               | ~{baseline_total_return:.2f}%                   | {return_change:+.2f}%       |")
    print(f"| Max Drawdown         | ${abs(max_drawdown_currency):.2f}               | ~${baseline_max_drawdown:.2f}                  | {drawdown_change:+.2f}       |")
    print(f"| Total Trades         | {total_trades}                    | {baseline_total_trades}                         | {trades_change:+d}           |")
    print(f"| Win Rate             | {win_rate:.1f}%               | ~{baseline_win_rate:.1f}%                    | {win_rate_change:+.1f}%       |")
    
    # PART 3: FINAL ASSESSMENT
    print(f"\n🎯 PART 3: FINAL ASSESSMENT")
    print("-" * 50)
    
    # Determine if MarketRegimeClassifier improved performance
    improved_profitability = net_pnl > baseline_net_pnl
    reduced_risk = abs(max_drawdown_currency) < baseline_max_drawdown
    reduced_trades = total_trades < baseline_total_trades
    
    if improved_profitability and reduced_risk and reduced_trades:
        conclusion = "✅ YES - The MarketRegimeClassifier successfully improved the portfolio's net profitability while significantly reducing both risk (drawdown) and unnecessary trades."
    elif improved_profitability and reduced_trades:
        conclusion = "✅ PARTIALLY - The MarketRegimeClassifier improved profitability and reduced unnecessary trades, though risk metrics need further analysis."
    elif reduced_risk and reduced_trades:
        conclusion = "⚠️ MIXED - The MarketRegimeClassifier successfully reduced risk and unnecessary trades, but profitability improvements are marginal."
    else:
        conclusion = "❌ NO - The MarketRegimeClassifier did not achieve the expected improvements in profitability, risk reduction, and trade efficiency."
    
    print(f"🔍 CONCLUSION: {conclusion}")
    
    print("\n" + "=" * 80)
    print("📋 ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    # Import pandas for NaN checking
    import pandas as pd
    
    print("🚀 Starting Regime-Filtered Performance Analysis...")
    
    # Run backtest and generate report
    backtest_output = run_regime_filtered_backtest()
    if backtest_output and len(backtest_output) == 2:
        results, engine = backtest_output
        generate_performance_report(results, engine)
    else:
        print("❌ Failed to run backtest")
    
    print("\n✅ Performance analysis complete!")


  
