#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Personal Budget Analysis Tool - Configuration
==========================================

This module contains configuration settings for the Personal Budget Analysis Tool.
It defines analysis parameters, financial targets, database configuration,
and visualization settings.

For security reasons, sensitive values should be moved to environment variables
in production. This config file serves as a template with example values.

Author: Dmitry Muzychuk
License: MIT
"""

from typing import Dict, List, Any
import os
from pathlib import Path
import matplotlib.colors as mcolors

# =============================================================================
# Environment Configuration
# =============================================================================
def get_env_int(key: str, default: int) -> int:
    """Get integer from environment variable with fallback to default."""
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default

def get_env_list(key: str, default: List[int]) -> List[int]:
    """Get list of integers from environment variable with fallback to default."""
    try:
        value = os.getenv(key)
        return [int(x) for x in value.split(',')] if value else default
    except (TypeError, ValueError):
        return default

# =============================================================================
# Analysis Parameters
# =============================================================================

YEARS_TO_ANALYZE: List[int] = get_env_list('BUDGET_YEARS', [2023, 2024, 2025])
TARGET_EXPENSE: int = get_env_int('BUDGET_TARGET_EXPENSE', 65000)
MAX_EXPENSE: int = get_env_int('BUDGET_MAX_EXPENSE', 70000)
TARGET_DIFF: int = get_env_int('BUDGET_TARGET_DIFF', 30000)
MAX_DIFF: int = get_env_int('BUDGET_MAX_DIFF', 50000)
TARGET_PASSIVE_INCOME: int = get_env_int('BUDGET_TARGET_PASSIVE_INCOME', 100000)


# =============================================================================
# Financial Targets (percentages)
# =============================================================================
FINANCIAL_TARGETS: Dict[str, float] = {
    'SAVING_RATE': get_env_int('BUDGET_SAVING_RATE_TARGET', 30),        # % of income to save
    'EXPENSE_RATIO': get_env_int('BUDGET_EXPENSE_RATIO_TARGET', 70),    # % of income for expenses
    'ASSIST_RATIO': get_env_int('BUDGET_ASSIST_RATIO_TARGET', 10),      # % of income for assistance
    'DISPOSABLE_INCOME_RATIO': get_env_int('BUDGET_DISPOSABLE_RATIO_TARGET', 80),  # % discretionary income
    'BURN_RATE_RATIO': get_env_int('BUDGET_BURN_RATE_TARGET', 4),       # % monthly burn rate
}

# Export individual constants for backward compatibility
SAVING_RATE_TARGET = FINANCIAL_TARGETS['SAVING_RATE']
EXPENSE_RATIO_TARGET = FINANCIAL_TARGETS['EXPENSE_RATIO']
ASSIST_RATIO_TARGET = FINANCIAL_TARGETS['ASSIST_RATIO']
DISPOSABLE_INCOME_RATIO_TARGET = FINANCIAL_TARGETS['DISPOSABLE_INCOME_RATIO']
BURN_RATE_RATIO_TARGET = FINANCIAL_TARGETS['BURN_RATE_RATIO']

# =============================================================================
# Database Configuration
# =============================================================================
TABLES_TO_LOAD: Dict[str, str] = {
    'balance': 'id',                # Balance history table
    'categories': 'category_id',    # Expense/Income categories
    'companies': 'company_id',      # Transaction counterparties
    'expenses': 'id',               # Expense transactions
    'income': 'id',                 # Income transactions
    'payment_method': 'method_id',  # Payment methods
    'products': 'product_id',       # Products/Services
    'subcategories': 'subcategory_id'  # Subcategories for detailed classification
}

# =============================================================================
# Visualization Configuration
# =============================================================================
# Color schemes for different expense levels
EXPENSE_COLORS: Dict[str, tuple] = {
    'high': mcolors.to_rgba('#E45756', alpha=0.8),    # Red for high expenses
    'medium': mcolors.to_rgba('#FAC05E', alpha=0.8),  # Yellow for medium
    'low': mcolors.to_rgba('#69B34C', alpha=0.8),     # Green for low
}

# Color scheme for income-expense difference
INCOME_DIFF_COLORS: Dict[str, tuple] = {
    'high': mcolors.to_rgba('#69B34C', alpha=0.9),    # Green for high savings
    'medium': mcolors.to_rgba('#FAC05E', alpha=0.9),  # Yellow for medium
    'low': mcolors.to_rgba('#E45756', alpha=0.9),     # Red for low savings
}

# Main income line color
INCOME_LINE_COLOR: tuple = mcolors.to_rgba('#006400', alpha=1.0)

# Expense categorization bins and labels
EXPENSE_BINS: List[float] = [-float('inf'), TARGET_EXPENSE, MAX_EXPENSE, float('inf')]
EXPENSE_LABELS: List[str] = ['low', 'medium', 'high']

# =============================================================================
# File Paths
# =============================================================================
# Project root directory
PROJECT_ROOT: Path = Path(__file__).parent.resolve()

# Directory for saving charts
CHARTS_DIR: Path = PROJECT_ROOT / 'charts'
CHARTS_DIR.mkdir(exist_ok=True)

# Ensure the charts directory exists
os.makedirs(CHARTS_DIR, exist_ok=True) 