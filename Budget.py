#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Personal Budget Analysis Tool
============================
A tool for analyzing personal budget data from a PostgreSQL database,
calculating financial metrics, and visualizing spending patterns.
"""

# =============================================================================
# Import Libraries
# =============================================================================
import logging
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.getLogger("matplotlib").setLevel(logging.ERROR)

import os
import calendar
from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from dotenv import load_dotenv

import psycopg2
from psycopg2 import sql

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns

from config import (
    YEARS_TO_ANALYZE,
    TARGET_EXPENSE,
    MAX_EXPENSE,
    TARGET_DIFF,
    MAX_DIFF,
    SAVING_RATE_TARGET,
    EXPENSE_RATIO_TARGET,
    ASSIST_RATIO_TARGET,
    DISPOSABLE_INCOME_RATIO_TARGET,
    BURN_RATE_RATIO_TARGET,
    TABLES_TO_LOAD,
    EXPENSE_COLORS,
    INCOME_DIFF_COLORS,
    INCOME_LINE_COLOR,
    EXPENSE_BINS,
    EXPENSE_LABELS,
)

# Ensure charts directory exists
CHARTS_DIR = os.path.join(os.path.dirname(__file__), 'charts')
os.makedirs(CHARTS_DIR, exist_ok=True)

# =============================================================================
# Database Functions
# =============================================================================

def get_db_connection_params() -> Dict[str, str]:
    """
    Load database connection parameters from environment variables.
    
    Returns:
        Dict[str, str]: Dictionary containing database connection parameters
    """
    load_dotenv()
    return {
        "dbname": os.getenv("DB_NAME", "budget_db"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "admin"),
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432")
    }


def load_table_to_df(table_name: str, index_col: str, connection) -> Optional[DataFrame]:
    """
    Load a database table into a pandas DataFrame.
    
    Args:
        table_name: Name of the table to load
        index_col: Column to use as index
        connection: Database connection object
        
    Returns:
        DataFrame or None: The loaded table data or None if an error occurs
    """
    try:
        query = sql.SQL("SELECT * FROM {}").format(sql.Identifier(table_name))
        df = pd.read_sql_query(query.as_string(connection), connection, index_col=index_col)
        logger.info(f"Table '{table_name}' successfully loaded ({len(df)} rows).")
        return df
    except Exception as e:
        logger.error(f"Error loading table {table_name}: {e}")
        return None


def load_data_from_db() -> Dict[str, DataFrame]:
    """
    Connect to database and load all required tables into DataFrames.
    
    Returns:
        Dict[str, DataFrame]: Dictionary of DataFrames containing loaded tables
    """
    dfs = {}  # Dictionary to store loaded DataFrames
    db_params = get_db_connection_params()
    
    try:
        logger.info("Connecting to database...")
        with psycopg2.connect(**db_params) as conn:
            for table, index in TABLES_TO_LOAD.items():
                dfs[table] = load_table_to_df(table, index, conn)
        logger.info("All tables loaded successfully.")
    except psycopg2.Error as e:
        logger.error(f"Database connection error: {e}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        
    return dfs

# =============================================================================
# Data Processing Functions
# =============================================================================

def date_to_month_and_year(df: DataFrame, date_column: str) -> DataFrame:
    """
    Add year, month, month_name and year_month columns based on a date column.
    
    Args:
        df: Input DataFrame
        date_column: Name of the date column to process
        
    Returns:
        DataFrame: DataFrame with added date-related columns
    """
    df_copy = df.copy()
    
    # Convert date column to datetime and extract components
    date_col = pd.to_datetime(df_copy[date_column])
    
    df_copy['year'] = date_col.dt.year.astype(int)
    df_copy['month'] = date_col.dt.month.astype(int)
    
    # Add formatted month names and combined year-month
    df_copy['month_name'] = df_copy['month'].apply(lambda month: calendar.month_name[month])
    df_copy['year_month'] = df_copy['year'].astype(str).str[-2:] + '-' + df_copy['month_name'].str[:3]
    
    # Filter for selected years
    df_copy = df_copy[df_copy['year'].isin(YEARS_TO_ANALYZE)]
    
    return df_copy


def join_products_with_categories(df_products: DataFrame, df_categories: DataFrame, 
                                 df_subcategories: DataFrame) -> DataFrame:
    """
    Join product data with category and subcategory information.
    
    Args:
        df_products: Products DataFrame
        df_categories: Categories DataFrame
        df_subcategories: Subcategories DataFrame
        
    Returns:
        DataFrame: Products with joined category and subcategory data
    """
    df_copy = df_products.copy()
    
    # Join with categories
    df_with_categories = df_copy.merge(
        df_categories,
        on='category_id',
        how='left'
    ).fillna(-1)
    
    # Join with subcategories
    df_with_subcategories = df_with_categories.merge(
        df_subcategories,
        on='subcategory_id',
        how='left'
    )
    
    # Clean up column names
    df_cleaned = (
        df_with_subcategories
        .drop(columns='category_id_y')
        .rename(columns={
            'category_id_x': 'category_id'
        })
    )
    
    # Ensure correct data types
    df_cleaned = df_cleaned.astype({
        'category_id': int,
        'subcategory_id': int
    })
    
    return df_cleaned


def filter_products_by_category(df_products: DataFrame, df_categories: DataFrame,
                              df_subcategories: DataFrame,
                              categories: Optional[List[str]] = None,
                              subcategories: Optional[List[str]] = None,
                              products: Optional[List[str]] = None,
                              include_categories: bool = True,
                              include_subcategories: bool = True,
                              include_products: bool = True) -> pd.Series:
    """
    Filter product IDs based on categories, subcategories, and product names.
    
    Args:
        df_products: Products DataFrame
        df_categories: Categories DataFrame
        df_subcategories: Subcategories DataFrame
        categories: List of category names to filter by
        subcategories: List of subcategory names to filter by
        products: List of product names to filter by
        include_categories: If True, include specified categories; if False, exclude them
        include_subcategories: If True, include specified subcategories; if False, exclude them
        include_products: If True, include specified products; if False, exclude them
        
    Returns:
        Series: Series of filtered product IDs
    """
    df = join_products_with_categories(df_products, df_categories, df_subcategories).copy()
    
    if categories is not None:
        mask = df['category_name'].isin(categories)
        df = df[mask if include_categories else ~mask]
    
    if subcategories is not None:
        mask = df['subcategory_name'].isin(subcategories)
        df = df[mask if include_subcategories else ~mask]
    
    if products is not None:
        mask = df['product_name'].isin(products)
        df = df[mask if include_products else ~mask]
    
    return df['product_id']


def filter_products_by_assist(df_expenses: DataFrame) -> DataFrame:
    """
    Filter expenses DataFrame to include only gift/assistance entries.
    
    Args:
        df_expenses: Expenses DataFrame
        
    Returns:
        DataFrame: Filtered DataFrame with only assistance/gift entries
    """
    df_copy = date_to_month_and_year(df_expenses, 'date').copy()
    
    # Filter for entries that are gifts or have assist value
    is_assist_mask = (df_copy['assist'] > 0) | (df_copy['is_gift'])
    
    df_copy = df_copy[is_assist_mask]
    
    # Calculate total assistance amount
    df_copy['assist'] = (
        df_copy['assist'].where(df_copy['assist'] > 0, 0) +
        df_copy['total_price'].where(df_copy['is_gift'], 0)
    )
    
    return df_copy


def get_df_grouped_by_month(df: DataFrame, groupby_column: str, 
                          new_column_name: str, agg_func: str = 'sum') -> DataFrame:
    """
    Group DataFrame by year and month, aggregating the specified column.
    
    Args:
        df: Input DataFrame
        groupby_column: Column to aggregate
        new_column_name: Name for the aggregated column
        agg_func: Aggregation function to apply (default: 'sum')
        
    Returns:
        DataFrame: Grouped DataFrame
    """
    df_copy = df.copy()
    
    if all(col in df_copy.columns for col in ['year', 'month']):
        return (
            df_copy
                .groupby(['year_month', 'year', 'month'])[groupby_column]
                .agg(agg_func)
                .reset_index(name=new_column_name)
                .sort_values(by=['year', 'month'])
                .reset_index(drop=True)
        )
    else:
        logger.warning('Required columns ["year", "month"] not found. Use date_to_month_and_year() first.')
        return df


def merge_all(dfs: List[DataFrame], on: List[str], how: str) -> DataFrame:
    """
    Merge multiple DataFrames on common columns.
    
    Args:
        dfs: List of DataFrames to merge
        on: List of column names to merge on
        how: Type of merge to perform ('inner', 'outer', etc.)
        
    Returns:
        DataFrame: Merged DataFrame
    """
    from functools import reduce
    
    df_copy = reduce(lambda left, right: pd.merge(left, right, on=on, how=how), dfs)
    df_copy = (
        df_copy
        .fillna(0)
        .sort_values(by=['year', 'month'])
        .reset_index(drop=True)
    )
    
    return df_copy


def prepare_data_for_analysis(dfs: Dict[str, DataFrame]) -> DataFrame:
    """
    Process and transform raw data into analysis-ready format.
    
    Args:
        dfs: Dictionary of raw DataFrames from database
        
    Returns:
        DataFrame: Processed and combined DataFrame ready for analysis
    """
    logger.info("Preparing data for analysis...")
    # Reset indices for easier processins
    df_expenses = dfs['expenses'].reset_index()
    df_income = dfs['income'].reset_index()
    df_balance = dfs['balance'].reset_index()
    df_products = dfs['products'].reset_index()
    df_categories = dfs['categories'].reset_index()
    df_subcategories = dfs['subcategories'].reset_index()
    
    # Process date columns and add year/month
    df_expenses_filtered = date_to_month_and_year(df_expenses, 'date')
    df_income_filtered = date_to_month_and_year(df_income, 'date')
    df_balance_filtered = date_to_month_and_year(df_balance, 'date')
    
    # Group data by month
    df_expenses_grouped = get_df_grouped_by_month(df_expenses_filtered, 'total_price', 'expenses')
    df_income_grouped = get_df_grouped_by_month(df_income_filtered, 'amount', 'income')
    df_balance_grouped = get_df_grouped_by_month(df_balance_filtered, 'balance', 'balance')
    
    # Process regular expenses
    regular_product_ids = filter_products_by_category(
        df_products, df_categories, df_subcategories, 
        categories=['Regular', 'Products']
    )
    df_regular_expenses = df_expenses[df_expenses['product_id'].isin(regular_product_ids)]
    df_regular_expenses = date_to_month_and_year(df_regular_expenses, 'date')
    df_regular_expenses_grouped = get_df_grouped_by_month(df_regular_expenses, 'total_price', 'regular_expenses')
    
    # Process gift and assistance data
    df_assist_filtered = filter_products_by_assist(df_expenses)
    df_assist_grouped = get_df_grouped_by_month(df_assist_filtered, 'assist', 'assist')
    
    # Process passive income data
    products_passive_income_ids = filter_products_by_category(
        df_products, df_categories, df_subcategories,
        categories=['Income'], products=['Salary'], include_products=False
    )
    df_passive_income = df_income_filtered.loc[df_income_filtered['product_id'].isin(products_passive_income_ids)]
    df_passive_income_grouped = get_df_grouped_by_month(df_passive_income, 'amount', 'passive_income')
    
    # Combine all processed data
    df_combined = merge_all(
        dfs=[
            df_expenses_grouped,
            df_regular_expenses_grouped,
            df_assist_grouped,
            df_income_grouped,
            df_passive_income_grouped,
            df_balance_grouped
        ],
        on=['year_month', 'year', 'month'],
        how='outer'
    )
    
    # Remove first row if income or expenses are 0
    if df_combined.iloc[0][['income', 'expenses']].eq(0).any():
        df_combined = df_combined.iloc[1:].reset_index(drop=True)
    
    logger.info("Data prepared for analysis.")
    return df_combined

# =============================================================================
# Metric Calculation Functions
# =============================================================================

def calculate_metrics(df_combined: DataFrame) -> Tuple[Dict[str, pd.Series], Dict[str, str], DataFrame]:
    """
    Calculate financial metrics based on combined data.
    
    Args:
        df_combined: Combined data DataFrame
        
    Returns:
        Tuple containing:
        - Dictionary of calculated metric Series
        - Dictionary of metric abbreviations
        - DataFrame with all metrics
    """
    logger.info("Calculating financial metrics...")
    # Calculate core metrics
    metrics = {
        'saving_rate': (df_combined['income'] - df_combined['expenses']) / df_combined['income'] * 100,
        'disposable_income_rate': (df_combined['income'] - df_combined['regular_expenses']) / df_combined['income'] * 100,
        'expenses_ratio': df_combined['expenses'] / df_combined['income'] * 100,
        'assist_ratio': df_combined['assist'] / df_combined['income'] * 100,
        'burn_ratio': df_combined['expenses'] / df_combined['balance'] * 100,
        'financial_independence_ratio_target': df_combined['passive_income'] / TARGET_EXPENSE * 100,
        'financial_independence_ratio_real': df_combined['passive_income'] / df_combined['regular_expenses'] * 100
    }
    
    # Define abbreviations for metrics
    metrics_abbr = {
        'saving_rate': 'SR',
        'disposable_income_rate': 'DIR',
        'expenses_ratio': 'ER',
        'assist_ratio': 'AR',
        'burn_ratio': 'BR',
        'financial_independence_ratio_target': 'FIR_Target',
        'financial_independence_ratio_real': 'FIR_Real'
    }
    
    # Create metrics DataFrame
    df_rates = pd.DataFrame({
        'Date': df_combined['year_month'],
        'Income': df_combined['income'],
        'Passive Income': df_combined['passive_income'],
        'Balance': df_combined['balance'],
        'Expenses': df_combined['expenses'],
        'Regular Expenses': df_combined['regular_expenses'],
        'Assist': df_combined['assist']
    })
    
    # Add metrics, medians, and moving averages
    for name, series in metrics.items():
        abbr = metrics_abbr[name]
        
        df_rates[abbr] = series
        df_rates[f'{abbr}_MED'] = series.median()
        df_rates[f'{abbr}_MA'] = series.rolling(window=3, min_periods=1).mean()
    logger.info("Metrics calculated.")
    return metrics, metrics_abbr, df_rates

def calculate_income_stats(dfs: Dict[str, DataFrame], years_to_analyze: List[int]) -> DataFrame:
    """
    Calculate income, salary and expenses statistics.
    
    Args:
        dfs: Dictionary containing DataFrames with financial data
        years_to_analyze: List of years to include in analysis
        
    Returns:
        DataFrame: Combined statistics for income, salary and expenses
    """
    # Prepare data
    products = dfs['products'].reset_index()
    categories = dfs['categories'].reset_index()
    subcategories = dfs['subcategories'].reset_index()

    # Get product IDs by category
    product_ids = filter_products_by_category(products, categories, subcategories, categories=['Income'])
    salary_ids = filter_products_by_category(products, categories, subcategories, categories=['Income'], products=['Salary'])

    # Process dates
    income_df = date_to_month_and_year(dfs['income'], 'date')
    expenses_df = date_to_month_and_year(dfs['expenses'], 'date')

    # Group salary data
    df_salary = (
        income_df[income_df['product_id'].isin(salary_ids)]
        .groupby(['year', 'month'], as_index=False)['amount']
        .sum()
        .rename(columns={'amount': 'salary'})
    )

    # Group total income data
    df_income = (
        income_df[income_df['product_id'].isin(product_ids)]
        .groupby(['year', 'month'], as_index=False)['amount']
        .sum()
        .rename(columns={'amount': 'income'})
    )

    # Group expenses data
    df_expenses = (
        expenses_df.reset_index()
        .groupby(['year', 'month'], as_index=False)['total_price']
        .sum()
        .rename(columns={'total_price': 'expenses'})
    )

    # Merge all data
    df_merged = (
        df_income
        .merge(df_salary, on=['year', 'month'], how='left')
        .merge(df_expenses, on=['year', 'month'], how='left')
    )
    
    # Filter by years
    df_merged = df_merged[df_merged['year'].isin(years_to_analyze)]
    
    return df_merged

# =============================================================================
# Visualization Functions
# =============================================================================

def get_expenses_color(value: float, max_expense: float = MAX_EXPENSE, 
                     target_expense: float = TARGET_EXPENSE) -> tuple:
    """
    Determine the color for expenses based on their value.
    
    Args:
        value: Expense value
        max_expense: Maximum acceptable expense threshold
        target_expense: Target expense threshold
        
    Returns:
        tuple: RGBA color tuple
    """
    if value >= max_expense:
        return EXPENSE_COLORS['high']
    elif value >= target_expense:
        return EXPENSE_COLORS['medium']
    return EXPENSE_COLORS['low']


def get_income_diff_color(income: float, expense: float, 
                        max_diff: float = MAX_DIFF, 
                        target_diff: float = TARGET_DIFF) -> tuple:
    """
    Determine the color for income-expense difference.
    
    Args:
        income: Income value
        expense: Expense value
        max_diff: Maximum desired difference
        target_diff: Target difference
        
    Returns:
        tuple: RGBA color tuple
    """
    diff = income - expense
    if diff >= max_diff:
        return INCOME_DIFF_COLORS['high']
    elif diff >= target_diff:
        return INCOME_DIFF_COLORS['medium']
    return INCOME_DIFF_COLORS['low']

def format_k(num, digits:int = 0):
    """
    Форматирует число в тысячах с суффиксом 'k'
    digits - количество знаков после запятой
    """
    num_thousands = round(num / 1000, digits)
    
    # Если число целое, убираем десятичную часть
    if digits == 0:
        return f"{int(num_thousands)}k"
    else:
        return f"{num_thousands}k"


def plot_expenses_income(df_combined: DataFrame) -> None:
    """
    Create a plot showing expenses and income over time and save it to charts directory.
    
    Args:
        df_combined: Combined data DataFrame
    """
    logger.info("Plotting expenses and income...")
    # Set style and prepare data
    sns.set_style("whitegrid")
    
    # Categorize expenses
    df_combined['expenses_category'] = pd.cut(
        df_combined['expenses'], 
        bins=EXPENSE_BINS, 
        labels=EXPENSE_LABELS, 
        right=False
    )
    
    # Create color maps
    expense_color_map = dict(zip(
        EXPENSE_LABELS, 
        [EXPENSE_COLORS['low'], EXPENSE_COLORS['medium'], EXPENSE_COLORS['high']]
    ))
    
    # Calculate colors for income-expense differences
    income_diff_colors = [
        get_income_diff_color(income, expense)
        for income, expense in zip(df_combined['income'], df_combined['expenses'])
    ]
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(18, 6))
    
    # Plot expenses as bars
    expense_barplot = sns.barplot(
        x='year_month',
        y='expenses',
        data=df_combined,
        hue='expenses_category',
        hue_order=EXPENSE_LABELS,
        palette=expense_color_map,
        alpha=0.8,
        legend=False
    )
    # Add expense values on top of bars
    for column in expense_barplot.patches:
        height = column.get_height()
        ax.text(
            column.get_x() + column.get_width() / 2,
            height + 1,
            f'{height:.0f}',
            ha='center',
            va='bottom',
            fontsize=9,
            color='black'
        )
    
    # Plot income as line
    income_line = sns.lineplot(
        x='year_month',
        y='income',
        data=df_combined,
        color=INCOME_LINE_COLOR,
        linewidth=2,
        marker='o',
        alpha=0.7,
        zorder=2,
        legend=False
    )
    # Add income points with colors based on income-expense difference
    income_scatter = ax.scatter(
        x=df_combined['year_month'],
        y=df_combined['income'],
        c=income_diff_colors,
        s=60,
        zorder=3
    )
    # Add income values above points
    for i, row in df_combined.iterrows():
        salary = row['income'] - row['passive_income']
        salary = format_k(salary)

        ax.text(
            x=row['year_month'],
            y=row['income'] + 10000,
            s=f'{row["income"]:.0f} ({salary})',
            ha='center',
            va='center',
            fontsize=9,
            color=income_diff_colors[i],
            bbox=dict(
                    facecolor='white', 
                    alpha=0.6, 
                    edgecolor='none', 
                    boxstyle='round, pad=0.1, rounding_size=1'
                ),
            zorder=999
        )
    
    # Plot salary line (income minus passive income)
    salary_line = sns.lineplot(
        x='year_month',
        y=df_combined['income']-df_combined['passive_income'],
        data=df_combined,
        color=mcolors.to_rgba('#90D5FF', alpha=1.0),
        linewidth=1.35,
        linestyle='--',
        alpha=0.70,
        legend=False
    )

    # Plot assist as line
    assist_line = sns.lineplot(
        x='year_month',
        y='assist',
        data=df_combined,
        color=mcolors.to_rgba('#A66FB5', alpha=1.0),
        linewidth=1.35,
        linestyle=':',
        alpha=0.70,
        legend=False
    )
    # Add assist value above assist line
    for i, row in df_combined.iterrows():
        if row['assist'] > 5000:
            assist = row['assist']
            assist = format_k(assist, 1)
            ax.text(
                x=row['year_month'],
                y=row['assist'] + 0.5,
                s=f"{assist}",
                ha='center',
                va='center',
                fontsize=9,
                color='#A66FB5',
                bbox=dict(
                    facecolor='white', 
                    alpha=0.6, 
                    edgecolor='none', 
                    boxstyle='round, pad=0.2'
                ),
                zorder=10
            )
    
    # Add threshold lines
    ax.axhline(
        y=TARGET_EXPENSE, 
        color=expense_color_map['medium'], 
        linestyle=':', 
        linewidth=2, 
        alpha=0.7, 
        label=f'Target Exp ({TARGET_EXPENSE})'
    )
    
    ax.axhline(
        y=MAX_EXPENSE, 
        color=expense_color_map['high'], 
        linestyle=':', 
        linewidth=2, 
        alpha=0.7, 
        label=f'Max Exp ({MAX_EXPENSE})'
    )
    
    # Set chart title and labels
    ax.set_title('Total Expenses and Income')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylabel('')
    
    # Create legend elements
    legend_elements = [
        # Bars
        mpatches.Patch(color=expense_color_map['low'], label=f'Expenses < {TARGET_EXPENSE}'),
        mpatches.Patch(color=expense_color_map['medium'], label=f'Expenses [{TARGET_EXPENSE} - {MAX_EXPENSE})'),
        mpatches.Patch(color=expense_color_map['high'], label=f'Expenses ≥ {MAX_EXPENSE}'),
        # Line
        mlines.Line2D([], [], color=INCOME_LINE_COLOR, marker='o', linestyle='-', markersize=7, label='Total Income'),
        # Dots
        mlines.Line2D([], [], color='none', marker='o', markerfacecolor=INCOME_DIFF_COLORS['high'], markersize=7, 
                     label=f'Income Diff ≥ {MAX_DIFF}'),
        mlines.Line2D([], [], color='none', marker='o', markerfacecolor=INCOME_DIFF_COLORS['medium'], markersize=7, 
                     label=f'Income Diff [{TARGET_DIFF} - {MAX_DIFF})'),
        mlines.Line2D([], [], color='none', marker='o', markerfacecolor=INCOME_DIFF_COLORS['low'], markersize=7, 
                     label=f'Income Diff < {TARGET_DIFF}'),
        # Expense target lines
        mlines.Line2D([], [], color=expense_color_map['high'], linestyle=':', linewidth=2, 
                     label=f'Max Expenses ({MAX_EXPENSE})'),
        mlines.Line2D([], [], color=expense_color_map['medium'], linestyle=':', linewidth=2, 
                     label=f'Target Expenses ({TARGET_EXPENSE})')
    ]
    
    # Add legend
    ax.legend(
        handles=legend_elements,
        title='Legend',
        bbox_to_anchor=(1.02, 0.5),
        loc='center left',
        frameon=True,
        fontsize=10,
        title_fontsize=10
    )
    
    # Final styling
    sns.despine()
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.xaxis.grid(False)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # Save the plot
    output_path = os.path.join(CHARTS_DIR, 'expenses_income.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Expenses and income plot saved to {output_path}")


def plot_metrics(df_rates: DataFrame) -> None:
    """
    Create a multi-panel plot showing financial metrics and save it to charts directory.
    
    Args:
        df_rates: DataFrame containing calculated metrics
    """
    logger.info("Plotting metrics...")
    # Target lines for each metric
    TARGET_LINES = {
        'Saving Rate': SAVING_RATE_TARGET,
        'Burn Ratio': BURN_RATE_RATIO_TARGET,
        'Expense Ratio': EXPENSE_RATIO_TARGET,
        'Assist Ratio': ASSIST_RATIO_TARGET,
        'Disposable Income Rate': DISPOSABLE_INCOME_RATIO_TARGET,
        'Financial Independence Ratio': None  # No target line for FIR
    }
    
    # Descriptive subtitles for each metric
    metric_subtitles = {
        'Saving Rate': 'Доля доходов, отложенных в накопления (↑)',
        'Burn Ratio': 'Отношение расходов к сумме всех накоплений (↓)',
        'Expense Ratio': 'Доля расходов от дохода за месяц (↓)',
        'Financial Independence Ratio': 'Отношение пассивного дохода к расходу (↑)',
        'Assist Ratio': 'Отношение расходов на помощь другим к сумме дохода (↓)',
        'Disposable Income Rate': 'Доля дохода после обязательных расходов (↑)'
    }
    
    # Metrics to plot in each panel
    metric_groups = {
        'Saving Rate': ['SR', 'SR_MA'],
        'Expense Ratio': ['ER', 'ER_MA'],
        'Disposable Income Rate': ['DIR', 'DIR_MA'],
        'Assist Ratio': ['AR', 'AR_MA'],
        'Financial Independence Ratio': ['FIR_Target', 'FIR_Target_MA', 'FIR_Real'],
        'Burn Ratio': ['BR', 'BR_MA']
    }
    
    def extend_to_length(lst: list, length: int) -> list:
        """Extend a list to a specific length by repeating elements."""
        return lst * (length // len(lst)) + lst[:length % len(lst)]
    
    def plot_metrics_group(ax, metrics: list, title: str, subtitle: str, 
                          markers: Optional[list] = None, 
                          linewidths: Optional[list] = None, 
                          alphas: Optional[list] = None, 
                          show_x_labels: bool = True) -> None:
        """Plot a group of related metrics on the given axis."""
        markers = extend_to_length(markers or ['o'], len(metrics))
        linewidths = extend_to_length(linewidths or [1], len(metrics))
        alphas = extend_to_length(alphas or [1], len(metrics))

        df_rates_filtered = df_rates.copy()
        df_rates_filtered.loc[df_rates_filtered['Expenses'] == 0] = np.nan
        df_rates_filtered = df_rates_filtered.dropna()
        df_rates_filtered = df_rates_filtered[df_rates_filtered['Income'] - df_rates_filtered['Passive Income'] > 0]
        
        # Collect all valid values to determine y-axis limits
        all_values = []
        for metric in metrics:
            cleaned_values = df_rates_filtered[metric].dropna()
            cleaned_values = cleaned_values[np.isfinite(cleaned_values)]
            all_values.extend(cleaned_values.tolist())
        
        if not all_values:
            raise ValueError(f"No valid data to plot for: {title}")
        
        # Calculate y-axis limits
        y_min = min(0, min(all_values) - 10) if min(all_values) < 0 else 0
        y_max = max(100, max(all_values))
        
        # Generate colors for each metric
        colors = sns.color_palette("husl", len(metrics))
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            ax.plot(
                df_rates_filtered['Date'],
                df_rates_filtered[metric],
                marker=markers[i],
                label=metric,
                linewidth=linewidths[i],
                alpha=alphas[i],
                color=colors[i],
                markersize=5,
                markerfacecolor=colors[i],
                markeredgewidth=1
            )
        
        # Add target line if applicable
        if title in TARGET_LINES and TARGET_LINES[title] is not None:
            ax.axhline(
                y=TARGET_LINES[title],
                color='gold',
                linestyle='--',
                label=f'TARGET_{title.replace(" ", "_")}',
                linewidth=2,
                alpha=0.65
            )
        
        # Style the plot
        ax.set_title(title, fontsize=14, fontweight='bold', color='#333', pad=20)
        ax.tick_params(axis='x', rotation=45, labelsize=10, colors='#666')
        ax.tick_params(axis='y', labelsize=10, colors='#666')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='gray')
        ax.margins(x=0.05, y=0.05)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(range(int(y_min), int(y_max) + 1, 10))
        
        # Hide x-axis labels for top row plots
        if not show_x_labels:
            ax.set_xticklabels([])
        
        # Add subtitle
        ax.text(0.5, 1.09, subtitle, transform=ax.transAxes, ha='center', va='top', fontsize=10, color='#555')
    
    # Create figure with subplots
    fig, axs = plt.subplots(3, 2, figsize=(18, 10))
    fig.suptitle('Target Metrics', fontsize=16, fontweight='bold', color='#222')
    
    # Plot each metric group
    for i, (title, metrics) in enumerate(metric_groups.items()):
        show_x_labels = (i >= 4)  # Show x-labels only for bottom row
        plot_metrics_group(
            axs.flatten()[i],
            metrics,
            title,
            subtitle=metric_subtitles[title],
            markers=['o', '', 'o'],
            linewidths=[1.2, 0.8, 0.6],
            alphas=[1, 0.8, 0.4],
            show_x_labels=show_x_labels
        )
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    output_path = os.path.join(CHARTS_DIR, 'metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Metrics plot saved to {output_path}")

def plot_income_stats(df_merged: DataFrame) -> None:
    """
    Create boxplots showing income, salary and expenses statistics.
    
    Args:
        df_merged: DataFrame containing merged financial data
    """
    logger.info("Plotting income statistics...")
    
    # Create figure
    fig, axs = plt.subplots(3, 1, figsize=(18, 6))
    sns.set_palette('Set2')

    # Common boxplot parameters
    boxplot_params = {
        'width': 0.4,
        'medianprops': {'color': 'red', 'linewidth': 2},
        'flierprops': {'marker': 'o', 'markersize': 7, 'markerfacecolor': 'darkred', 'markeredgecolor': 'black'},
        'showmeans': True,
        'meanprops': {'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 6},
        'boxprops': dict(facecolor='#6C9DC8', edgecolor='#3B6B8C'),
        'legend': False
    }

    # Plot income boxplot
    sns.boxplot(x='income', data=df_merged, ax=axs[0], **boxplot_params)
    axs[0].set_title('Income', fontsize=18, fontweight='bold', color='darkblue')
    
    # Plot salary boxplot
    sns.boxplot(x='salary', data=df_merged, ax=axs[1], **boxplot_params)
    axs[1].set_title('Salary', fontsize=18, fontweight='bold', color='darkblue')
    
    # Plot expenses boxplot
    sns.boxplot(x='expenses', data=df_merged, ax=axs[2], **boxplot_params)
    axs[2].set_title('Expenses', fontsize=18, fontweight='bold', color='darkblue')

    # Style all subplots
    max_value = int(df_merged['income'].max()) + 10000
    for ax in axs:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', labelsize=14)
        ax.set_xticks(range(0, max_value, 10000))
        ax.set(xlabel='')

    # Show x-labels only for bottom plot
    axs[2].set_xticklabels(range(0, max_value, 10000))

    # Add overall styling
    plt.tight_layout()

    # Save plot
    output_path = os.path.join(CHARTS_DIR, 'income_stats.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print("\nFinancial Statistics:")
    print(df_merged[['salary', 'income', 'expenses']].describe().iloc[1:])
    
    # Calculate and print yearly aggregates
    df_filtered = df_merged[df_merged['year'] > min(df_merged['year'])]
    agg_df = df_filtered.groupby('year')[['income', 'salary', 'expenses']].agg(['mean', 'median']).fillna(0).astype(int)
    
    # Add total row
    total_row = pd.DataFrame({
        ('income', 'mean'): [agg_df[('income', 'mean')].mean()],
        ('salary', 'mean'): [agg_df[('salary', 'mean')].mean()],
        ('expenses', 'mean'): [df_filtered['expenses'].mean()],
        ('income', 'median'): [agg_df[('income', 'median')].median()],
        ('salary', 'median'): [agg_df[('salary', 'median')].median()],
        ('expenses', 'median'): [df_filtered['expenses'].median()]
    }, index=['total'])
    
    print("\nYearly Aggregates:")
    print(pd.concat([agg_df, total_row]).astype(int), "\n")
    
    logger.info(f"Income statistics plot saved to {output_path}")

# =============================================================================
# Main Execution
# =============================================================================

def get_data_for_analysis():
    dfs = load_data_from_db()
    df_combined = prepare_data_for_analysis(dfs)
    _, _, df_rates = calculate_metrics(df_combined)
    return dfs, df_combined, df_rates

def main():
    """Main execution function"""
    logger.info("Starting main execution...")
    # Load data and prepare for analysis
    dfs, df_combined, df_rates = get_data_for_analysis()
    
    # Create visualizations
    plot_expenses_income(df_combined)
    plot_metrics(df_rates)
    
    # Create income statistics visualization
    df_income_stats = calculate_income_stats(dfs, YEARS_TO_ANALYZE)
    plot_income_stats(df_income_stats)

    logger.info("Main execution complete.")
    return dfs, df_combined, df_rates

if __name__ == "__main__":
    dfs, df_combined, df_rates = main()