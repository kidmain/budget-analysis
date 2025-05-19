# Personal Budget Analysis Tool

A comprehensive Python tool for analyzing personal budget data, tracking expenses, and visualizing financial metrics.

## Features

- Track and analyze income and expenses
- Calculate key financial metrics (saving rate, burn ratio, etc.)
- Generate detailed visualizations
- PostgreSQL database integration
- Configurable financial targets and thresholds

## Requirements

- Python 3.8+
- PostgreSQL database
- Required Python packages listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kidmain/budget-analysis.git
cd budget-analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up the database:
- Create a PostgreSQL database
- Run the SQL schema from `sql/schema.sql`
- Copy `.env.example` to `.env` and update with your settings

## Configuration

The tool can be configured through:
1. Environment variables
2. `.env` file
3. `config.py` settings

Key configuration options include:
- Analysis years
- Target expense thresholds
- Financial ratio targets
- Database connection settings

## Usage

Run the main analysis script:
```bash
python Budget.py
```

This will:
1. Load data from the database
2. Calculate financial metrics
3. Generate visualization charts in the `charts/` directory

## Generated Charts

The tool generates three types of visualizations:
1. `expenses_income.png` - Monthly expenses and income comparison
2. `metrics.png` - Financial metrics dashboard
3. `income_stats.png` - Income statistics and distributions

## License

MIT License - See LICENSE file for details

## Author:
Dmitry Muzychuk

## 🌟 Features

- **Data Analysis**
  - Monthly income and expense tracking
  - Salary vs total income analysis
  - Expense categorization and analysis
  - Financial ratio calculations
  - Passive income tracking

- **Visualizations**
  - Income vs Expenses trends
  - Financial metrics dashboard
  - Statistical boxplots
  - Category-wise expense breakdown

- **Financial Metrics**
  - Saving rate
  - Expense ratio
  - Burn rate
  - Disposable income ratio
  - Financial independence progress

## 🛠️ Tech Stack

- **Python 3.8+**
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Database**: PostgreSQL
- **Environment**: python-dotenv
- **Type Checking**: mypy

## 📋 Prerequisites

- Python 3.8 or higher
- PostgreSQL database
- pip (Python package manager)

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kidmain/budget-analysis.git
   cd budget-analysis
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up the database:
   ```bash
   # Create PostgreSQL database
   createdb budget_db
   
   # Import database schema
   psql budget_db < sql/schema.sql
   ```

5. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## 🔧 Configuration

The project uses environment variables for configuration. Key settings include:

- Analysis parameters (target expenses, income goals)
- Financial ratio targets
- Database connection details
- Visualization preferences

See `.env.example` for all available configuration options.

## 📊 Usage

1. Ensure your database is populated with financial data
2. Run the main analysis:
   ```bash
   python Budget.py
   ```
3. View generated charts in the `charts/` directory:
   - `expenses_income.png`: Monthly expense vs income trends
   - `metrics.png`: Financial metrics dashboard
   - `income_stats.png`: Statistical analysis of income/expenses

## 📈 Sample Visualizations

The tool generates three main types of visualizations:

1. **Expenses vs Income Trend**
   - Monthly comparison of income and expenses
   - Color-coded expense categories
   - Income difference indicators

2. **Financial Metrics Dashboard**
   - Saving rate trends
   - Expense ratio analysis
   - Financial independence progress

3. **Statistical Analysis**
   - Income distribution
   - Expense patterns
   - Salary trends

## 🗄️ Project Structure
```
budget-analysis/
├── sql/                    # Database related files
│   ├── schema.sql         # Database structure
│   └── Structure.png      # Database diagram
├── charts/                # Generated visualizations
├── .env.example          # Environment variables template
├── .gitignore           # Git ignore rules
├── LICENSE              # MIT License
├── README.md            # Project documentation
├── requirements.txt     # Python dependencies
├── Budget.py           # Main analysis script
└── config.py           # Configuration settings
```

## 📊 Database Structure

The project uses PostgreSQL 17.2 with the following key tables:

### Core Tables
- `balance` - Account balance tracking
- `expenses` - Expense transactions with detailed attributes
- `income` - Income records with categorization
- `products` - Products/services catalog

### Classification Tables
- `categories` - Main expense/income categories
- `subcategories` - Detailed subcategories
- `companies` - Transaction counterparties
- `payment_method` - Payment methods

### Key Features
- Automatic balance calculation
- Transaction timestamps and history
- Shared expenses tracking
- Gift and assistance tracking
- Category hierarchy

For detailed database structure, see [sql/schema.sql](sql/schema.sql).

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Dmitry Muzychuk**

- GitHub: [@kidmain](https://github.com/kidmain)

## 🙏 Acknowledgments

- Financial analysis best practices
- Python data science community
- Open source visualization libraries 