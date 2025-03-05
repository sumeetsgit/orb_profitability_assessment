
# ORB Profitability Assessment


## Installation & Setup

### Install required libraries from "requirements.txt"
    pip install --upgrade pip
    pip install -r requirements. txt    

### Data Requirements

    Create below directory strcture to store raw data.

```
|-data
    |-logs
    |-output
    |-processed-data
        |-NSE-Equity-List(Available).csv
    |-raw-data
        |-raw-data-1minute
            |-nse
                |-equity
                    |-2021
                        |-January_2021
                            |-.CNX100.csv
                        |-.....
                        |-December_2021
                    |-2022
                        |-January_2022
                        |-.....
                        |-December_2021
|-src
|-requirements.txt

```

    Download data at https://drive.google.com/drive/folders/1pjOkfxDE1zY9lpzkSW6ZLIcG57rkvYml`

    Ensure that the directory structure is maintained as defined in settings.py.

    Place the raw stock data CSV files in the directory specified by NSE_EQUITY_RAW_DATA_1MIN_DIR

    Ensure that the "data" folder follows appropriate folder structure & is included as a part of .gitignore file
    
    
### Run the main function 
    cd src
    python orb_main.py

### Run the sensitivity analysis
    cs src
    python sensitivity_analysis.py


## Project Structure
    settings.py:
    Contains configuration parameters (e.g., multipliers, market open times, paths, and criteria) used across the project.

    auxiliary_functions.py:
    Includes helper functions for logging, creating directories, data preprocessing, and file management.

    metrics.py:
    Provides functions to compute performance metrics such as net return, annualized return, Sharpe ratio, Omega ratio, and more.

    orb_main.py:
    The main script that integrates data loading, strategy application, backtesting, parameter optimization, and report generation.

    reports.py:
    Contains functions to create visual reports including line charts for cumulative returns and bar charts for performance metrics comparisons.

    sensitivity_analysis.py:
    Implements sensitivity analysis for the ORB strategy by evaluating performance across different volatility regimes and liquidity groups.



## Output

### Charts and Reports:
    The code generates multiple charts:

    Cumulative Returns Chart: Overlays cumulative returns for all strategies.

    Performance Metrics Comparison: Bar charts comparing risk, return, and ratio metrics.

    Sensitivity Analysis Charts (if any): Saved as separate CSV files and graphs.

    CSV Files: Aggregated metrics for all tickers are saved (e.g., all_tickers_metrics.csv, sensitivity_analysis_metrics.csv).

### Memory and Performance

The code is optimized to reduce memory usage by releasing temporary variables and calling garbage collection (gc.collect()) after plots are closed.
Plotting functions include options to save figures to disk and then close the figures to free up memory.


## Contact and Support
    For further questions or support, please contact Sumeet Shah / Wai Hang Lau (Zachary) at sumeetsworkplace@gmail.com