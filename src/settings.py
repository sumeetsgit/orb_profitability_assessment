from datetime import datetime


RUN = str(datetime.now()).replace(" ", "_").replace(":", "-").replace(".", "_")
# OPENING_RANGE_MINUTES = 15  # Duration of the opening range in minutes (e.g., first 15 minutes)
# PROFIT_TARGET_MULTIPLIER = 2  # Multiplier for setting profit targets
# STOP_LOSS_MULTIPLIER = 1  # Multiplier for setting stop-loss levels
# MARKET_OPEN_TIME = '09:30:00'  # Market opening time

# PROFIT_TARGET_MULTIPLIERS = [1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
# STOP_LOSS_MULTIPLIERS = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
# OPENING_RANGE_MINUTES_LIST = [5, 10, 15, 20, 25, 30]
# MARKET_OPEN_TIMES = ['09:15:00', '09:20:00', '09:25:00', '09:30:00']

PROFIT_TARGET_MULTIPLIERS = [1.25, 2, 2.75]
STOP_LOSS_MULTIPLIERS = [0.5, 1, 2]
OPENING_RANGE_MINUTES_LIST = [10, 20]
MARKET_OPEN_TIMES = ['09:15:00','09:30:00']

CRITERIA_SHARPE_THRESHOLD = 1.0 #0.5
CRITERIA_MAX_DRAWDOWN = 0.1

RISK_FREE_RATE = 0.06 / 252



YEARS = ["2021", "2022"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

BASE_DIR = "C:/SUMEET/PERSONAL/WQU/WQU - Capstone Project/CODE/orb_profitability_assessment/"
DATA_DIR = BASE_DIR + "data/"

OUTPUT_DIR = DATA_DIR + "output/"
LOGS_DIR = DATA_DIR + "logs/"
RAW_DATA_DIR = DATA_DIR + "raw-data/"

NSE_EQUITY_LIST_PATH = RAW_DATA_DIR + "others/nse-equity-list/NSE-Equity-List.csv"
NSE_AVAILABLE_EQUITY_LIST_PATH = DATA_DIR + "processed-data/NSE-Equity-List(Available).csv"
NSE_EQUITY_RAW_DATA_1MIN_DIR = RAW_DATA_DIR + "raw-data-1minute/nse/equity"

# URL of the NSE India equity stock list
NSE_URL = "https://www.nseindia.com/market-data/securities-available-for-trading"

# ALPHAVANTAGE_API_KEY = "QVK8YT68MJS91TNS"
