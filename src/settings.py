from datetime import datetime

RUN = str(datetime.now()).replace(" ", "_").replace(":", "-").replace(".", "_")
PROFIT_TARGET_MULTIPLIERS = [1.25, 2, 2.75]
STOP_LOSS_MULTIPLIERS = [0.5, 1, 2]
OPENING_RANGE_MINUTES_LIST = [10, 20]
MARKET_OPEN_TIMES = ['09:15:00','09:30:00']
CRITERIA_SHARPE_THRESHOLD = 0.4
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
NSE_AVAILABLE_EQUITY_LIST_PATH = DATA_DIR + "processed-data/NSE-Equity-List(Available)_Z.csv"
NSE_EQUITY_RAW_DATA_1MIN_DIR = RAW_DATA_DIR + "raw-data-1minute/nse/equity"

# URL of the NSE India equity stock list
NSE_URL = "https://www.nseindia.com/market-data/securities-available-for-trading"
