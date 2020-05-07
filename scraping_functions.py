import re
import time
import requests
import datetime
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def businessinsider_usd_history(start_date, end_date=datetime.date.today().strftime("%Y%m%d"), directory=""):
    html = scrape_businessinsider_usd_history(start_date, end_date)
    df = parse_businessinsider_usd_history(html)
    save_backup_json(df, directory, "dxy", start_date, end_date)
    return df

def coinmarketcap_history(coin, start_date, end_date=datetime.date.today().strftime("%Y%m%d"), directory=""):
    """A function that scrapes from CoinMarketCap and parses the results into a DataFrame. A backup of the data
    can also be exported to a json specified by setting a directory."""

    html = scrape_coinmarketcap_history(coin, start_date, end_date)
    df = parse_coinmarketcap_history(html)
    save_backup_json(df, directory, coin, start_date, end_date)
    return df

def parse_businessinsider_usd_history(html):
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.select('table.instruments tbody')[0]
    columns = np.array(list(map(lambda x: x.text.strip(), table.select('th'))))
    rows = np.array([list(map(lambda x: x.text.strip(), row.select("td"))) for row in table.select('tr:not(.header-row)')])
    return pd.DataFrame(rows, columns=columns)

def parse_coinmarketcap_history(html):
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.select('.cmc-table__table-wrapper-outer table')[2]
    columns = np.array(list(map(lambda x: x.text.replace("*", ""), table.select("thead th"))))
    rows = np.array([list(map(lambda x: x.text, row.select("td div"))) for row in table.select("tbody tr")])
    return pd.DataFrame(rows, columns=columns)

def scrape_businessinsider_usd_history(start_date, end_date):
    # Creates a Selenium connection object connected to a seperate Selenium container.
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    driver = webdriver.Remote(
        command_executor='http://hub:4444/wd/hub',
        desired_capabilities=DesiredCapabilities.CHROME)

    # Makes sure the driver waits for the page to load.
    driver.implicitly_wait(10)

    # Creates the custom url based on the start and end dates.
    sy, sm, sd = int(start_date[0:4]), int(start_date[4:6]), int(start_date[6:8])
    ey, em, ed = int(end_date[0:4]), int(end_date[4:6]), int(end_date[6:8])
    url = f"https://markets.businessinsider.com/index/historical-prices/us-dollar-index/{sd}.{sm}.{sy}_{ed}.{em}.{ey}"

    # Captures the html from the table
    driver.get(url)
    table = driver.find_element_by_css_selector("#historic-price-list .box")
    html = table.get_attribute('innerHTML')
    driver.quit()
    return html

def scrape_coinmarketcap_history(coin, start_date, end_date):
    url = f"https://coinmarketcap.com/currencies/{coin}/historical-data/?start={start_date}&end={end_date}"
    return requests.get(url).text


def save_backup_json(df, directory, title, start_date, end_date):
    if type(directory) == str:
        directory = re.sub(r'([^/])$', r'\1/', directory)
        df.to_csv(f"{directory}{title}_{start_date}-{end_date}.csv", index=False)

def wait_cycle(wait_min, wait_max):
    """Randomly chooses a wait time based on the min and max values and waits for that duration."""

    wait_time = np.random.uniform(wait_min, wait_max)
    time.sleep(wait_time)