from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
import time 

def getMarginRate(driver, name):
    """
    function to a margin fee interest rates given a driver already open on the page afer 
    having handled the cookies of the binance dedicated page 
    Inputs:
        -driver: Webdriver of selenium
        -name: str
                name of the asset
    Output:
        -rate: float
    """
    
    driver.find_element(By.XPATH, '//*[@id="__APP"]/div[2]/div/div/div[2]/div[1]/div[1]/div[2]/div[1]/input').click()
    path2 = '//*[@id="__APP"]/div[2]/div/div/div[2]/div[1]/div[1]/div[2]/div[2]/div/div/input'
    driver.find_element(By.XPATH,path2).send_keys(name)
    driver.find_element(By.XPATH,path2).send_keys(Keys.ENTER)
    path3 = '//*[@id="__APP"]/div[2]/div/div/div[2]/div[1]/button[1]'
    driver.find_element(By.XPATH,path3).click()
    time.sleep(0.5)
    path_wait =  '//*[@id="__APP"]/div[2]/div/div/div[2]/div[3]/div[1]/div/div/table/tbody/tr[1]/td[2]'
    while driver.find_element(By.XPATH,path_wait).text != name:
        pass
    path4 = '//*[@id="__APP"]/div[2]/div/div/div[2]/div[3]/div[1]/div/div/table/tbody/tr[1]/td[3]'
    rate = driver.find_element(By.XPATH,path4).text
    rate = float(rate.split("%")[0])/100
    return rate
  
  
def getDictionaryMarginRate(list_names):
    """
    function to retrieve a dictionary of margin fee interest rates
    Inputs:
        -list_name: List
                list of crypto names, such as ["ETH", "BTC"]
    Output:
        -rates: dict
                dictionary containing the rate as value and as key the name of the asset
    """
    options = Options()
    options.add_argument("--disable-notifications");
    options.add_argument("disable-infobars");
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    driver = webdriver.Chrome(options=options)
    url = "https://www.binance.com/en/margin/interest-history"
    driver.get(url)
    wait = WebDriverWait(driver, 15)
    wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="onetrust-accept-btn-handler"]')))
    driver.find_element(By.XPATH, '//*[@id="onetrust-accept-btn-handler"]').click()
    rates = dict(zip(list_names, [getMarginRate(driver, j) for j in list_names]))
    return rates
    
