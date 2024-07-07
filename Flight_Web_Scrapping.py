'''PROJETO DE MONITORAMENTO DE PREÇO DE PASSAGENS AÉREAS'''

import random
from datetime import datetime, timedelta
import time
from bs4 import BeautifulSoup
import requests
import demjson3
import pyodbc
import pywinauto
import traceback

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

conexaoDB=pyodbc.connect(
    DRIVER='ODBC Driver 17 for SQL Server',
    SERVER='EMERSONPC',
    DATABASE='DataScrapping',
    Trusted_Connection='yes') #Se desejar fazer a conexão com usuário e senha, basta adicionar os seguintes parâmetros: 'usuario=Emerson;' e 'senha=1234;'
cursor=conexaoDB.cursor()



date_start_scrapping=datetime(day=8, month=5, year=2024)
gap_date=[1,2,3,4,5,6,7,10,14,21,28]

date_today=datetime(day=23, month=3, year=2024)


'''
def recaptcha_solver(url):
    driver=webdriver.Edge()
    driver.get(url)
    WebDriverWait(driver, 20).until(EC.frame_to_be_available_and_switch_to_it((By.CSS_SELECTOR,"iframe[src^='https://www.google.com/recaptcha/enterprise/anchor']")))
    WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "div.recaptcha-checkbox-border"))).click()
    time.sleep(30)
'''
def recaptcha_solver():
    pywinauto.Application().start(r"C:/Users/emers/OneDrive/Área de Trabalho/macroscrapping.exe")
    time.sleep(30)
class DatesScrapping():
    def __init__(self, date_start_scrapping):
        self.flight_dates=[date_start_scrapping+timedelta(i) for i in gap_date if (date_start_scrapping+timedelta(i)) >= (datetime.today()-timedelta(1))]
        print(self.flight_dates)

        mother_URLs=[
                     'https://www.kayak.com.br/flights/GRU-VVI/?sort=bestflight_a&fs=stops=0',
                     'https://www.kayak.com.br/flights/GRU-EZE/?sort=bestflight_a&fs=stops=0',
                     'https://www.kayak.com.br/flights/GRU-SCL/?sort=bestflight_a&fs=stops=0',
                     ]

        self.child_URLs=[j[:41] + str(i.date()) + j[41:] for i in self.flight_dates for j in mother_URLs]
        print(self.child_URLs)
        random.shuffle(self.child_URLs)
        print(self.child_URLs)

        self.conexaoDB=pyodbc.connect(DRIVER='ODBC Driver 17 for SQL Server',
                                      SERVER='EMERSONPC',
                                      DATABASE='DataScrapping',
                                      Trusted_Connection='yes')
        self.cursor=conexaoDB.cursor()

    def Date_Scrapping(self):
        self.user_agent_list=["Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.74 Safari/537.36",
                              "Mozilla/5.0 (X11; Linux x86_64; rv:98.0) Gecko/20100101 Firefox/98.0",
                              "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36",
                              "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:98.0) Gecko/20100101 Firefox/98.0",
                              "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36 Edg/99.0.1150.30",
                              "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.83 Safari/537.36",
                              "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:98.0) Gecko/20100101 Firefox/98.0",
                              "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15"
                             ]


        for i in self.child_URLs:
            flight=i[33:36]+ '_'+ i[37:40]
            url = i
            print(url)
            headers = {
                        "cookie": f"Apache=KAQpuFApkRQAWhMPFSUggg{random.randint(1,100000)}-AAABjVXiG8Q-c4-Rk3D_g; kayak=WT9fne7UE81vSnWFCCCa{random.randint(1,100000)}; kmkid=Ayw_z3hsRb5gKnAZQ10fZBk; csid=5153d9e4-7c6e-4021-aaa7-3d8a5d4d79b8; cluster=4; p1.med.sid=R-4rh6H_G2AswSMdf3_Xnod-2UOOeL_VQizD1idB1b_bThBcsnZDZ0E2OhtZODylY; kayak.mc=AegHg-Ppk6Tna0kFL7TkEyA6uwSkIZwNCY3tTpwPp1m51ASP5_JuQ1NdwykL-EcU2SeD3pFvngbefN12nYuJFgZaBf913QiKp7MtQktWMWcCiqteAjUOQgZ7Qb_4zv9FxUOUDpDOqQiHt28PIZqhE62G0L4hgTpgMWVLYy6XyCuvYFGCqe9WdJPLHS0zySQdKd6I2fNSeToetPMNkUX8n4Wwsq_9PDrRnmqLdYe7gyuOZz7V-nIpR1shACNdNmBGgaGI3FLH2H7LwHau4kCyLi4ceN8b20VOupkacePfgLR-mItJn2JxC3xJlHE5uD9gM696VjKYSL_RyWMVr8w8C456ka-9sZzou2GBnlJFiy5RWC9CZgvudBJgXajY_M7rZNC01DxjvBTzztE6y8uR4cYEdk3BvmDUrPsbRnW3LVG_Ee-leqfZmfDPDEzstitUDYTB5Y0MEsqJbfycno84tNdf_EGEauS6UoDkyAp7N3TNm3geVtdgazF19cKIeuZYRA; mst_iBfK2g=Hn3p5ZB9CGvnU_HhRVxl3KAQHFpbT5Sif2S_TDOR4sN8_UHk6vKfQWmPV-xYyu_rvIAKKELoKMRG1W9nKnmuX5zTEoVLQ0ApwF74dpSyG58; mst_ADIrkw=2aZQ8-I0KrQ3k4flhAYqXaAQHFpbT5Sif2S_TDOR4sN8_UHk6vKfQWmPV-xYyu_rigHuDFb-lHUXKR3MHF35j56PBRZ67tI9cFm_su1QwfA",
                        "authority": "www.kayak.com.br",
                        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                        "accept-language": "pt-BR,pt;q=0.7",
                        "cache-control": "max-age=0",
                        "sec-ch-ua-mobile": "?0",
                        "sec-fetch-dest": "document",
                        "sec-fetch-mode": "navigate",
                        "sec-fetch-site": "none",
                        "sec-fetch-user": "?1",
                        "sec-gpc": "1",
                        "upgrade-insecure-requests": "1",
                        "user-agent": random.choice(self.user_agent_list),
                        "referer": "https://www.google.com/search?q=passagens+a%C3%A9reas+baratas&sca_esv=391f93f6c7aeb7e3&ei=rucFZpzcKL7d1sQPx4qS0Ac&udm=&ved=0ahUKEwic68PT-ZeFAxW-rpUCHUeFBHoQ4dUDCBA&uact=5&oq=passagens+a%C3%A9reas+baratas&gs_lp=Egxnd3Mtd2l6LXNlcnAiGXBhc3NhZ2VucyBhw6lyZWFzIGJhcmF0YXMyBhAAGAcYHjIGEAAYBxgeMgoQABiABBgKGLEDMgYQABgHGB4yBhAAGAcYHjIGEAAYBxgeMgYQABgHGB4yBhAAGAcYHjIGEAAYBxgeMgYQABgHGB5ItBBQsAhY3Q9wAXgBkAEAmAH0AaAB8AaqAQUwLjEuM7gBA8gBAPgBAZgCAqACnALCAgoQABhHGNYEGLADwgINEAAYgAQYigUYQxiwA8ICGRAuGIAEGIoFGEMYxwEY0QMYyAMYsAPYAQGYAwCIBgGQBhG6BgYIARABGAiSBwUxLjAuMaAHiR4&sclient=gws-wiz-serp"
                        }
            count=0
            a=0
            while True:
                count+=1
                print(count)
                try:
                    '''
                    host = 'brd.superproxy.io'
                    port = 22225
                    username = 'brd-customer-hl_2651a58c-zone-isp_proxy1'
                    password = '2y7ka7mnd6wa'
                    session_id = random.random()
                    proxy_url = ('http://{}-session-{}:{}@{}:{}'.format(username, session_id,password, host, port))
                    proxy=random.choice(proxies)
                    proxies = {'http': proxy, 'https': proxy}
                    '''
                    #Scrapping in HTML
                    print('Scrapping in HTML')
                    response = requests.request("GET", url, headers=headers, timeout=30)
                    time.sleep(5)
                    soup=BeautifulSoup(response.content, 'html.parser')
                    #print(response.text)
                    result_js=soup.findAll('script', type='text/javascript')
                    result_str=str(result_js)
                    if result_str.find('reducer:')!=-1:
                        element_posion1=result_str.find('reducer:')
                        element_posion2=result_str.find('brand: ["kayak"]')
                        result_str=result_str[(element_posion1+9):(element_posion2-3)]
                        result_dict=demjson3.decode(result_str)
                        initial_key='initialState'
                        print('Scrapping in HTML Successfully')
                    else:
                        '''
                        if count>=5:
                            recaptcha_solver()
                            count=0
                            continue
                        '''
                        if count>=8:
                            a+=1
                            if a>=2:
                                break
                            else:
                                #recaptcha_solver()
                                count=0
                                continue
                        #Scrapping in JavaScript

                        '''
                        print('Scrapping in JavaScript')
                        driver_path="C:\\Users\\emers\\AppData\\Local\\Programs\\Python\\Python38\\chromedriver.exe"
                        brave_path="C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe"
                        options=webdriver.ChromeOptions()
                        options.binary_location=brave_path
                        options.headless=True
                        driver=webdriver.Chrome(executable_path=driver_path, chrome_options=options)
                        driver.get(url)
                        driver.implicitly_wait(5)
                        squadPage=driver.page_source
                        if squadPage.find('{"serverData":')!=-1:
                            element_posion1=squadPage.find('{"serverData":')
                            element_posion2=squadPage.find('"serverFunctionCache":{}')
                            result_str=squadPage[(element_posion1):(element_posion2+len('"serverFunctionCache":{}')+1)]
                            result_dict=demjson3.decode(result_str)
                            initial_key='serverData'
                            print('Scrapping in JavaScript Successfully')
                        '''
                    if result_dict[initial_key]['FlightResultsList']['resultIds']==[]:
                        print("Não há resultado")
                        break
                    else:
                        resultIds=[i for i in result_dict[initial_key]['FlightResultsList']['resultIds'] if len(i)==32]
                        #print(resultIds)
                        for i in resultIds:
                            data=[(str(date_start_scrapping.date()),
                                   str(datetime.today()),
                                   str(datetime.today().strftime("%A")),
                                   str(datetime.today().strftime("%H:%M")),
                                   result_dict[initial_key]['FlightResultsList']['results'][i]['legs'][0]['segments'][0]['departure']['isoDateTimeLocal'][:10],
                                   result_dict[initial_key]['FlightResultsList']['results'][i]['legs'][0]['segments'][0]['departure']['isoDateTimeLocal'][11:],
                                   datetime.strptime(result_dict[initial_key]['FlightResultsList']['results'][i]['legs'][0]['segments'][0]['departure']['isoDateTimeLocal'][:10], "%Y-%m-%d").strftime("%A"),
                                   str((datetime.strptime(result_dict[initial_key]['FlightResultsList']['results'][i]['legs'][0]['segments'][0]['departure']['isoDateTimeLocal'][:10], "%Y-%m-%d") - datetime.today()).days +1),
                                   str(result_dict[initial_key]['FlightResultsList']['results'][i]['trackingDataLayer']['tagLayerPrice']),
                                   str(result_dict[initial_key]['FlightResultsList']['results'][i]['legs'][0]['legDurationDisplay']),
                                   str(result_dict[initial_key]['FlightResultsList']['results'][i]['legs'][0]['segments'][0]['departure']['airport']['displayName']),
                                   str(result_dict[initial_key]['FlightResultsList']['results'][i]['legs'][0]['segments'][0]['arrival']['airport']['displayName']),
                                   str(result_dict[initial_key]['FlightResultsList']['results'][i]['legs'][0]['segments'][0]['airline']['name']),
                                   str(result_dict[initial_key]['FlightResultsList']['results'][i]['optionsByFare'][0]['options'][0]['fareAmenities'][0]['restriction']),
                                   str(result_dict[initial_key]['FlightResultsList']['results'][i]['optionsByFare'][0]['options'][0]['fareAmenities'][1]['restriction'])
                                  )]

                            table_name=flight
                            query=f"INSERT INTO {table_name} (datetime_scrapping_start, datatime_scrapping, dow_date_scrapping, time_scrapping, date_flight_departure, time_flight_departure, dow_flight_departure, time_lapse, price, flight_duration, place_boarding, place_destination, company, carryon_bag, checked_bag     ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
                            self.cursor.executemany(query, data)
                            #self.cursor.execute(f"DELETE FROM {table_name}")
                            self.cursor.commit()
                            print(data)
                            count=0
                        break
                except:
                    print('Erro')
                    print(traceback.print_exc())
                    if count>=8:
                        a+=1
                        if a>=2:
                            break
                        else:
                            #recaptcha_solver()
                            count=0
                            continue
                    #time.sleep(10)
                    try:
                        if count>=5:
                            count=0
                            recaptcha_solver()
                    except:
                        time.sleep(10)
                        print('Erro recaptcha solver')
        self.cursor.close()
        self.conexaoDB.close()

a=DatesScrapping(date_start_scrapping)
a.Date_Scrapping()


