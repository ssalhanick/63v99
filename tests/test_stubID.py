import requests
from config import COURTLISTENER_TOKEN, COURTLISTENER_BASE_URL
headers = {'Authorization': f'Token {COURTLISTENER_TOKEN}'}
for oid in [11233869, 10755384, 9847853, 9795106, 9513889, 9433305, 9424643, 9424320, 4212343, 48, 9806364]:
    r = requests.get(f'{COURTLISTENER_BASE_URL}/opinions/{oid}/', headers=headers).json()
    cluster = requests.get(r['cluster'], headers=headers).json()
    print(oid, cluster.get('case_name'), cluster.get('date_filed'))