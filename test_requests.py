addr = [1,2]
cnnOut = 2
import requests
import json

data = {'account': 'MES'}

response = requests.post('http://127.0.0.1:5000/get_account', json=data)
print('MES')
print(response.text)

data = {'account': 'REQUESTER1'}
response = requests.post('http://127.0.0.1:5000/get_account', json=data)
print('REQUESTER1')
print(response.text)

data = {'req_addr': 'REQUESTER1',
        'data': {'hash': hash(str(cnnOut)),
                 'sol': {'name': 'hello',
                         'age': 1}
                 },
        'time_stamp': '201821199'
        }
response = requests.post('http://127.0.0.1:5000/create_transaction_by_contract', json=data)
print(response.text)

data = {'account': 'MES'}

response = requests.post('http://127.0.0.1:5000/get_account', json=data)
print('MES')
print(response.text)

data = {'account': 'REQUESTER1'}

response = requests.post('http://127.0.0.1:5000/get_account', json=data)
print('REQUESTER1')
print(response.text)