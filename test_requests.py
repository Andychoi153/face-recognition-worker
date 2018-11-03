addr = [1,2]
cnnOut = 2
import requests
import json

data = {'req_addr': str(addr[0]) + ':' + str(addr[1]),
        'data': {'hash': hash(str(cnnOut)),
                 'sol': {'name': 'hello',
                         'age': 'goodmorning'}
                 },
        'time_stamp': '201821199'
        }
requests.post('http://127.0.0.1:5000/create_transaction_by_contract', json=data)
print(json.dumps(data))
