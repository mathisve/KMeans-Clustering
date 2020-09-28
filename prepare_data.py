import numpy as np
import pandas as pd
import requests

url = 'https://storage.googleapis.com/kagglesdsdata/datasets/571434/1035931/data.txt?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20200928%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20200928T185025Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=6a5c7187bd81854fed900007d1606c7c12e4eb39c7fd0a7061ddc8e4382628aff9d5a69cad5e4397ae76496e3f1457ae5eca1d07232899e12a8eb5d223fb1f24acecac677b283363d6b74c1ea77204ac6509a227748bffc0dbcbea475e77e03dc1b8e79a73b86e412c19789c9a7b24fd4f4a448dc94a20662b9b4b63d92c0ba91b2396d48a929d2ef163a69a37e94212a402e16027805dfe915098544d0c2f318e2f599e1dc51f9bc75ea6a3e14b6a81e217be7545c19fe144af6f99ee9cea08c8c553735c559146e1fc6e362432307ac1adbb82a62eda70897c008cc39cb2ff9ee19669e1d4db891dc3de381a08fd43b194fb22de630b9ebdb55b53e2a92995'

file = requests.get(url)
open('data.txt', 'wb').write(file.content)

x = []

with open('data.txt', 'r') as file:
    data = file.read()
    lines = data.split("\n")
    for line in lines:
        split = line.split("	")
        x.append([float(split[0].rstrip()), float(split[1].rstrip())])
       
x = np.asarray(x)
print(x)
df = pd.DataFrame(data=x, columns=["x","y"])
df.to_csv("data.csv", index=False)

print(df.head)
