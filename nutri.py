import requests
import json

#Insert grams and type of food in order to recieve JSON of the nutrients.
def insert_grams_and_food(grams, food):
    api_url = 'https://api.calorieninjas.com/v1/nutrition?query='
    query = grams + "g " + food #Example: 300g rice
    print(query + " has been requested:")
    print("Calculating Nutrients...")
    print("Loading...")
    response = requests.get(api_url + query, headers={'X-Api-Key': 'dCCISnMtnXaYp6JAYg+W4A==gIEtvqXr5ctaoorg'})
    data = response.json()
    # print(data['items'][0]['calories']) #Specific nutri
    if response.status_code == requests.codes.ok:
        print(data['items'][0])
        return data['items'][0]


    else:
        print("Error:", response.status_code, response.text)
        



