import requests


def fetch_weather_data(city, start_date, end_date, api_key):
    base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
    url = f"{base_url}{city}/{start_date}/{end_date}?unitGroup=metric&include=days&key={api_key}&contentType=csv"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to retrieve data for {city}. Status code: {response.status_code}")
        return None

def save_data(city, data):
    if data:
        filename = f"{city.replace(' ', '_').lower()}_weather.csv"
        with open(filename, 'w') as file:
            file.write(data)
        print(f"Data for {city} saved as {filename}.")

# List of cities
cities = ["Berlin", "London", "New York"]

# API Key - Replace with your actual key
api_key = "6EYHQL3BW7NVBUHA2B24LUQUQ"

# Dates
start_date = "2021-04-01"
end_date = "2023-07-01"

# Fetch and save data for each city
for city in cities:
    data = fetch_weather_data(city, start_date, end_date, api_key)
    save_data(city, data)

import pandas as pd

all_data = pd.DataFrame()

for city in cities:
    data = fetch_weather_data(city, start_date, end_date, api_key)
    if data:
        city_data = pd.read_csv(pd.compat.StringIO(data))
        city_data['City'] = city  # Add a city column
        all_data = pd.concat([all_data, city_data], ignore_index=True)

all_data.to_csv('all_cities_weather.csv', index=False)
