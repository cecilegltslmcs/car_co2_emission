import requests

url = "http://localhost:9696/predict"

car = {
    "make": "renault",
    "model" : "kangoo",
    "energy" : "gasoline",
    "car_classification" : "leisure_activity_vehicle",
    "cylinder_capacity" : 1332,
    "market_category" : "low",
    "tax_horsepower" : 7,
    "max_horsepower" : 96,
    "weight" : 1519,
    "weight_horsepower_ratio" : 0.06,
    "no_gears" : 7,
    "transmission_type" : "automatic",
    "bonus_malus" : "malus",
    "low_speed_fuel_consumption" : 8.7405,
    "average_speed_fuel_consumption" : 6.788,
    "high_speed_fuel_consumption" : 6.188,
    "very_high_speed_fuel_consumption" : 7.9175,
    "combined_speed_fuel-consumption" : 7.2635
}

response = requests.post(url, json=car)
response.raise_for_status()
data = response.json()

print(f"Estimated CO2 emission: {data} g/km.")