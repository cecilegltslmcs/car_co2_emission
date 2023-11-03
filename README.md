# CO2 emissions for cars sold in France

*Author: Cécile Guillot*

*Last Update: 3/11/2023*

## Summary

The fight against greenhouse gas emissions is becoming an important issue in our world. By changing our habits, we can ensure that we don't add too much to global warming. One of the habits we can change is the way we travel. We can take public transport for our daily journeys, or the bus or train for journeys further afield, but sometimes we have no choice but to use the car. So it can be useful to know the CO2 emissions produced by your own vehicle, or to help you buy a new one.

The aim of this project is to help people get an idea of the CO2 emissions produced by their car, and also to help them make the right decision when buying a new vehicle. Based on the data that motorists can find in their car's documentation and using our tool, they can get an idea of their car's CO2 emissions.


## Dataset

Original dataset comes from ADEME (French Agency for Ecological Transition) and can be downloaded [here](https://data.ademe.fr/datasets/ademe-car-labelling). Data are also available in the folder "[data](https://github.com/cecilegltslmcs/car_co2_emission/tree/main/data)".

## Methodology

All the data was cleaned up to simplify use of the final tool. As the data comes from a French website, a translation has been made to make it easier to understand. All these modifications are included in notebook 001.preparation_translation.ipynb.

Exploratory data analysis and machine learning modelling were then carried out in notebook 002.notebook.ipynb.

The best model was then trained and serialized in a Python script called train.py. The final model is deployed using Flask (code available in the predict.py script). Finally, this application was placed in a Docker container.

## How to used the app ?

- Install and run Docker
- Build the image by using `docker build -t co2_emission .`
- Run the container with the command `docker run co2_emission`
- Run the script `predict-test.py`