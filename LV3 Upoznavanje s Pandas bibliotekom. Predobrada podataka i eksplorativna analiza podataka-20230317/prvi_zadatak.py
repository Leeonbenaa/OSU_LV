import pandas as pd
import numpy as np

data = pd.read_csv('data_C02_emission.csv')
print('Ukupno mjerenja:', len(data))
print('Veličine su tipa:', data.dtypes)
print('Broj Izostalih vrijednosti:', data.isnull().sum())
print('Broj duplikata:', data.duplicated().sum())
data.dropna()
data.drop_duplicates()
data = data.reset_index ( drop = True )
print('Ukupno mjerenja bez null vrijednosti i duplikata:', len(data))

cols = ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']
data[cols] = data[cols].astype('category')
print(data.dtypes)

fuelConsumptionMax = data[['Make', 'Model', 'Fuel Consumption City (L/100km)']].nlargest(
    3, ['Fuel Consumption City (L/100km)'])
print('Tri automobila po najvećoj potrošnji:',fuelConsumptionMax)

fuelConsumptionMin = data[['Make', 'Model', 'Fuel Consumption City (L/100km)']].nsmallest(
    3, ['Fuel Consumption City (L/100km)'])
print('Tri automobila po najmanjoj potrošnji:',fuelConsumptionMin)

engineSize=data[(data['Engine Size (L)'] > 2.5 ) & ( data ['Engine Size (L)'] < 3.5 )]
print('Broj  vozila koji imaju velicinu motora izmedu 2.5 i 3.5 L:',len(engineSize))

averageCO2Emision = data[(data['Engine Size (L)'] > 2.5) & (
    data['Engine Size (L)'] < 3.5)]['CO2 Emissions (g/km)'].mean()

print('Prosječna emisja CO2:',averageCO2Emision)
numberOfAudi = len(data[(data)['Make'] == 'Audi'])
print('Broj audi automobila:',numberOfAudi)

averageCo2EmisionAudi = data[(data['Make'] == 'Audi') & (
    data['Cylinders'] == 4)]['CO2 Emissions (g/km)'].mean()
print('Prosječna emisija CO2 audi automobila',averageCo2EmisionAudi)

evenCylinders = data[data['Cylinders'] % 2 == 0]
averageCo2EmisionByCylinders = evenCylinders.groupby(
    'Cylinders')['CO2 Emissions (g/km)'].mean()
print('Broj parnih cilindara:',len(evenCylinders))
print('Prosječan broj emisije CO2 automobila parnih cilindara:',averageCo2EmisionByCylinders)

averageCityConsumptionDiezel = data[(data['Fuel Type'] == 'D')]['Fuel Consumption City (L/100km)'].mean()
averageCityConsumptionBenzin = data[(data['Fuel Type'] == 'X')]['Fuel Consumption City (L/100km)'].mean()
print('Prosječna gradska potrošnja automobila koji koriste dizel:',averageCityConsumptionDiezel)
print('Prosječna gradska potrošnja automobila koji koriste regularni benzin:',averageCityConsumptionBenzin)
medianCityConsumptionDiezel = data[(data['Fuel Type'] == 'D')]['Fuel Consumption City (L/100km)'].median()
medianCityConsumptionBenzin = data[(data['Fuel Type'] == 'X')]['Fuel Consumption City (L/100km)'].median()
print('Median gradska potrošnja automobila koji koriste dizel:',medianCityConsumptionDiezel)
print('Median gradska potrošnja automobila koji koriste regularni benzin:',medianCityConsumptionBenzin)

maxCityConsumption = data[(data['Cylinders'] == 4) & (
    data['Fuel Type'] == 'D')]['Fuel Consumption City (L/100km)'].sort_values()
maxCityConsumption=maxCityConsumption.tail(1)
print(maxCityConsumption)


numberOfManualDrivers = len(data[(data['Transmission'] == 'M')])
print(numberOfManualDrivers)

print (data.corr( numeric_only = True ))

