# -*- coding: utf-8 -*-
"""
Created on Sat May 13 00:24:33 2023

@author: User
"""

# Importing Libraries for the purpose of this analysis

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import random
import matplotlib.pyplot as plt
import cluster_tools as ct
import matplotlib.patches as mpatches
import scipy.optimize as opt
import numpy as np
import errors as err


def read_gdp_data(file_path):
    GDP_per_cap = pd.read_csv(file_path, skiprows=4)
    GDP_per_cap = GDP_per_cap.dropna().drop(['Indicator Code', 'Country Code', 'Indicator Name'], axis=1)
    GDP_per_cap.set_index('Country Name', drop=True, inplace=True)
    return GDP_per_cap

# Reading the  Gdp of countries data into a dataframe
GDP_per_cap = pd.read_csv('Gdp_of_Countries.csv', skiprows=4)
GDP_per_cap = GDP_per_cap.dropna().drop(['Indicator Code', 'Country Code', 'Indicator Name'], axis=1)
GDP_per_cap.set_index('Country Name', drop=True, inplace=True)
print(GDP_per_cap)

# 60 countries are randomly selected from the list
selected_countries = random.sample(list(GDP_per_cap.index), 60)
df_selected = GDP_per_cap.loc[selected_countries]

print(df_selected)

# List of countries that are not needed are removed
countries_to_remove = ['East Asia & Pacific (excluding high income) ',
'IDA total',
'Euro area',
'Sub-Saharan Africa',
'IDA & IBRD total',
'North America',
'Middle income',
'Lower middle income',
'South Asia (IDA & IBRD)',
'Latin America & Caribbean (excluding high income)',
'IDA only','Latin America & the Caribbean (IDA & IBRD countries)','Sub-Saharan Africa (IDA & IBRD countries) ', 'Early-demographic dividend', 'Pre-demographic dividend', 'Low & middle income', 'Africa Eastern and Southern', 'East Asia & Pacific (excluding high income)', 'Euro area', 'Latin America & Caribbean', 'Africa Eastern and Southern', 'World', 'Sub-Saharan Africa (IDA & IBRD countries)']

# The countries are removed from the DataFrame
df_selected = df_selected[~df_selected.index.isin(countries_to_remove)]

# Print the resulting DataFrame
print(df_selected.index)

GDP_per_cap1 = df_selected[["1960", "1970", "1980", "1990", "2000", "2010","2015", "2020"]]
print(GDP_per_cap1)

GDP_per_cap1 = GDP_per_cap1.drop(["1960"], axis=1)
print(GDP_per_cap1)

pd.plotting.scatter_matrix(GDP_per_cap1, figsize=(9, 9), s=5, alpha=0.8)

plt.show()

# Columns are extracted for the purpose of fitting, also using copy(), prevents changes in order to get df_fit to affect df_fish.
GDP_per_cap1_fit = GDP_per_cap1[["1980", "2020"]].copy()

# normalise dataframe and inspect result
# normalisation is done only on the extract columns. .copy() prevents
# changes in GDP_fit to affect GDP_fit. This makes the plots with the
# original measurements
GDP_per_cap1_fit, df_min, df_max = ct.scaler(GDP_per_cap1_fit)
print(GDP_per_cap1_fit.describe())

print("n score")

# Trial numbers of clusters are looped over for calculating the silhouette
for ic in range(2, 7):
# set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(GDP_per_cap1_fit)

# extracting labels and calculate silhoutte score
labels = kmeans.labels_
print (ic, skmet.silhouette_score(GDP_per_cap1_fit, labels))

# Fit k-means with 2 clusters
kmeans = cluster.KMeans(n_clusters=2)
kmeans.fit(GDP_per_cap1_fit)

# Cluster label columns are added to the original dataframe
GDP_per_cap1["cluster_label"] = kmeans.labels_

# Countries are grouped by cluster label
grouped = GDP_per_cap1.groupby("cluster_label")

# Print countries in each cluster
for label, group in grouped:
    print("Cluster", label)
    print(group.index.tolist())
    print()

# Clusters are plotted with labels
plt.scatter(GDP_per_cap1_fit["1980"], GDP_per_cap1_fit["2020"], c=kmeans.labels_, cmap="Set1")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c="k", marker="d", s=80)
plt.xlabel("1960")
plt.ylabel("2020")
plt.title("CLUSTER GDP OF COUNTRIES")
plt.show()

def read_population_data(file_path):
    df_pop = pd.read_csv(file_path, skiprows=4)
    df_pop = df_pop.dropna().drop(['Indicator Code', 'Country Code', 'Indicator Name'], axis=1)
    df_pop.set_index('Country Name', drop=True, inplace=True)
    return df_pop


# Read the World Bank population data into a dataframe
df_pop = pd.read_csv("Pop_world_data.csv", skiprows=4)

df_pop = df_pop.dropna().drop(['Indicator Code', 'Country Code', 'Indicator Name'], axis=1)

df_pop.set_index('Country Name', drop=True, inplace=True)
print(df_pop)
# Defining countries of choice and transposing the dataframe
countries = ['China', 'Japan']

print(countries)


df_pop_countries = df_pop.loc[countries]

df_pop_countries = df_pop_countries.transpose()
df_pop_countries = df_pop_countries.rename_axis('Year')

df_pop_countries = df_pop_countries.applymap(lambda x: pd.to_numeric(x, errors='coerce'))

print(df_pop_countries)

"""
creating a def function to read in the population dataset logistics function and returns with scale and incr as free parameters

"""

def logistics(t, a, k, t0):
    """
    Computes the logistic function with scale and increment as free parameters.
    
    Args:
        t: Input value or an array of input values.
        a: Scale parameter.
        k: Increment parameter.
        t0: Midpoint parameter.
    
    Returns:
        The computed value(s) of the logistic function.
    """
    return a / (1.0 + np.exp(-k * (t - t0)))

# Converting index to numeric and use it in curve fitting
df_pop_countries.index = pd.to_numeric(df_pop_countries.index)

popt, pcorr = opt.curve_fit(logistics, df_pop_countries.index, df_pop_countries["China"], p0=(16e8, 0.04, 1985.0))
print("Fit parameter", popt)

# Extract variances and calculate sigmas
sigmas = np.sqrt(np.diag(pcorr))

df_pop_countries["pop_logistics"] = logistics(df_pop_countries.index, *popt)

# call function to calculate upper and lower limits with extrapolation
# create extended year range
years = np.arange(1950, 2051)
lower, upper = err.err_ranges(years, logistics, popt, sigmas)

plt.figure()
plt.title("logistics function")
plt.plot(df_pop_countries.index, df_pop_countries["China"], label="data")
plt.plot(df_pop_countries.index, df_pop_countries["pop_logistics"], label="fit")

# Error ranges are plotted with transparency
plt.fill_between(years, lower, upper, alpha=0.5)

plt.legend(loc="upper left")
plt.show()

fig, axs = plt.subplots(2, 1, figsize=(10, 10)) # create a plot for Japan
countries = ['China', 'Japan']



# Initial parameters for each country with difference are considered
p0_values = [(16e8, 0.04, 1985.0), (16e8, 0.03, 1980.0)] # adjust these initial guesses as needed

# The curve fitting is applied for each country
for i, country in enumerate(countries):
    popt, pcorr = opt.curve_fit(logistics, df_pop_countries.index, df_pop_countries[country], p0=p0_values[i], maxfev=10000)

# Extract variances and calculate sigmas
sigmas = np.sqrt(np.diag(pcorr))

# Create extended year range
years = np.arange(1950, 2051)
# Calculate the fitted GDP and the confidence intervals
pop_logistics = logistics(years, *popt)
lower, upper = err.err_ranges(years, logistics, popt, sigmas)

# Plotting the original data, the fitted function, and the confidence ranges
axs[i].plot(df_pop_countries.index, df_pop_countries[country], label="data")
axs[i].plot(years, pop_logistics, label="fit")
axs[i].fill_between(years, lower, upper, alpha=0.5)

axs[i].set_title(f"logistics function for {country}")
axs[i].legend(loc="upper left")
# Adjust the subplots to fit in to the figure area.
plt.tight_layout()
plt.show()



def poly(t, c0, c1, c2, c3):
    """ Computes a polynomial c0 + c1*t + c2*t^2 + c3*t^3 """
    t = t - 1950
    f = c0 + c1*t + c2*t**2 + c3*t**3
    return f

# Fit the polynomial to the data
popt, pcorr = opt.curve_fit(poly, df_pop_countries.index, df_pop_countries["China"])
print("Fit parameters:", popt)

# Extract variances and calculate sigmas
sigmas = np.sqrt(np.diag(pcorr))

# Calculate the polynomial values for the data range
df_pop_countries["poly"] = poly(df_pop_countries.index, *popt)

# Create an extended year range for plotting
years = np.arange(1950, 2051)

# Calculate the upper and lower error ranges
lower, upper = err.err_ranges(years, poly, popt, sigmas)

# Plot the data, fit, and error ranges
plt.figure()
plt.title("Polynomial Fit with Error Ranges")
plt.plot(df_pop_countries.index, df_pop_countries["China"], label="Data")
plt.plot(df_pop_countries.index, df_pop_countries["poly"], label="Fit")

# Plot the error ranges with transparency
plt.fill_between(years, lower, upper, alpha=0.5, label="Error Ranges")

plt.xlabel("Years")
plt.ylabel("Population")
plt.legend(loc="upper left")

plt.show()

import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt

def poly(t, c0, c1, c2, c3):
    """ Computes a polynomial c0 + c1*t + c2*t^2 + c3*t^3 """
    t = t - 1950
    f = c0 + c1*t + c2*t**2 + c3*t**3
    return f

def err_ranges(x, func, popt, perr):
    """ Calculate upper and lower errors """
    popt_up = popt + perr
    popt_dw = popt - perr
    fit = func(x, *popt)
    fit_up = func(x, *popt_up)
    fit_dw = func(x, *popt_dw)
    return fit_up, fit_dw

# Ensure index is of integer type
df_pop_countries.index = df_pop_countries.index.astype(int)

# Initialize a figure
fig, axs = plt.subplots(1, 2, figsize=(15,7))

# Loop over the countries in the list
for i, country in enumerate(countries):
    popt, pcorr = opt.curve_fit(poly, df_pop_countries.index, df_pop_countries[country])
    print(f"Fit parameters for {country}: ", popt)
    # extract variances and calculate sigmas
    sigmas = np.sqrt(np.diag(pcorr))
    # call function to calculate upper and lower limits with extrapolation
    # create extended year range
    years = np.arange(1950, 2051)
    lower, upper = err_ranges(years, poly, popt, sigmas)
    axs[i].plot(df_pop_countries.index, df_pop_countries[country], label="data")
    axs[i].plot(years, poly(years, *popt), label="fit")
    # plot error ranges with transparency
    axs[i].fill_between(years, lower, upper, alpha=0.5)
    axs[i].set_title(f"Polynomial Fit for {country}")
    axs[i].legend(loc="upper left")

# Adjust layout for neat presentation
plt.tight_layout()
plt.show()
