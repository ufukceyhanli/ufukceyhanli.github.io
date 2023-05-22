---
title: "Forecasting of Gasoline MED-FOB Price Based on Historical Data with Python"
date: 2020-07-08T08:06:25+06:00
hero: /posts/forecasting-of-gasoline-med-fob-price-based-on-historical-data/gas.jpg
description: Forecasting of Gasoline MED-FOB Price Based on Historical Data
menu:
  sidebar:
    name: Forecasting of Gasoline MED-FOB Price Based on Historical Data
    identifier: Forecasting of Gasoline MED-FOB Price Based on Historical Data
    weight: 30
---
Petroleum refineries are large, capital-intensive, continuous-flow manufacturing facilities. 
They transform crude oils into finished, refined products by separating crude oils into different fractions and then processing these fractions into finished products, through a sequence of physical and chemical conversions. Production planning and process scheduling is one of the most important parts of refineries for maximizing the profit of production. 
Although there are many inputs to optimize this problem, price of blend crude oil (raw material), product price to raw material price ratio, and crack margin ( product price minus raw material price) are critically important since very small change in these variable can lead to very different scenarios.

The availability of LP-based commercial software for refinery production planning, such as PIMS (Process Industry Modeling System - Bechtel, 1993), has allowed the development of general production plans for the whole refinery, which can be interpreted as general trends. 
General approach for production and scheduling process has two parts: First, monthly rolling plans for crude selection and conducting refinery operations in line with foreseen demands. Secondly, based on monthly planning, implementing short-term (weekly/daily) plans for finding operating strategies regarding either precise or a good level of knowledge about crude availability, product delivery, operational and logistic constraints, as well as economic issues. One of the most important inputs for monthly planning are price of product and crude oil. However, since the aim is to plan next month, price for both inputs and outputs are actually unknown.

In this project, the aim is to develop a methodology to gasoline crack margin by using time series analysis and forecasting techniques. 
We used real data which can be extracted from Platts. We implemented following forecasting methods: Moving Average, Exponential Smoothing, Exponentially Weighted Moving Average, Holt-Winter’s. 
However, the most important part of the project is applying Box-Jenkins methodology to data and make an automated tool which carries out all Box-Jenkins process without any need for individual intervention.

Solving this problem will help Tupras to increase their production and scheduling accuracy, hence their profit. 
Moreover, we will try to compare all forecasting techniques and their accuracy.


###### Github Repository: https://github.com/tanerceyhanli/Forecasting-of-Gasoline-MED-FOB-Price-Based-on-Historical-Data
###### Jupyter Notebook: https://github.com/tanerceyhanli/Forecasting-of-Gasoline-MED-FOB-Price-Based-on-Historical-Data/blob/main/Forecasting-of-Gasoline-MED-FOB-Price-Based-on-Historical-Data.ipynb
