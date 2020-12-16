---
title: "Data Analysis of House Sales Prices and Venues in Prague with Python"
date: 2020-08-08T08:06:25+06:00
hero: /posts/data-analysis-of-house-sales-prices-and-venues-in-prague-with-python/prague.jpg
description: Data Analysis of House Sales Prices and Venues in Prague with Python
menu:
  sidebar:
    name: Data Analysis of House Sales Prices and Venues in Prague with Python
    identifier: Data Analysis of House Sales Prices and Venues in Prague with Python
    weight: 1
---

### Introduction
Prague is one of the most popular cities in Europe with many outstanding views and historic, exciting structures. My journey to Prague has started with the hiring of my wife by a global company there. In this study, I give a brief insight about the population, house prices and types by comparing municipal parts of the city for anyone to buy a property in Prague or interested in the real estate industry. I'll go over these steps;

- Observe the municipal parts of Prague
- Population distribution by municipal parts
- House prices by municipal parts
- Number of different types of venues inside 1km radius of the center of municipal parts
- Cluster municipal parts of Prague

#### Municipal Parts of Prague
If you visited Prague, probably you might have been to Prague 1, 2 and 3, however, the city has fifty-seven municipal parts according to the official resources, while the first twenty-second region is called by a number, others by names.

Here is a map of Prague regions which I created using Folium library of Python.
<br/><br/>
{{< img src="1.gif">}}
<br/><br/>
#### Population of Prague by Municipal Parts
I downloaded the population data for 57 Municipal Parts of Prague from the web page below and created a pandas data frame. Resource: https://www.citypopulation.de/en/czechrep/praguecity/
<br/><br/>
{{< img src="2.png">}}
<br/><br/>
Let's visualize the population of Prague, with the help of choropleth library.
<br/><br/>
{{< img src="3.png">}}
<br/><br/>
There are more people living in Praha 4 and 10 than other municipal regions.

#### House Sale Prices
Sreality.com is the most used website for advertisement of houses in Czech Republic. I scraped all available house advertisements in the website for Prague and refined the data. In the data frame below, there are 4390 advertising with its features.
<br/><br/>
{{< img src="3g.png">}}
<br/><br/>
By using latitude and longitude and Prague municipal part geojson file, I received region info for each object and drop title column. First 5 items of the data frame are below.
<br/><br/>
{{< img src="4g.png">}}
<br/><br/>
Let's do some Exploratory Data Analysis excluding the items belonging to the municipal parts having advertisements less than 30.
<br/><br/>
{{< img src="4.png">}}
<br/><br/>
Average prices for a house in Praha 1 and Praha 6 are higher than others.
<br/><br/>
{{< img src="5.png">}}
<br/><br/>
The houses in Praha 1 and Praha 6 are larger than others.
Price per square meter is one of the most popular metrics in the real estates industry.
<br/><br/>
{{< img src="6.png">}}
<br/><br/>
The price per square meter is high for Praha 1 and Praha 2, compared to others.
Praha 6 is not in the first ranks for the price per square meter, however it was in the first ranks for the average price for a house, it seems that the reason for this, the size of the houses in Praha 6 is larger.

Here I would like to describe type of the houses in Czech Republic. The naming could be unfamiliar to you but basically,

- 1+1: one room and one separated kitchen 1+kk: one room and the kitchen is inside room 
- 2+1: two rooms and one separated kitchen 2+kk: two rooms and the kitchen is inside one of the room
- 2+kk is most prevalent house type in Prague.
<br/><br/>
{{< img src="7.png">}}
<br/><br/>
We may wonder where the most expensive houses are located in Prague.
<br/><br/>
{{< img src="8.png">}}
<br/><br/>
The most expensive houses are mainly located in Praha 1.

#### Clustering Municipal Parts by Venues
In this section, I will cluster municipal parts based on the number of venues in different categories around the center of the municipal parts. I defined the latitude and longitude of the center of the municipal parts and obtained the available venues in 1 km radius of the center point by using Google Places API.
<br/><br/>
{{< img src="9.png">}}
<br/><br/>
I created a data frame for the first ten municipal parts and its surrounding supermarkets, bus stations, museums, subway stations and pharmacies.
<br/><br/>
{{< img src="10.png">}}
<br/><br/>
#### K-means Clustering
After applying k-means clustering from scikit-learn library, the municipal parts are divided into 3 clusters.
<br/><br/>
{{< img src="11.png">}}
<br/><br/>
It seems that Praha 1 and 2 have many venues around the center so these two are distinctive than others. Praha 5 can be separated because it comes forward with bus station availability. Other regions have similar features.

### Results and Discussions
There are more people living in Praha 4 and 10 than other municipal regions.
Average prices for a house in Praha 1 and 6 are higher than others.
The houses in Praha 1 and Praha 6 are larger than others.
The price per square meter is high for Praha 1 and 2, compared to others.
Praha 6 is not in the first ranks for the price per square meter, however it was in the first ranks for the average price for a house, it seems that the reason of this, the size of the houses in Praha 6 is larger.
2+kk is the most prevalent house type in Prague.
The most expensive houses are mainly located in Praha 1.
Based on the venues available, Prague municipal parts can be divided in three clusters.
Do the venues around municipal parts contribute to the price of the house?
<br/><br/>
{{< img src="12.png">}}
<br/><br/>
{{< img src="13.png">}}
<br/><br/>
Yes. We may divide the municipal parts to 3 groups based on price per square meter. Praha 1 and 2 are more expensive than other municipal parts, following Praha 5 and others. This fits the cluster which is created only by the numbers of different types of venues around the municipal parts' center.

### Conclusion
Prague has 57 municipal parts and each of them may have different features. However some of them have more venues around and this contributes to the price of the house.

Ufuk Taner CEYHANLI

11/17/2020

###### Github Repository: https://github.com/tanerceyhanli/Data-Analysis-of-House-Sales-Prices-and-Venues-in-Prague-with-Python
###### Jupyter Notebook: https://github.com/tanerceyhanli/Data-Analysis-of-House-Sales-Prices-and-Venues-in-Prague-with-Python/blob/main/Data-Analysis-of-House-Sales-Prices-and-Venues-in-Prague-with-Python.ipynb
###### Linkedin Post: https://www.linkedin.com/pulse/house-sales-prices-venues-data-analysis-prague-python-ceyhanl%C4%B1/
