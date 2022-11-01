# Model: Energy Consumption in Zurich
This is the model, made by ewz, rewritten by shu. Further information at [EWZ](https://www.ewz.ch/de/ueber-ewz/newsroom/aus-aktuellem-anlass/versorgung-sichergestellt/energieverbrauch-stadt-zuerich.html)

## How the model works
> The statistically expected electricity consumption generated with machine learning is calculated for the previous seven days and displayed as a daily average in a bandwidth. In addition, the actual measured weather data is used and the upper and lower limits of the statistically expected electricity consumption are calculated with a regression model (Prophet library). The actual electricity consumption is also based on the average measured values of the last seven days (rolling average) so that the two values can be compared with each other. The regression model was trained with the measured consumption and weather data of the city of Zurich from 1 January 2010 to 31 December 2021. Thanks to this procedure, deviations due to weekends and public holidays can be taken into account, as these do not fall on the same date every year and energy consumption is lower than during the week.

[Source](https://www.ewz.ch/de/ueber-ewz/newsroom/aus-aktuellem-anlass/versorgung-sichergestellt/energieverbrauch-stadt-zuerich.html)

## Data sources
* [Energy Consumption data](https://data.stadt-zuerich.ch/dataset/ewz_stromabgabe_netzebenen_stadt_zuerich)
* Weather data (Meteo Schweiz)

## Good to know
Columns:
* NE5 (Netzebene 5) = Bigger Companies
* NE7 (Netzebene 7) = Households, small Companies (KMU)