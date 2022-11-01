# Model: Energy Consumption in Zurich
This is the model, made by ewz, rewritten by shu. Further information at [EWZ](https://www.ewz.ch/de/ueber-ewz/newsroom/aus-aktuellem-anlass/versorgung-sichergestellt/energieverbrauch-stadt-zuerich.html)

## How the model works
> The statistically expected electricity consumption generated with machine learning is calculated for the previous seven days and displayed as a daily average in a bandwidth. In addition, the actual measured weather data is used and the upper and lower limits of the statistically expected electricity consumption are calculated with a regression model (Prophet library). The actual electricity consumption is also based on the average measured values of the last seven days (rolling average) so that the two values can be compared with each other. The regression model was trained with the measured consumption and weather data of the city of Zurich from 1 January 2010 to 31 December 2021. Thanks to this procedure, deviations due to weekends and public holidays can be taken into account, as these do not fall on the same date every year and energy consumption is lower than during the week. [Source](https://www.ewz.ch/de/ueber-ewz/newsroom/aus-aktuellem-anlass/versorgung-sichergestellt/energieverbrauch-stadt-zuerich.html)

## Data sources
* [Energy Consumption data](https://data.stadt-zuerich.ch/dataset/ewz_stromabgabe_netzebenen_stadt_zuerich)
* Weather data (Meteo Schweiz)

## Scripts
* `PrepareTrainData.ipynb` Prepare the data
* `TrainFinalModel.ipynb` Train model

## Good to know
Columns:
* NE5 (Netzebene 5) = Bigger Companies
* NE7 (Netzebene 7) = Households, small Companies (KMU)

## Installation on Mac M1
Mac Silicon is a pain... again... Scikit-Learn needs a lot of love to get installed. And Prophet does not work on newest Python Version in Mac M1 (error Message: `python3.10/site-packages/prophet/stan_model/prophet_model.bin Reason: image not found`)... So:  

**Use Python 3.8!**

```
python3.8 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

If Scikit-learn throws errors:
```
brew install openblas
export OPENBLAS=$(/opt/homebrew/bin/brew --prefix openblas)
export CFLAGS="-falign-functions=8 ${CFLAGS}"
pip install scikit-learn
```
[Source](https://github.com/scipy/scipy/issues/13409)

Sometimes Openblas is not linked correctly. Do:
```
brew link openblas --force
```

Other ways...
```
pip install cython pybind11 pythran numpy
OPENBLAS=$(brew --prefix openblas) CFLAGS="-falign-functions=8 ${CFLAGS}" pip install --no-use-pep517 scipy==1.3.2
```
or
```
pip3 install -U --no-use-pep517 scikit-learn
```



