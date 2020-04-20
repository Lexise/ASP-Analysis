# ASP-Analysis
A Dash application that uses data mining algorithms (KMeans, DBscan clustering) to analyses the whole space of answer sets, identifies the inner patterns, and helps users to find the interesting attributes for further investigation.

## Prerequisites

Python 3.6 or later version

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Running the app locally


First create a virtual environment with conda or venv inside a temp folder, then activate it.
```
virtualenv venv
# Windows
venv\Scripts\activate
# Or Linux
source venv/bin/activate
```

Clone the git repo
```
git clone https://github.com/Lexise/ASP-Analysis

```
### Installing

To install all of the required packages to your virtual environment, simply run:
```
pip install -r requirements.txt
```

Run the app

```
python app.py
```



## Built With

* [Dash](https://dash.plotly.com/) - Main server and interactive components
* [Plotly Python](https://dash.plotly.com/) - Used to create the interactive plots

## Screenshots
The following are screenshots for the app in this repo:
![Image of main interface](https://github.com/Lexise/ASP-Analysis/Screenshots/tab_cluster_scatter.png)
![Image of Attribute anaysis](https://github.com/Lexise/ASP-Analysis/tree/master/Screenshots/attribute_distribution_analysis.png)
![Image of Attribute anaysis](https://github.com/Lexise/ASP-Analysis/tree/master/Screenshots/attribute_selection.png)
![Image of correlation matrix](https://github.com/Lexise/ASP-Analysis/tree/master/Screenshots/correlation_matrix.png)

## Acknowledgments

* Inspired by gallery of dash plotly

