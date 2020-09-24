# ASP-Analysis
A Dash application that uses data mining algorithms (KMeans, DBscan clustering) to analyses the whole space of answer sets, identifies the inner patterns, and helps users to find the interesting attributes for further investigation.

## Prerequisites

Python 3.6 or later version
Microsoft Visual C++ 14.0

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
### Upload data form

If you want to process your own data, it should be named in the way that end with ".apx" for arguments ans ".EE-PR" for answer sets.
For example:

>[metro-st-louis_20130916_1522.gml.80.apx](https://github.com/Lexise/ASP-Analysis/blob/master/Screenshots/metro-st-louis_20130916_1522.gml.80.apx)

>[metro-st-louis_20130916_1522.gml.80.EE-PR](https://github.com/Lexise/ASP-Analysis/blob/master/Screenshots/metro-st-louis_20130916_1522.gml.80.EE-PR)



## Built With

* [Dash](https://dash.plotly.com/) - Main server and interactive components
* [Plotly Python](https://dash.plotly.com/) - Used to create the interactive plots

## Screenshots
The following are screenshots for the app in this repo:

![main interface](https://github.com/Lexise/ASP-Analysis/blob/master/Screenshots/tab_cluster_scatter.png)
![Attribute anaysis1](https://github.com/Lexise/ASP-Analysis/blob/master/Screenshots/attribute_selection.png)
![Attribute anaysis1](https://github.com/Lexise/ASP-Analysis/blob/master/Screenshots/attribute_distribution_analysis.png)
![correlation matrix](https://github.com/Lexise/ASP-Analysis/blob/master/Screenshots/correlation_matrix.png)

## Acknowledgments

* Inspiration

* Academic usage
