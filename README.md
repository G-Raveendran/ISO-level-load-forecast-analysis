# ISO-level-load-forecast-analysis
## Project Overview
This project focuses on analyzing and forecasting electricity demand using the PJM Hourly Load data for the AP zone. We explore the seasonal variations in load patterns and build a predictive model using the Random Forest algorithm to generate day-ahead forecasts of electricity demand. The goal is to create an accurate and reliable model to predict future electricity load based on historical data.

## Table of Contents
1. Project Overview
2. Dataset
3. Setup
4. Data Cleaning
5. Seasonal Analysis
6. Visualization
7. Feature Engineering and Modeling
8. Usage
9. Conclusion
10. References
## Dataset
The dataset used in this project is the "PJM Hourly Load: Metered data for AP zone," available from the PJM Interconnection. The key columns used in this analysis are:

datetime_beginning_ept: Timestamp in Eastern Prevailing Time (EPT)
mw: Load in megawatts (MW)
## Data Source
PJM Hourly Load Data

## Usage
1. Load the Dataset: Load the PJM hourly load dataset.
2. Run Data Cleaning: Execute the data cleaning script to prepare the data.
3. Analyze Seasons: Segment the data into different seasons and visualize the load patterns.
4. Build the Model: Train the Random Forest model and evaluate its performance.
5. Forecast Load: Use the model to generate day-ahead forecasts of electricity demand.
## Conclusion
This project demonstrates the application of machine learning techniques to forecast electricity demand based on historical load data. The Random Forest model provides a reliable approach to predicting future electricity usage, which can be valuable for grid management and energy planning.

## References
PJM Hourly Load Data
PJM Interconnection
