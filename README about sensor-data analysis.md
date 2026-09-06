# Data Science and Machine Learning Portfolio

This repository presents three data-focused projects developed through MSc Human Computer Interaction coursework and dissertation preparation. The projects cover machine learning, sensor-data analysis, data preparation with Pandas, exploratory analysis, and model evaluation.

## Projects

1. [Machine Learning for Characterising Caring Touch](#machine-learning-for-characterising-caring-touch)
2. [Investigative Data Preparation with Pandas](#investigative-data-preparation-with-pandas)
3. [Classifying Human Interaction Patterns in Inertial Sensor Data](#classifying-human-interaction-patterns-in-inertial-sensor-data)

---

## Machine Learning for Characterising Caring Touch

### Project Summary

This dissertation project investigates how pressure-sensor data can be used to analyse caring touch in haptic and robotic systems. The current work includes a Python data processing and visualisation pipeline for tactile pressure recordings collected from an instrumented infant doll, with the longer-term aim of developing machine learning methods to characterise different touch patterns.

### Problem

Caring touch is an important but complex form of human interaction. In haptics, healthcare technology, and social robotics, understanding touch patterns could help support the design of more responsive and human-centred systems. The challenge is to convert raw tactile interaction data into features that can be interpreted and analysed using machine learning.

### Data

The project uses pressure-sensor CSV recordings collected from baby-doll interaction sessions. The available dataset includes 111 pressure recordings from 20 participant IDs, with files labelled by interaction conditions such as holding, natural interaction, and patting. Each recording contains timestamped pressure values across 42 saved pressure channels.

### Methods

- Built a Python workflow to load and validate pressure-sensor CSV files
- Checked that pressure-channel columns were present and consecutively numbered
- Converted timestamp and pressure columns into numeric arrays for analysis
- Estimated recording duration, median sampling interval, and sampling rate
- Checked duplicate, decreasing, and irregular timestamps
- Calculated total pressure across saved sensor channels for each recording
- Applied rolling-mean smoothing to support signal-level interpretation
- Created overview plots of total pressure over time
- Created channel-level heatmaps to inspect how pressure activity changed across the sensor array
- Exported a recording summary with duration, sample count, channel count, sampling-rate estimate, and pressure statistics
- Prepared the workflow for future feature extraction and machine learning modelling

### Tools

Python, Pandas, NumPy, Scikit-learn, PyTorch, Matplotlib, Seaborn, Jupyter Notebook

### Current Status

The data processing and visualisation pipeline has been implemented. The project currently supports exploratory analysis of pressure recordings, including total-pressure curves, channel heatmaps, and recording-level quality summaries. Machine learning model results have not yet been established, so final accuracy, F1-score, or clustering results should only be added after formal modelling and evaluation are complete.

### What This Project Demonstrates

- Sensor-data preprocessing
- Feature extraction from tactile interaction data
- Applied machine learning for human-centred AI
- Research-focused data analysis
- Time-series pressure-data visualisation
- Recording-level quality checking
- Reproducible analysis workflow design

---

## Investigative Data Preparation with Pandas

### Project Summary

This project uses Pandas to clean, combine, and analyse structured tabular data in an investigative scenario. The workflow prepares suspect and transaction datasets, identifies missing values and outliers, engineers new features, aggregates transaction behaviour, and exports cleaned datasets for further analysis.

### Problem

Raw tabular data is often incomplete, inconsistent, or spread across multiple files. Before meaningful analysis can take place, the data must be inspected, cleaned, transformed, and combined into a reliable master dataset. This project focuses on building a clear data preparation pipeline using Python and Pandas.

### Data

The project uses structured CSV files containing suspect records and transaction records. The analysis combines personal attributes, transaction counts, total spending, transaction categories, and date-based features.

### Methods

- Loaded multiple CSV datasets using Pandas
- Inspected data structure, data types, missing values, and summary statistics
- Converted transaction dates into datetime format
- Detected missing transaction categories
- Used interquartile range analysis to identify outlier transaction amounts
- Aggregated transactions by suspect ID
- Created new fields such as total spending and number of transactions
- Merged suspect and transaction summaries into a master dataset
- Engineered additional features including BMI, first name, surname, and transaction day name
- Filtered records based on investigation criteria
- Exported cleaned outputs including `master_suspects.csv` and `motive_suspects.csv`

### Tools

Python, Pandas, NumPy, Jupyter Notebook, CSV

### Outcome

The project produced cleaned and structured datasets that could support further exploratory analysis or modelling. It demonstrates practical data wrangling skills, including joins, aggregation, missing-value checks, feature engineering, filtering, and reproducible export of processed data.

### What This Project Demonstrates

- Data cleaning with Pandas
- Data type conversion
- Missing-value inspection
- Outlier detection
- Grouped aggregation
- Dataset merging
- Feature engineering
- Exporting analysis-ready datasets

---

## Classifying Human Interaction Patterns in Inertial Sensor Data

### Project Summary

This machine learning project classifies human activity patterns using inertial sensor data from the UCI Human Activity Recognition dataset. A Random Forest classifier was trained to distinguish between six activities: walking, walking upstairs, walking downstairs, sitting, standing, and laying.

### Problem

Human activity recognition is useful for haptic systems, wearable devices, mobile interaction, and robotics. The project investigates whether high-dimensional accelerometer and gyroscope features can accurately distinguish between static and dynamic human movement patterns, and which inertial features are most predictive.

### Data

The project uses the UCI Human Activity Recognition dataset. The data was collected from 30 participants aged 19 to 48 wearing a Samsung Galaxy S II smartphone on the waist. The smartphone captured 3-axis linear acceleration and 3-axis angular velocity at 50 Hz.

The dataset contains 561 pre-extracted time-domain and frequency-domain features. It is split by participant, with 70% of volunteers used for training and 30% used for testing on unseen individuals.

### Methods

- Loaded training and testing datasets using Pandas
- Checked dataset structure and missing values
- Separated feature columns from activity labels
- Encoded categorical activity labels using label encoding
- Analysed activity distribution across the training data
- Generated a correlation heatmap to inspect feature redundancy
- Applied variance threshold checking to identify constant features
- Trained a Random Forest classifier with 100 estimators
- Evaluated predictions using accuracy, classification report, and confusion matrix
- Used feature-importance analysis to identify the most predictive inertial features

### Tools

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Jupyter Notebook

### Results

The Random Forest model achieved approximately 93% accuracy on the unseen test set. The model performed strongly when distinguishing dynamic movements from static postures. It also classified laying very effectively because the gravity orientation signal is highly distinctive.

The main source of error was confusion between sitting and standing. These two activities have similar waist-level sensor orientation, making them harder to separate using the available statistical features alone.

Feature-importance analysis showed that gravity-based orientation features were among the strongest predictors. This suggests that tilt-sensing and movement-intensity signals are important for future haptic and robotic systems.

### What This Project Demonstrates

- Supervised machine learning
- Human activity recognition
- Sensor-data classification
- High-dimensional feature analysis
- Random Forest modelling
- Confusion matrix interpretation
- Feature-importance analysis
- Translating model results into design implications

---

## Skills Demonstrated Across the Portfolio

- Python programming
- Data cleaning and preprocessing
- Exploratory data analysis
- Feature engineering
- Sensor-data analysis
- Supervised machine learning
- Classification model evaluation
- Data visualisation
- Research interpretation
- Reproducible analysis with Jupyter Notebook
