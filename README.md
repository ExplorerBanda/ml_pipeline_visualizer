# ML Pipeline Visualizer

An automated Machine Learning pipeline builder that analyzes datasets, suggests preprocessing steps, applies transformations, and compares multiple models through cross validation. Built as a portfolio project and final year project to demonstrate practical ML engineering skills.

---

## Project Overview

ML Pipeline Visualizer is a tool designed to simplify the early stages of a Machine Learning workflow. Instead of manually exploring data, cleaning it, selecting models, and evaluating performance, this system automates the process and presents insights through an interactive interface.

The system performs:

• Data profiling
• Data quality analysis
• Automated preprocessing suggestions
• Data transformation
• Model selection and evaluation
• Cross validation based comparison

The goal of this project is to reduce the time required to understand a dataset and identify suitable ML approaches.

---

## Features

### Data Analysis

Automatically analyzes uploaded datasets and extracts:

• Number of rows and columns
• Missing values
• Duplicate rows
• Numeric vs categorical columns
• Unique value counts
• Identifier columns
• Low and high cardinality detection

### Data Quality Report

Provides insights such as:

• Column wise null values
• Null percentages
• Dataset cleanliness summary

### Automated Preprocessing Suggestions

The system recommends preprocessing actions such as:

• Dropping identifier columns
• Handling missing values
• Encoding categorical variables
• Feature scaling
• Handling high cardinality columns

### Preprocessing Execution

With one click the system can:

• Drop unnecessary columns
• Fill numeric values with mean
• Fill categorical values with mode
• Apply encoding
• Apply MinMax scaling

### Model Evaluation

The system automatically:

• Detects classification or regression problems
• Selects suitable ML models
• Runs cross validation
• Compares performance scores
• Identifies the best model

Models currently supported:

• Logistic Regression
• Random Forest
• K Nearest Neighbour

---

## Tech Stack

Python
Pandas
Scikit learn
Streamlit

---

## Project Structure

```
ml_pipeline_visualizer/

Modules/
│
├── data_analyzer.py
├── preprocessing.py
├── model_trainer.py
│
├── app.py

train.csv

README.md
```

### Module Description

data_analyzer.py
Handles dataset profiling and preprocessing decision logic.

preprocessing.py
Applies recommended preprocessing steps.

model_trainer.py
Handles model selection, cross validation, and comparison.

app.py
Streamlit interface that connects all modules into one workflow.

---

## How It Works

Pipeline flow:

```
Dataset Upload
      ↓
Data Analysis
      ↓
Preprocessing Suggestions
      ↓
Preprocessing Execution
      ↓
Target Selection
      ↓
Model Evaluation
      ↓
Best Model Selection
```

---

## Installation

Clone the repository:

```
git clone https://github.com/yourusername/ml_pipeline_visualizer.git
```

Navigate into project:

```
cd ml_pipeline_visualizer
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the application:

```
streamlit run app.py
```

---

## Usage

1 Upload a CSV dataset
2 Review dataset analysis
3 Check preprocessing suggestions
4 Run preprocessing
5 Select target column
6 Run model comparison
7 View best performing model

---

## Future Improvements

Planned improvements include:

• Feature importance visualization
• Model persistence
• Prediction module
• Hyperparameter tuning
• Exportable reports
• Additional ML models

---

## Learning Outcomes

This project helped demonstrate practical understanding of:

• Data preprocessing workflows
• ML pipeline architecture
• Cross validation strategies
• Model comparison techniques
• UI integration with ML systems
• Modular software design

---

## Author

Yash Rawat

Machine Learning enthusiast focused on building practical ML systems and learning by creating real projects.

---

## License

This project is for educational and portfolio purposes.

