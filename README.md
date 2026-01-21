# Brain Volume Visualization Tool

A Streamlit-based interactive visualization tool for exploring brain regional volume data.
This application supports dataset upload, cohort-wise comparison, preprocessing (including ComBat harmonization),
and multiple visualization modules such as boxplots, scatterplots, and PCA.

---

## Overview

This tool is designed for exploratory analysis and comparison of brain regional volume measurements
(e.g., FreeSurfer-derived volumes) across multiple cohorts and datasets.

The typical workflow is as follows:

1. Upload one or more CSV datasets
2. Automatically convert uploaded CSV files into internal Feather (`.f`) format
3. Select one or more datasets for analysis
4. Select multiple cohorts for comparison
5. Apply preprocessing methods (scaling, transformation, harmonization)
6. Visualize data distributions and multivariate structure

The application is implemented using **Streamlit** and modularized into `app.py`, `deta.py`, and `util.py`.

---

## Features

- CSV dataset upload and automatic conversion to Feather (`.f`) format
- Multi-dataset selection and merging
- Multi-cohort comparison
- Preprocessing options:
  - No preprocessing (raw values)
  - Standard scaling (z-score normalization)
  - Power transformation
  - **ComBat harmonization for batch-effect correction**
- Visualization modules:
  - Boxplot
  - Scatterplot
  - Principal Component Analysis (PCA)
- Interactive web-based interface built with Streamlit

---

## Project Structure

```
.
├── app.py              # Main Streamlit application (UI and workflow control)
├── deta.py             # Data loading, preprocessing, and cohort handling logic
├── util.py             # Visualization and statistical utility functions
├── assets/
│   ├── *.csv            # Input datasets (user-provided)
│   ├── *.f              # Converted Feather-format datasets
│   └── *.png            # Brain region images used in visualization
├── requirements.txt    # Python dependencies
└── README.md
```

---

## Requirements

- Python 
- streamlit
- pandas
- numpy
- scikit-learn
- scipy
- pycombat
- matplotlib
- seaborn

All required packages are listed in `requirements.txt`.

---

## Installation

Clone the repository and create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

After running the command, open the provided local URL (e.g., `http://localhost:8501`) in your web browser.

---

## Input Data Format

Input CSV files must contain the following information.

### Required columns

- `cohort`  : cohort or study identifier (used for comparison and ComBat harmonization)
- `age`     : age of the subject
- `sex` or `gender` : biological sex

### Brain volume columns

- Numeric columns representing regional brain volume measurements
  (e.g., FreeSurfer-derived cortical or subcortical volumes)

### Example

```csv
subject_id,cohort,age,sex,Left-Hippocampus,Right-Hippocampus
001,BICWALZS,72,M,3120.5,2987.3
002,KoGES,68,F,3255.2,3101.8
```

---

## Preprocessing Methods

The following preprocessing methods are available in the application:

- **None**: Use raw values without preprocessing
- **StandardScaler**: Z-score normalization across subjects
- **Log transform**: Logarithmic transformation to reduce right-skewed distributions
- **Log transform + Z-score**: Logarithmic transformation followed by z-score normalization
- **ICV normalization**: Regional brain volumes divided by intracranial volume (ICV)
- **ComBat**: Harmonization to remove cohort-related batch effects

> **Note:** ComBat requires at least two cohorts to be selected.

---

## Visualization Modules

- **Boxplot**  
  Compare the distribution of brain regional volumes across cohorts.

- **Scatterplot**  
  Visualize relationships between brain volumes and demographic variables
  (e.g., age or sex), stratified by cohort.

- **PCA (Principal Component Analysis)**  
  Explore multivariate structure and cohort separation in reduced-dimensional space.

All visualizations are interactive and dynamically updated based on user selections.

---

## Notes & Limitations

- Selected datasets are concatenated before analysis.
- Cohort labels are assumed to represent batch effects when using ComBat.
- Large datasets may increase loading time and affect interactive performance.

---

## License

This project is made available for research and academic use under the policies and guidelines of the **National Institutes of Health (NIH)**.

