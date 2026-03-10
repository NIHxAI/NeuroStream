# Brain Volume Visualization Tool (Neuro Stream)

A Streamlit-based interactive visualization tool for exploring brain regional volume data.
This application supports dataset upload, cohort-wise comparison, preprocessing (including ComBat harmonization),
and multiple visualization modules such as boxplots, scatterplots, and PCA.

---

## Overview

This tool is designed for exploratory analysis and comparison of brain regional volume measurements
(e.g., FreeSurfer-derived volumes) across multiple cohorts and datasets.

The typical workflow is as follows:

1. **Upload**: Upload one or more CSV datasets via the 'Upload' tab.
2. **Convert**: Automatically convert uploaded CSV files into internal Feather (`.f`) format for high performance.
3. **Select**: Choose cohorts and data subsets for analysis.
4. **Preprocess**: Apply normalization or **ComBat harmonization** using customized covariates.
5. **Visualize**: Explore data distributions, statistical summaries, and multivariate structures.

The application is implemented using **Streamlit** and modularized into `app.py`, `deta.py`, and `util.py`.

---

## Key Features (2026 Update)

- **Smart Variable Classification**:
  - **`cov_` Prefix Rule**: Automatically identifies columns starting with `cov_` (e.g., `cov_gender`, `cov_edu`) as categorical covariates.
  - **Brain Region Filtering**: Automatically isolates numeric brain volume columns (ending in numbers) for X/Y axis selection to keep the UI clean.
- **Enhanced ComBat Harmonization**:
  - Supports multi-center batch-effect correction.
  - **Automatic Default Selection**: Pre-selects `age` and `gender` covariates when ComBat is enabled.
  - **Clean UI Labels**: Automatically hides technical prefixes like `cov_` in dropdowns and charts (e.g., `cov_gender` appears as `Gender`).
- **Interactive Visualization**:
  - **Boxplots**: Group-wise distribution with automated T-test/ANOVA results.
  - **Scatterplots**: Relationship mapping with LOWESS trajectory fitting.
  - **PCA**: Exploration of multivariate cohort separation.

---

## Project Structure

```text
.
├── app.py              # Main Streamlit application (UI and workflow control)
├── deta.py             # Data loading, preprocessing, and covariate mapping logic
├── util.py             # Visualization and statistical utility functions
├── assets/
│   ├── *.f             # Converted Feather-format datasets (zstd compressed)
│   └── *.png           # Brain region images used in visualization
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
001,Cohort1,72,M,3120.5,2987.3
002,Cohort2,68,F,3255.2,3101.8
```

---

## Preprocessing Methods

The following preprocessing methods are available in the application to ensure data quality and comparability across different cohorts:

- **None**: Use raw volume values without any transformation.
- **Scale (Z-score)**: Standardize features by removing the mean and scaling to unit variance (Z-score normalization).
- **Log Transform (log1p)**: Apply $log(1+x)$ transformation to reduce the impact of right-skewed distributions.
- **Log Transform + Z-score**: Successive application of logarithmic transformation followed by Z-score normalization.
- **Divide by Intracranial Volume (ICV)**: Regional brain volumes are divided by the total intracranial volume to account for individual differences in head size.
- **ComBat**: A powerful harmonization technique to remove cohort-related batch effects while preserving biological variance (e.g., Age and Gender).

> **Note:** ComBat harmonization requires at least two cohorts to be selected. The tool automatically pre-selects `age` and `gender` as default covariates when this method is enabled.

---

## Visualization Modules

The tool provides several interactive modules to explore and validate your neuroimaging data:

- **Boxplot** Compare the distribution of brain regional volumes across cohorts. This module includes automated statistical summaries (Mean, Median, Std) and performs T-test/ANOVA to identify significant differences.

- **Scatterplot** Visualize the relationship between brain volumes and demographic variables (primarily Age). It features **LOWESS (Locally Weighted Scatterplot Smoothing)** trajectory fitting to show trends across different cohorts.

- **PCA (Principal Component Analysis)** Explore the multivariate structure of the data in a reduced-dimensional space. This is particularly useful for detecting cohort-wise clusters and verifying the effectiveness of harmonization (e.g., seeing clusters merge after ComBat).

All visualizations are interactive, powered by Plotly and Seaborn, and update dynamically based on your sidebar selections.

---

## Notes & Limitations

- **Data Merging**: If multiple datasets are uploaded, they are concatenated row-wise before analysis.
- **Batch Definition**: In the context of ComBat, the `cohort` label is strictly treated as the batch effect identifier.
- **Label Sanitization**: For a cleaner UI, the application automatically strips the `cov_` prefix from labels in charts and legends (e.g., `cov_gender` is displayed as `Gender`).
- **Performance**: While the tool uses optimized Feather (`.f`) formats, extremely large datasets or high-resolution images may impact the loading speed of interactive plots.

---

## License

This project is made available for research and academic use under the policies and guidelines of the **Korea National Institutes of Health (KNIH)**.

