# A/B Testing of Marketing Campaigns

## Project Description
This project focuses on conducting A/B testing to evaluate the effectiveness of two marketing campaigns. Using statistical analysis and hypothesis testing, we determine which campaign is more effective in improving conversion rates.

## Project Structure
The project consists of three main parts, each represented in a corresponding Jupyter Notebook:

1. **01_data_preprocessing.ipynb**: Data loading and preprocessing.
2. **02_stat_analysis.ipynb**: Statistical analysis and visualization of results.
3. **03_conclusion.ipynb**: Conclusions and recommendations based on the analysis.

## Files
- **datasets/control_group.csv**: Data for the control group.
- **datasets/test_group.csv**: Data for the test group.
- **datasets/control_i.csv**: Processed data for the control group.
- **datasets/test_i.csv**: Processed data for the test group.
- **functions.py**: File containing functions for data processing and analysis.

## Key Skills
- **Data Processing**: Using Pandas and NumPy libraries for data loading, cleaning, and preprocessing.
- **Statistical Analysis**: Performing hypothesis testing (t-test) to assess statistical significance between groups.
- **Data Visualization**: Utilizing Matplotlib and Seaborn libraries to create graphs and visualize results.
- **Data Consistency Checks**: Verifying data for logical consistency and correcting anomalies.

## Findings
Based on the analysis, we concluded that the new (test) marketing campaign is more effective compared to the control campaign. Key insights include:

- The conversion rate for the control group was **1.23%**, while for the test group, it reached **2.54%**.
- The test group was **2.07 times more effective** than the control group.
- Statistical significance was confirmed via a t-test, showing a p-value of **0.001**, indicating that the improvement in conversion rate is not random.
![Histogram of Conversion Comparison](https://github.com/user-attachments/assets/889afde6-8350-481b-ab52-9a1669002553).

## Recommendations
Based on the results, it is recommended to adopt the test campaign as the primary marketing strategy to optimize user engagement and improve conversions.

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/ituvtu/DataMining-AB-Testing
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
3. Launch Jupyter Notebook and open the following files for execution and analysis:
- 01_data_preprocessing.ipynb
- 02_stat_analysis.ipynb
- 03_conclusion.ipynb

## Author
ituvtu ([LinkedIn](https://www.linkedin.com/in/ivanturenko/)).


## This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
