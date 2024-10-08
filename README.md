
# Oil Well Production Forecasting using Machine Learning

## Overview

This project aims to predict oil well production rates based on well characteristics such as well depth, reservoir pressure, oil density, and temperature using machine learning techniques. By leveraging data science and predictive modeling, this project demonstrates how machine learning can be applied to optimize production processes in the oil and gas industry.

## Project Objectives

- To develop a machine learning model to forecast oil production rates.
- To explore the relationship between well characteristics and production levels.
- To apply feature engineering and hyperparameter tuning to improve the model's accuracy.

## Dataset

This project uses synthetic data generated to simulate oil well characteristics. The dataset includes the following features:

- **Well Depth (meters)**: The depth of the well.
- **Reservoir Pressure (bars)**: The pressure in the reservoir.
- **Oil Density (g/cm³)**: The density of the oil being extracted.
- **Temperature (Celsius)**: The temperature at the well.
- **Production Rate (barrels per day)**: The target variable representing the daily oil production rate.

You can adapt this project to real-world datasets such as those provided by public sources like the **Society of Petroleum Engineers (SPE)** or **Kaggle**.

## Methodology

1. **Data Preprocessing**:
   - Generated synthetic data for the well features.
   - Split the data into training and test sets.
   - Scaled the features to ensure proper model performance.

2. **Feature Engineering**:
   - Engineered features that represent the characteristics of oil wells, using synthetic data.
   - Considered well depth, reservoir pressure, oil density, and temperature as the primary input features.

3. **Modeling**:
   - Implemented a **Random Forest Regressor** for predicting production rates.
   - Performed hyperparameter tuning using **RandomizedSearchCV** to optimize model performance.

4. **Evaluation**:
   - Evaluated the model using **Mean Squared Error (MSE)** and **R-squared** to measure the accuracy of the predictions.
   - Compared the predicted production rates against the actual values.

## Results

- **Best Hyperparameters**:
  - `n_estimators`: 100
  - `min_samples_split`: 2
  - `max_features`: 'log2'
  - `max_depth`: None

- **Model Performance**:
  - **Mean Squared Error (MSE)**: 145.84
  - **R-squared**: 0.783

These metrics indicate that the model explains approximately 78% of the variance in the production rates, showing a good fit but with potential for improvement.

## How to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/oil-well-production-forecasting.git
   ```

2. **Install the required Python packages**:
   Ensure you have Python installed and then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**:
   Launch the Jupyter Notebook to view the code and results:
   ```bash
   jupyter notebook Oil_Well_Production_Forecasting.ipynb
   ```

## Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `statsmodels`

You can install all dependencies by running:
```bash
pip install -r requirements.txt
```

## Future Work

- **Advanced Modeling**: Experiment with models such as **XGBoost**, **LightGBM**, or **Gradient Boosting** to potentially improve performance.
- **Feature Engineering**: Introduce interaction features between well characteristics and apply log transformations to normalize the data.
- **Use Real Data**: Apply the methodology to real-world datasets from public oil production data repositories.
- **Deploy the Model**: Package the model and deploy it via a web API using Flask or FastAPI for real-time production rate predictions.

## Conclusion

This project demonstrates the application of Random Forest regression for oil well production forecasting, providing insights into how machine learning can improve decision-making in the oil and gas industry. Future improvements in feature engineering, advanced modeling, and real-world data could enhance its accuracy and usefulness.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by open-source oil and gas datasets.
- Special thanks to the data science community for providing resources and tutorials.

---

This `README.md` provides an overview of your project, how to run it, the methods used, and the potential future work, making it ready for a professional GitHub portfolio. Let me know if you'd like to modify or expand any section!
