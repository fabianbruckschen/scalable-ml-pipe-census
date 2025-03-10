# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- XGBoost Classifier model
- Hyperparameters:
  - max_depth: 6
  - learning_rate: 0.1
  - n_estimators: 100
  - objective: binary:logistic
- The model predicts whether a person's income is above or below $50K/year

## Intended Use
- This model is intended to predict income levels based on census data
- Primary intended users would be researchers and policy makers studying income demographics
- Not suitable for making individual decisions about loan approvals, employment, or other high-stakes decisions

## Training Data
- Source: Census Income Dataset
- Features include:
  - Numerical: age, education-num, capital-gain, capital-loss, hours-per-week
  - Categorical: workclass, education, marital-status, occupation, relationship, race, sex, native-country
- Binary classification target: income <=50K or >50K
- Data contains demographic and employment information
- Dataset shows some imbalance with majority of samples in <=50K category

## Evaluation Data
- Random split from the original dataset
- Same feature distribution as training data
- Maintains original demographic proportions

## Metrics
The model is evaluated using:
- Precision: Accuracy of positive predictions
- Recall: Ability to find all positive instances
- F1-beta Score: Harmonic mean of precision and recall
- Metrics are computed for:
  - Overall model performance
  - Slice-based evaluation on categorical features (with minimum 30 samples per slice)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.90      | 0.95   | 0.92     | 4942    |
| 1     | 0.79      | 0.67   | 0.73     | 1571    |

Note:
- Class 0 represents income <=50K
- Class 1 represents income >50K
- Support shows the number of samples in each class

## Ethical Considerations
- Dataset contains sensitive demographic information (race, sex, nationality)
- Potential for perpetuating historical biases present in census data
- Model predictions could reinforce existing socioeconomic disparities
- Care should be taken when using predictions that could affect individual opportunities

## Caveats and Recommendations
- Model should not be used for individual decision-making
- Regular monitoring for prediction bias across demographic groups recommended
- Some categories (like certain nationalities or racial groups) may have limited representation
- Missing values in some features (marked as '?') may affect prediction quality
- Consider updating with more recent census data as social and economic conditions change
- Recommend performing regular bias audits across different demographic slices