R version used: these codes run well on R version 3.5.1 (64-bit)

Dataset: Indian Liver Patient Records from https://www.kaggle.com/uciml/indian-liver-patient-records

Objective: Predict liver disease based of the following 10 independent variables: a.) Age
of the patient; b.) Gender of the patient; c.) Total Bilirubin; d.) Direct Bilirubin; e.) Alkaline Phosphotase; f.) Alamine Aminotransferase; g.) Aspartate Aminotransferase; h.) Total Protiens; i.) Albumin; and j.) Albumin and Globulin Ratio

Exploratory data analysis conducted: generated summary statistics (xda) and visualization; checked for correlation, principal components, variable importance, multi-collinearity (variance inflation factor), normality, linearity, and outliers (chi-squared test)

Treatment of missing values: initially imputed the 4 missing values with plausible substitutes using multivariate imputation by chained equations (MICE) using the mice library, but then finally decided to simply remove the 4 records since they only comprise 0.69% of the original dataset.

Machine learning algorithms implmented: logistic regression, neural network/deep learning, random forests, gradient boosting machine, support vector machine, naive bayes, classification tree, C5.0 classification tree, evolutionary classification tree, logistic model-based recursive partitioning, principal components analysis (PCA), and principal components analysis - singular value decomposition (PCA-SVD) method

Model performance measured: accuracy, sensitivity, specificity, and precision
