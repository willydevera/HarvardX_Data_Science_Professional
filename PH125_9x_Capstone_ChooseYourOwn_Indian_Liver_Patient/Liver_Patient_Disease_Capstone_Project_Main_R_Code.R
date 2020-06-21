####################################################
#
# HarvardX Professional Certificate in Data Science
#
####################################################
#
# Harvardx: PH125.9x: Data Science: Capstone
# Capstone Project: Choose Your Own: Predicting Indian Liver Disease
# Wilfredo A. de Vera
# Date submitted: June 21, 2020

###################################################
###################################################

# load libraries
# (Please refer to Annex 1 below to install packages not currently installed)
library(tidyverse)
library(caret)
library(data.table)
library(xda)
library(mice)
library(ggplot2)
library(knitr)
library(h2o)
library(earth)
library(car)
library(C50)
library(e1071)
library(mlbench)
library(evtree)
library(NeuralNetTools)
library(reshape2)


################## Step 1. Data Load

# Read the "indian_liver_patient.csv" dataset from my github subfolder
# and automatically save it into your working directory using this code:

url <- "https://github.com/willydevera/HarvardX_Data_Science_Professional/raw/master/PH125_9x_Capstone_ChooseYourOwn_Indian_Liver_Patient/indian_liver_patient.csv"

path <- paste(getwd(), "indian_liver_patient.csv", sep="/")

if (!file.exists(path)) {
  download.file(url, path)
}

liver <- read.csv(path)

# Or you could download directly from source: https://www.kaggle.com/uciml/indian-liver-patient-records
# and manually save into your working directory and then read the data using this code:
# path <- paste(getwd(), "indian_liver_patient.csv", sep="/")
# liver <- read.csv(path)

# remove unnecessary temorary files
rm(url, path)


################## Step 2. Exploratory Data Analysis (EDA) and Wrangling

## Initial data exploration

str(liver)
# data.frame':	583 obs. of  11 variables:
#$ Age                       : int  65 62 62 58 72 46 26 29 17 55 ...
#$ Gender                    : Factor w/ 2 levels "Female","Male": 1 2 2 2 2 2 1 1 2 2 ...
#$ Total_Bilirubin           : num  0.7 10.9 7.3 1 3.9 1.8 0.9 0.9 0.9 0.7 ...
#$ Direct_Bilirubin          : num  0.1 5.5 4.1 0.4 2 0.7 0.2 0.3 0.3 0.2 ...
#$ Alkaline_Phosphotase      : int  187 699 490 182 195 208 154 202 202 290 ...
#$ Alamine_Aminotransferase  : int  16 64 60 14 27 19 16 14 22 53 ...
#$ Aspartate_Aminotransferase: int  18 100 68 20 59 14 12 11 19 58 ...
#$ Total_Protiens            : num  6.8 7.5 7 6.8 7.3 7.6 7 6.7 7.4 6.8 ...
#$ Albumin                   : num  3.3 3.2 3.3 3.4 2.4 4.4 3.5 3.6 4.1 3.4 ...
#$ Albumin_and_Globulin_Ratio: num  0.9 0.74 0.89 1 0.4 1.3 1 1.1 1.2 1 ...
#$ Dataset                   : int  1 1 1 1 1 1 1 1 2 1 ...

# number of no disease and diseased persons
table(disease = as.factor(ifelse(liver$Dataset==1, "N", "Y")))
#  disease
#   N   Y 
# 416 167 

# Note that Dataset is the dependent variable; i.e  1 - no disease; 2 - disease


# number of no disease and diseased persons by gender

table(liver$Gender, disease = as.factor(ifelse(liver$Dataset==1, "N", "Y")))
#         disease
#         N   Y
# Female  92  50
# Male   324 117

# Out of 583 cases, there were a total of 167 cases of liver disease.
# Of the 167 liver disease cases, 50 were females; 117 were males


## 2.1 Generate summary statistics

# install xda
# reference: https://github.com/ujjwalkarn/xda
# library(devtools)
# install_github("ujjwalkarn/xda")
library(xda)
numSummary(liver)
#                             n    mean      sd    max  min  range nunique nzeros   iqr lowerbound upperbound noutlier kurtosis
#Age                        583  44.746  16.190   90.0  4.0   86.0      72      0  25.0      -4.50      95.50        0   -0.574
#Total_Bilirubin            583   3.299   6.210   75.0  0.4   74.6     113      0   1.8      -1.90       5.30       84   36.699
#Direct_Bilirubin           583   1.486   2.808   19.7  0.1   19.6      80      0   1.1      -1.45       2.95       81   11.196
#Alkaline_Phosphotase       583 290.576 242.938 2110.0 63.0 2047.0     263      0 123.0      -9.00     482.50       66   17.520
#Alamine_Aminotransferase   583  80.714 182.620 2000.0 10.0 1990.0     152      0  37.2     -32.88     116.38       73   49.954
#Aspartate_Aminotransferase 583 109.911 288.919 4929.0 10.0 4919.0     177      0  62.0     -68.00     180.00       66  149.095
#Total_Protiens             583   6.483   1.085    9.6  2.7    6.9      58      0   1.4       3.70       9.30        8    0.210
#Albumin                    583   3.142   0.796    5.5  0.9    4.6      40      0   1.2       0.80       5.60        0   -0.404
#Albumin_and_Globulin_Ratio 579   0.947   0.320    2.8  0.3    2.5      70      0   0.4       0.10       1.70       10    3.222
#Dataset                    583   1.286   0.452    2.0  1.0    1.0       2      0   1.0      -0.50       3.50        0   -1.114

#                           skewness  mode miss miss%     1%     5%   25%    50%   75%    95%     99%
#Age                         -0.0292  60.0    0 0.000  9.640  18.00  33.0  45.00  58.0  72.00   75.00
#Total_Bilirubin              4.8823   0.8    0 0.000  0.582   0.60   0.8   1.00   2.6  16.35   28.20
#Direct_Bilirubin             3.1959   0.2    0 0.000  0.100   0.10   0.2   0.30   1.3   8.40   12.96
#Alkaline_Phosphotase         3.7458 198.0    0 0.000 97.820 137.00 175.5 208.00 298.0 698.10 1555.40
#Alamine_Aminotransferase     6.5155  25.0    0 0.000 11.820  15.00  23.0  35.00  60.5 232.00 1004.00
#Aspartate_Aminotransferase  10.4920  23.0    0 0.000 12.000  15.10  25.0  42.00  87.0 400.90  976.20
#Total_Protiens              -0.2842   7.0    0 0.000  3.682   4.61   5.8   6.60   7.2   8.10    8.62
#Albumin                     -0.0435   3.0    0 0.000  1.482   1.80   2.6   3.10   3.8   4.39    4.90
#Albumin_and_Globulin_Ratio   0.9872   1.0    4 0.686  0.386   0.50   0.7   0.93   1.1   1.50    1.81
#Dataset                      0.9423   1.0    0 0.000  1.000   1.00   1.0   1.00   2.0   2.00    2.00


charSummary(liver)
#          n miss miss% unique     top5levels:count
# Gender 583    0     0      2 Male:441, Female:142


## 2.2 Visualization of the original liver dataset

# Since the variables Direct_Bilirubin, Alamine_Aminotransferase, Age, and Alkaline_Phosphotase were determined
# to be important in as far as the earth package is concerned, we will limit our visualization to these 4 variables,
# instead of trying to visualize all the 10 independent variables.
library(ggplot2)

## 2.2.1 Plot of Direct_Bilirubin vs. Alamine_Aminotransferase
liver %>%
  mutate(disease = as.factor(ifelse(liver$Dataset==1, "N", "Y"))) %>%
  ggplot() + 
  geom_point(aes(x = Direct_Bilirubin,
                 y = Alamine_Aminotransferase,
                 colour=disease)) +
  # coord_trans(x = "log10") + # because Direct_Bilirubin ranges from 0.1 to 19.7
  coord_trans(y = "log10") +   # because Alamine_Aminotransferase ranges from 10 to 2000
  xlab('Direct_Bilirubin') +
  ylab('Alamine_Aminotransferase') +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("Scatterplot of Direct_Bilirubin vs. Alamine_Aminotransferase")


## 2.2.2 Plot of Direct_Bilirubin vs. Age
liver %>%
  mutate(disease = as.factor(ifelse(liver$Dataset==1, "N", "Y"))) %>%
  ggplot() + 
  geom_point(aes(x = Direct_Bilirubin,
                 y = Age,
                 colour=disease)) +
  # coord_trans(x = "log10") +   # because Direct_Bilirubin ranges from 0.1 to 19.7
  coord_trans(y = "log10") +     # because Age ranges from 4 to 90
  xlab('Direct_Bilirubin') +
  ylab('Age') +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("Scatterplot of Direct_Bilirubin vs. Age")


## 2.2.3 Plot of Direct_Bilirubin vs. Alkaline_Phosphotase
liver %>%
  mutate(disease = as.factor(ifelse(liver$Dataset==1, "N", "Y"))) %>%
  ggplot() + 
  geom_point(aes(x = Direct_Bilirubin,
                 y = Alkaline_Phosphotase,
                 colour=disease)) +
  # coord_trans(x = "log10") + # because Direct_Bilirubin ranges from 0.1 to 19.7
  coord_trans(y = "log10") +   # because Alkaline_Phosphotase ranges from 63  to 2110
  xlab('Direct_Bilirubin') +
  ylab('Alkaline_Phosphotase') +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("Scatterplot of Direct_Bilirubin vs. Alkaline_Phosphotase")


## 2.2.4 Plot of Alamine_Aminotransferase vs. Age
liver %>%
  mutate(disease = as.factor(ifelse(liver$Dataset==1, "N", "Y"))) %>%
  ggplot() + 
  geom_point(aes(x = Alamine_Aminotransferase,
                 y = Age,
                 colour=disease)) +
  coord_trans(x = "log10") +     # because Alamine_Aminotransferase ranges from 10 to 2000 
  # coord_trans(y = "log10") +   # because Age ranges from 4 to 90
  xlab('Alamine_Aminotransferase') +
  ylab('Age') +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("Scatterplot of Age vs. Alamine_Aminotransferase")


## 2.2.5 Plot of Alamine_Aminotransferase vs. Alkaline_Phosphotase
liver %>%
  mutate(disease = as.factor(ifelse(liver$Dataset==1, "N", "Y"))) %>%
  ggplot() + 
  geom_point(aes(x = Alamine_Aminotransferase,
                 y = Alkaline_Phosphotase,
                 colour=disease)) +
  coord_trans(x = "log10") +     # because Alamine_Aminotransferase ranges from 10 to 2000 
  # coord_trans(y = "log10") +   # because Alkaline_Phosphotase ranges from 63 to 2110
  xlab('Alamine_Aminotransferase') +
  ylab('Alkaline_Phosphotase') +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("Scatterplot of Alamine_Aminotransferase vs. Alkaline_Phosphotase")


## 2.2.6 Plot of Age vs. Alkaline_Phosphotase
liver %>%
  mutate(disease = as.factor(ifelse(liver$Dataset==1, "N", "Y"))) %>%
  ggplot() + 
  geom_point(aes(x = Age,
                 y = Alkaline_Phosphotase,
                 colour=disease)) +
  coord_trans(x = "log10") +   # because Age ranges from 4 to 90 
  coord_trans(y = "log10") +   # because Alkaline_Phosphotase ranges from 63 to 2110
  xlab('Age') +
  ylab('Alkaline_Phosphotase') +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("Scatterplot of Age vs. Alkaline_Phosphotase")



## 2.3 Data cleaning & wrangling

# Check for and handle NAs, if any
# if(!require(mice)) install.packages("mice", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(mice)

# check for NAs in liver
anyNA(liver)
# TRUE

# Note: there were 4 NAs in Albumin_and_Globulin_Ratio variable
liver %>% mutate(y = Dataset,
                 Gender = as.numeric(Gender)) %>%
  select(-c(Dataset)) %>%
  md.pattern(., plot=TRUE, rotate.names = TRUE)


# display specific records where Albumin_and_Globulin_Ratio==NA
liver %>% mutate(disease = as.factor(ifelse(liver$Dataset==1, "N", "Y")),
                 Gender = as.numeric(Gender)) %>%
  select(-c(Dataset)) %>%
  filter(is.na(Albumin_and_Globulin_Ratio)==TRUE)
#    Age Gender Total_Bilirubin Direct_Bilirubin Alkaline_Phosphotase Alamine_Aminotransferase Aspartate_Aminotransferase
# 1  45 Female             0.9              0.3                  189                       23                         33
# 2  51   Male             0.8              0.2                  230                       24                         46
# 3  35 Female             0.6              0.2                  180                       12                         15
# 4  27   Male             1.3              0.6                  106                       25                         54

#   Total_Protiens Albumin Albumin_and_Globulin_Ratio disease
# 1            6.6     3.9                         NA       N
# 2            6.5     3.1                         NA       N
# 3            5.2     2.7                         NA       Y
# 4            8.5     4.8                         NA       Y


################## Data Cleaning & Wrangling

## Clean the liver dataset of NAs
# As previously noted in Section 2.2, there were only 4 obs whose Albumin_and_Globulin_Ratio is NA,
# and that the proportion with respect to the dependent variable y are the same - i.e. 2 records each.
# Since the amount of missing data is small at 0.7%, we could just easily remove them,
# or during modeling, we could set na.action=na.omit for observations (rows) that contain missing values, if such
# setting is applicable at all.

prop.table(table(is.na(liver$Albumin_and_Globulin_Ratio)==TRUE))
#   FALSE    TRUE 
# 0.99314 0.00686 

liver %>% mutate(disease = as.factor(ifelse(liver$Dataset==1, "N", "Y")),
                 Gender = as.numeric(Gender)) %>%
  select(-c(Dataset)) %>%
  filter(is.na(Albumin_and_Globulin_Ratio)==TRUE) %>% pull(disease) %>% table()
# N Y 
# 2 2 

# Hence if we opt to remove the 4 records with Albumin_and_Globulin_Ratio = NA, the code should be:
liver_clean <- liver %>% filter(!is.na(Albumin_and_Globulin_Ratio==TRUE))

dim(liver_clean)
# 579  11

# check for NAs in liver_clean
anyNA(liver_clean)
# FALSE

str(liver_clean)
# data.frame':	579 obs. of  11 variables:
# however the number of obs is reduced from 583 to 579, because we removed the 4 records with missing values.


# But then, assuming that if we badly need the 4 observations and want them included in the dataset,
# and likewise on the assumption that the NAs are missing at random,
# then we could impute the 4 missing values with plausible substitutes.
# In this case we could use the mice library - Multivariate Imputation by Chained Equations (MICE)
# Reference: https://www.r-bloggers.com/imputing-missing-data-with-r-mice-package/

templiver <- mice(data=liver, m=10, method="rf", maxit=50, seed=1)
# where m = Number of multiple imputations
#       method = random forest as imputation method
#                note that predictive mean mapping (pmm) could likewise be used
#       maxit = number of iterations

templiver$imp$Albumin_and_Globulin_Ratio
#        1   2   3   4   5   6    7   8   9  10
# 210 1.00 1.1 1.0 1.1 1.5 1.2 1.00 1.3 1.5 1.2
# 242 0.50 0.8 1.0 1.0 1.0 1.0 1.34 0.9 0.9 0.8
# 254 1.51 1.0 0.8 1.5 0.7 1.0 1.00 0.7 0.8 0.8
# 313 1.00 1.0 1.2 1.0 1.0 1.0 1.00 1.8 1.0 1.0

# then we could complete the dataset using the complete() function using e.g. the 8th imputation
liver_imputed <- complete(templiver, 8)

# check for NAs in liver_imputed
anyNA(liver_imputed)
# FALSE

str(liver_imputed)
# data.frame':	583 obs. of  11 variables:
# so the number of obs is maintained at 583, because we imputed values of Albumin_and_Globulin_Ratio variable
# that initially had 4 records with missing values.

# remove unnecessary temporary dataframe templiver
rm(templiver)


# convert the variable Dataset from integer to factor disease with values "N" and Y"
# create & retain variable = Dataset as integer
# and do this across-the-board for liver, liver_clean, and liver_imputed datasets
# disease <- as.factor(ifelse(liver$Dataset==1, "N", "Y"))
# table(disease)

# liver dataset
liver <- liver %>% mutate(y = Dataset,
                          disease = as.factor(ifelse(Dataset==1, "N", "Y"))) %>%
  select(-c(Dataset))

# distribution of liver disease by gender on the liver dataset
table(liver$Gender, liver$disease)
#         N   Y
#Female  92  50
#Male   324 117


# liver_clean dataset
liver_clean <- liver_clean %>% mutate(y = Dataset,
                                      disease = as.factor(ifelse(Dataset==1, "N", "Y"))) %>%
  select(-c(Dataset))

# distribution of liver disease by gender on the liver_clean dataset
table(liver_clean$Gender, liver_clean$disease)
#         N   Y
#Female  91  49
#Male   323 116


# liver_imputed dataset
liver_imputed <- liver_imputed %>% mutate(y = Dataset,
                                          disease = as.factor(ifelse(Dataset==1, "N", "Y"))) %>%
  select(-c(Dataset))

# distribution of liver disease by gender on the liver_imputed dataset
table(liver_imputed$Gender, liver_imputed$disease)
#         N   Y
#Female  92  50
#Male   324 117



## 2.4 Correlation

liver_clean %>%
  mutate(Gender = as.numeric(Gender)) %>%
  select(-c(disease, y, Gender)) %>%
  scale() %>%
  as.matrix() %>%
  cor()
#                                Age Total_Bilirubin Direct_Bilirubin Alkaline_Phosphotase Alamine_Aminotransferase
#Age                         1.00000         0.01100         6.78e-03               0.0789                 -0.08780
#Total_Bilirubin             0.01100         1.00000         8.74e-01               0.2057                  0.21338
#Direct_Bilirubin            0.00678         0.87448         1.00e+00               0.2340                  0.23318
#Alkaline_Phosphotase        0.07888         0.20574         2.34e-01               1.0000                  0.12478
#Alamine_Aminotransferase   -0.08780         0.21338         2.33e-01               0.1248                  1.00000
#Aspartate_Aminotransferase -0.02050         0.23732         2.57e-01               0.1666                  0.79186
#Total_Protiens             -0.18625        -0.00791         3.27e-05              -0.0271                 -0.04243
#Albumin                    -0.26421        -0.22209        -2.28e-01              -0.1634                 -0.02866
#Albumin_and_Globulin_Ratio -0.21641        -0.20627        -2.00e-01              -0.2342                 -0.00237

#                           Aspartate_Aminotransferase Total_Protiens Albumin Albumin_and_Globulin_Ratio
#Age                                           -0.0205      -1.86e-01 -0.2642                   -0.21641
#Total_Bilirubin                                0.2373      -7.91e-03 -0.2221                   -0.20627
#Direct_Bilirubin                               0.2570       3.27e-05 -0.2284                   -0.20012
#Alkaline_Phosphotase                           0.1666      -2.71e-02 -0.1634                   -0.23417
#Alamine_Aminotransferase                       0.7919      -4.24e-02 -0.0287                   -0.00237
#Aspartate_Aminotransferase                     1.0000      -2.58e-02 -0.0849                   -0.07004
#Total_Protiens                                -0.0258       1.00e+00  0.7831                    0.23489
#Albumin                                       -0.0849       7.83e-01  1.0000                    0.68963
#Albumin_and_Globulin_Ratio                    -0.0700       2.35e-01  0.6896                    1.00000


## 2.5 Principal components

# for liver_clean: with 579 obs
liver_clean %>%
  mutate(Gender = as.numeric(Gender)) %>%
  select(-c(disease, y)) %>%
  prcomp(~ ., data=., scale=TRUE) %>%
  summary()
# Importance of components:
#                          PC1   PC2   PC3   PC4    PC5    PC6    PC7    PC8    PC9    PC10
#Standard deviation     1.666 1.424 1.170 1.030 0.9584 0.8963 0.8130 0.4511 0.3545 0.23519
#Proportion of Variance 0.278 0.203 0.137 0.106 0.0919 0.0803 0.0661 0.0203 0.0126 0.00553
#Cumulative Proportion  0.278 0.480 0.617 0.723 0.8151 0.8954 0.9616 0.9819 0.9945 1.00000

# The first 5 principal components account for 81.5% of the variability of the dataset


# for liver_imputed: with 583 obs
liver_imputed %>%
  mutate(Gender = as.numeric(Gender)) %>%
  select(-c(disease, y)) %>%
  prcomp(~ ., data=., scale=TRUE) %>%
  summary()
# Importance of components:
#                         PC1   PC2   PC3   PC4   PC5    PC6    PC7    PC8    PC9    PC10
#Standard deviation     1.668 1.425 1.169 1.029 0.959 0.8956 0.8097 0.4510 0.3543 0.23562
#Proportion of Variance 0.278 0.203 0.137 0.106 0.092 0.0802 0.0656 0.0203 0.0126 0.00555
#Cumulative Proportion  0.278 0.481 0.618 0.724 0.816 0.8960 0.9616 0.9819 0.9944 1.00000

# The first 5 principal components account for 81.6% of the variability


## 2.6 Variable importance
# We will use the earth package to determine important variables.
# if(!require(earth)) install.packages("earth", repos = "http://cran.us.r-project.org")
library(earth)

# scale all columns except Gender {col 2}, y {11}, and disease {12}
# liver_scaled <- liver_clean %>% mutate(Gender = as.numeric(Gender)) %>%
#   select(-c(2,12)) %>%
#   mutate_at(-c(10), funs(scale(.)))
# earth_liver <- earth(y ~ ., data=liver_scaled); earth_liver

liver_clean %>% mutate(Gender = as.numeric(Gender)) %>%
  select(-c(12)) %>%
  mutate_at(-c(2,11), funs(scale(.))) %>%
  earth(y ~ ., data=.) %>%
  evimp(trim=FALSE)
#                                  nsubsets   gcv    rss
#Direct_Bilirubin                         5 100.0  100.0
#Alamine_Aminotransferase                 4  52.2   60.2
#Age                                      3  33.3   44.1
#Alkaline_Phosphotase                     2  23.5   33.8
#Gender-unused                            0   0.0    0.0
#Total_Bilirubin-unused                   0   0.0    0.0
#Aspartate_Aminotransferase-unused        0   0.0    0.0
#Total_Protiens-unused                    0   0.0    0.0
#Albumin-unused                           0   0.0    0.0
#Albumin_and_Globulin_Ratio-unused        0   0.0    0.0

# Based on the earth library, only the variables:
# Direct_Bilirubin; Alamine_Aminotransferase; Age; and Alkaline_Phosphotase are important


## 2.7 Check for multi-collinearity - Variance Inflation Factor
# We will use the car package to determine redundant variables.
# if(!require(car)) install.packages("car", repos = "http://cran.us.r-project.org")
library(car)

# create train-test
library(caret)
set.seed(1)
test_index <- createDataPartition(y = liver_clean$disease, times = 1,
                                  p = 0.30, list = FALSE)
# on the liver_clean set
test_set <- liver_clean[test_index, ]
train_set <- liver_clean[-test_index, ]


# check for mult-collinearity

set.seed(1)

# build the model
# dropping response variables for calculating multi-collinearity
# model <- lm(y ~ ., data=train_set[1:11])
train_set[1:10] <- train_set[1:10] %>% mutate(Gender = as.numeric(Gender)) %>% scale()

model <- lm(y ~ ., data=train_set[1:11])

# Make predictions
test_set[1:10] <- test_set[1:10] %>% mutate(Gender = as.numeric(Gender)) %>% scale()

predictions <- predict(model, test_set[1:10])


# Model performance
data.frame(RMSE = RMSE(predictions, test_set$y),
           R2 = R2(predictions, test_set$y))

#          RMSE          R2
#  0.4285702646 0.1008179244

# R2 - coeff of determination - statistical measure of how close the data are to the fitted regression line


# Variance Inflation Factor
car::vif(model)

#                  Age                     Gender             Total_Bilirubin           Direct_Bilirubin 
#          1.092611769                1.028734192                 3.051157575                3.294180067 
# Alkaline_Phosphotase   Alamine_Aminotransferase   Aspartate_Aminotransferase             Total_Protiens 
#          1.142404623                2.055956795                  2.086726198                6.160682883 
#              Albumin   Albumin_and_Globulin_Ratio
#         11.712996066                  4.208556640 


# Typically in practice there is a small amount of collinearity among the predictors. As a rule of thumb,
# a VIF value that exceeds 5 or 10 indicates a problematic amount of collinearity. Intro to Statistical Learning 7ed p.101 

# Hence, Albumi and Total_Protiens appear to be multi-collinear or redundant as
# they have VIF values of 11.712996066 and 6.160682883, respectively.



## 2.8 Check for normality

# This is to investigate whether the observed sample is from a normal distribution. It is used for assessing
# whether the sample data are randomly obtained from a normally distributed population. It does not require
# that the mean or variance of the hypothesized normal distribution be specified in advance.

# Sources: Lewis, N.D. 100 Statistical Tests in R: With over 300 illustrations and examples (Page 295).
# www.AusCov.com. Kindle Edition. 
# https://stackoverflow.com/questions/21239826/using-shapiro-test-on-multiple-columns-in-a-data-frame

# Run shapiro test to generate & report p-values
# Null hypothesis H0: Is the sample from a normal distribution?
# if p-value >= 0.05 we do not reject the null hypothesis that the data are from normal distribution
# if p-value < 0.05 we reject the null hypothesis that the data are from normal distribution

# exclude Gender factor {col 2} and dependent vars y {col 11} and disease {col 12}
df <- liver_clean[, c(1,3:10)]

# generate a shapiro list
lshap <- lapply(df, shapiro.test)

# select only the p-values and transpose
shap_p_val <- sapply(lshap, `[`, c("p.value")) %>% data.frame() %>% data.frame(row.names="p_values") %>% t()

shap_p_val

# Since the p-values for all the 9 numeric variables are less than 0.05, we reject the null hypothesis
# that the data are from normal distribution


#############

## 2.9 Check for linearity

# This is investigate whether the observed sample is linear. The null hypothesis is that the regression model is linear.
# This test attempts to detect non-linearities when the data is ordered with respect to a specific variable.

# Source: Lewis, N.D. 100 Statistical Tests in R: With over 300 illustrations and examples (Page 385).
# www.AusCov.com. Kindle Edition. 

# Run Harvey-Collier test to generate & report p-values
# Null hypothesis H0: Is the regression model correctly specified as linear?
# if p-value >= 0.05 we do not reject the null hypothesis of linearity
# if p-value < 0.05 we reject the null hypothesis of linearity

# Harvey-Collier test for linearity
# dependent var y as numeric, all other variables should be numeric
options(digits=10)

library(lmtest)

f <- y ~ Age + Total_Bilirubin + Direct_Bilirubin + Alkaline_Phosphotase + Alamine_Aminotransferase +
  Aspartate_Aminotransferase + Total_Protiens + Albumin + Albumin_and_Globulin_Ratio

harvtest(formula = f, data = liver_clean)

# Since p-value = 0.5785027, which is >= 0.05, we do not reject the null hypothesis of linearity


#############

## 2.10 Test for outliers: chi-squared test

# This is investigate whether the sample data contain outliers. The function chisq.out.test{outliers} can be used to
# perform this test takes the form chisq.out.test(data, variance=1). The parameter variance refers to the known
# population variance.

# Source: Lewis, N.D. 100 Statistical Tests in R: With over 300 illustrations and examples (Page 360).
# www.AusCov.com. Kindle Edition.


# derive the residual of the regression model
f <- y ~ Age + Total_Bilirubin + Direct_Bilirubin + Alkaline_Phosphotase + Alamine_Aminotransferase +
  Aspartate_Aminotransferase + Total_Protiens + Albumin + Albumin_and_Globulin_Ratio

# train model
reg_model <- lm(formula = f, data = liver_clean)

# residuals
reg_residual <- rstudent(reg_model)

library(outliers)

outlier <- chisq.out.test(reg_residual, variance = 1)

outlier

# chi-squared test for outlier
# data:  reg_residual
# X-squared.116 = 6.5277471, p-value = 0.01062044
# alternative hypothesis: highest value 2.556298918594 is an outlier

outlier_alternative <- outlier$alternative

outlier_pvalue <- outlier$p.value[[1]]


print(paste0("The ", outlier_alternative, " with a p-value of ", outlier_pvalue))


###
# boxplot 1 - plot chemical substances with low values
# Total_Bilirubin
# Direct_Bilirubin
# Total_Protiens
# Albumin
# Albumin_and_Globulin_Ratio

liver_clean[c(3, 4, 8, 9, 10)] %>% boxplot(main="Compare chemical substances with low values",
                                           horizontal=FALSE,
                                           notch=TRUE,
                                           boxwex = 0.25)


## boxplot 2 - plot chemical substances with high values
# Alkaline_Phosphotase
# Alamine_Aminotransferase
# Aspartate_Aminotransferase

liver_clean[c(5, 6, 7)] %>% boxplot(main="Compare chemical substances with high values",
                                    horizontal=FALSE,
                                    notch=TRUE,
                                    boxwex = 0.25)







################## Step 3. Split liver_clean dataset into train_set and test_set in prep for modeling

# For now we will ignore the liver_imputed dataset that we have created whose 4 null obs in the
# Albumin_and_Globulin_Ratio variable were imputed with values using MICE imputation algorithm.

# remove unnecessary datasets
rm(liver_imputed, liver)

# We will use the liver_clean dataset to generate the train_set and test_set datasets
# split liver_clean into 70% train_set and 30% test_set
# code trainControl if cross-validation is required 
library(caret)
set.seed(1)
#set.seed(1, sample.kind="Rounding")

test_index <- createDataPartition(y = liver_clean$disease, times = 1,
                                  p = 0.30, list = FALSE)

# on the liver_clean set
test_set <- liver_clean[test_index, ]
train_set <- liver_clean[-test_index, ]

# on the liver_imputed set
# test_impute_set <- liver_imputed[test_index, ]
# train_impute_set <- liver_imputed[-test_index, ]

# distribution of disease on train_set regardless of Gender
table(disease=train_set$disease)
# disease
#   N   Y 
# 289 115 

# distribution of disease on train_set by Gender
table(gender=train_set$Gender, disease=train_set$disease)
#            disease
# gender      N   Y
#    Female  61  39
#    Male   228  76


# distribution of disease on test_set regardless of Gender
table(disease=test_set$disease)
# disease
#   N   Y 
# 125  50 

# distribution of disease on test_set by Gender
table(gender=test_set$Gender, disease=test_set$disease)
#            disease
# gender      N   Y
#    Female  30  10
#    Male    95  40

# Note that there are 50/175 cases of liver disease in the test_set
# We will monitor this number in the confusion matrix table as we generate the models, and calculate accuracy,
# sensitivity and specificity.

# remove unnecessary temporary dataframe test_index
rm(test_index)


###################################################

################## Step 4. Modeling and performance assessment

options(digits=10)
library(knitr)

##########
## 4.1 Build logistic regression (LR) model h2o
# source: http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/glm.html

# load h2o and define training control

library(h2o)
h2o.removeAll() ## clean slate - just in case the cluster was already running

set.seed(1)
h2o.init(nthreads = -1)

train_control <- trainControl(method="cv", number=10)

# Fitting LR classifier to the training set
# dependent variable disease {col 12}
# exclude y int {col 11}
lr.model.h2o <- h2o.glm(x=c(1:10),
                        y=12,
                        training_frame=as.h2o(train_set),
                        family="binomial",
                        nfolds = 10,
                        model_id = "lr_h2o_model",
                        lambda=0,
                        compute_p_values = TRUE,
                        seed=1)

# extract the coefficients
lr.model.h2o@model$coefficients_table
summary(lr.model.h2o)

# Predicting the test set results
# exclude y int {col 11} and factor disease {col 12}
prob_pred.lr.h2o = as.data.frame(h2o.predict(lr.model.h2o, type='raw', newdata=as.h2o(test_set[-c(11,12)])))

# Making the Confusion Matrix
cm <- confusionMatrix(data=prob_pred.lr.h2o$predict, reference=test_set$disease)


# https://rafalab.github.io/dsbook/introduction-to-machine-learning.html Section 27.4.4

cm$table
#          Reference
# Prediction   N  Y
#           N 84 17
#           Y 41 33

# The raw prediction accuracy of the model is defined as (TruePositives + TrueNegatives)/SampleSize.
accuracy_01 <- cm$overall["Accuracy"][[1]]; accuracy_01
# 0.6685714286

# Sensitivity (or Recall), which is the True Positive Rate (TPR) or the proportion of identified positives among
# the liver disease-positive population (class = 1). Sensitivity = TP/(TP + FN).
sensitivity_01 <- cm$byClass["Sensitivity"][[1]]; sensitivity_01
# 0.672 

# Specificity, which measures the True Negative Rate (TNR), that is the proportion of identified negatives among
# the liver disease-negative population (class = 0). Specificity = TN/(TN + FP).
specificity_01 <- cm$byClass["Specificity"][[1]]; specificity_01
# 0.66 

# Precision, which is the proportion of true positives among all the individuals that have been predicted
# to have liver disease-positive by the model. This represents the accuracy of a predicted positive outcome.
# Precision = TP/(TP + FP).
precision_01 <- cm$byClass["Precision"][[1]]; precision_01
# 0.8316831683

# create a results dataframe that indicates all accuracy results on the test_set
results <- data.frame(method = "h2o logistic regression",
                      accuracy = accuracy_01,
                      sensitivity = sensitivity_01,
                      specificity = specificity_01,
                      precision = precision_01)

results %>% knitr::kable()
#|method                  |     accuracy| sensitivity| specificity|    precision|
#|:-----------------------|------------:|-----------:|-----------:|------------:|
#|h2o logistic regression | 0.6685714286|       0.672|        0.66| 0.8316831683|


##########
## 4.2 Build neural network (deep learning) model h2o

# Fitting neural network model to the training set
# dependent variable disease {col 12}
# exclude y int {col 11}
# since there are 10 variables, 1st hidden = 10/2 + 1 = 6
#                               2nd hidden = 1st hidden/2 = 3
set.seed(1)

neural_deeplearn_h2o <- h2o.deeplearning(x=c(1:10),
                                         y=12,
                                         training_frame=as.h2o(train_set),
                                         seed=1,
                                         nfolds = 10,
                                         epochs=50,
                                         hidden = c(6,3),
                                         activation="Rectifier",
                                         overwrite_with_best_model=TRUE,
                                         use_all_factor_levels = TRUE,
                                         variable_importances = TRUE,
                                         export_weights_and_biases = TRUE,
                                         verbose=TRUE)

h2o.performance(neural_deeplearn_h2o)


# Predicting the test set results
# exclude y int {col 11} and factor disease {col 12}
prob_pred.dl.h2o = as.data.frame(h2o.predict(neural_deeplearn_h2o, type='raw', newdata=as.h2o(test_set[-c(11,12)])))

# Making the Confusion Matrix
cm <- confusionMatrix(data=prob_pred.dl.h2o$predict, reference=test_set$disease)

cm$table
#          Reference
# Prediction    N  Y
#            N 87 19
#            Y 38 31


# accuracy = (TruePositives + TrueNegatives)/SampleSize
accuracy_02 <- cm$overall["Accuracy"][[1]]; accuracy_02
# 0.6742857143

# Sensitivity = TP/(TP + FN)
sensitivity_02 <- cm$byClass["Sensitivity"][[1]]; sensitivity_02
# 0.696

# Specificity = TN/(TN + FP)
specificity_02 <- cm$byClass["Specificity"][[1]]; specificity_02
# 0.62

# Precision = TP/(TP + FP)
precision_02 <- cm$byClass["Precision"][[1]]; precision_02
# 0.820754717

# add to the results dataframe previously created
results <- results %>% add_row(method = "h2o neural network/deep learning (6,3) hidden layers",
                               accuracy = accuracy_02,
                               sensitivity = sensitivity_02,
                               specificity = specificity_02,
                               precision = precision_02)

results %>% knitr::kable()
#|method                                               |     accuracy| sensitivity| specificity|    precision|
#|:----------------------------------------------------|------------:|-----------:|-----------:|------------:|
#|h2o logistic regression                              | 0.6685714286|       0.672|        0.66| 0.8316831683|
#|h2o neural network/deep learning (6,3) hidden layers | 0.6742857143|       0.696|        0.62| 0.8207547170|


# plot the neural network model
# source: https://stackoverflow.com/questions/51432797/use-rs-neuralnettoolslibrary-to-plot-the-network-structure-of-a-h2o-deep-neural
library(NeuralNetTools)

net <- neural_deeplearn_h2o

wts <- c()
for (l in 1:(length(net@allparameters$hidden)+1)){
  wts_in <- h2o.weights(net, l)
  biases <- as.vector(h2o.biases(net, l))
  for (i in 1:nrow(wts_in)){
    wts <- c(wts, biases[i], as.vector(wts_in[i,]))
  }
}
# generate struct from column 'units' in model_summary
struct <- net@model$model_summary$units

# plot it
plotnet(wts, struct = struct)






##########
## 4.3 Build random forest (rf) model h2o

# Fitting RF model to the training set
# dependent variable disease {col 12}
# exclude y int {col 11}
set.seed(1)

random_forest_h2o <- h2o.randomForest(x=c(1:10),
                                      y=12,
                                      training_frame=as.h2o(train_set),
                                      seed=1,
                                      nfolds = 10,
                                      ntrees = 1000,
                                      mtries = 5,
                                      max_depth = 5)

h2o.performance(random_forest_h2o)

# Predicting the test set results
# exclude y int {col 11} and factor disease {col 12}
prob_pred.rf.h2o = as.data.frame(h2o.predict(random_forest_h2o, type='raw', newdata=as.h2o(test_set[-c(11,12)])))

# Making the Confusion Matrix
cm <- confusionMatrix(data=prob_pred.rf.h2o$predict, reference=test_set$disease)

cm$table
#          Reference
# Prediction   N  Y
#           N 67 11
#           Y 58 39

# accuracy = (TruePositives + TrueNegatives)/SampleSize
accuracy_03 <- cm$overall["Accuracy"][[1]]; accuracy_03
# 0.6057142857

# Sensitivity = TP/(TP + FN)
sensitivity_03 <- cm$byClass["Sensitivity"][[1]]; sensitivity_03
# 0.536

# Specificity = TN/(TN + FP)
specificity_03 <- cm$byClass["Specificity"][[1]]; specificity_03
# 0.78

# Precision = TP/(TP + FP)
precision_03 <- cm$byClass["Precision"][[1]]; precision_03
# 0.858974359

# add to the results dataframe previously created
results <- results %>% add_row(method = "h2o random forests model",
                               accuracy = accuracy_03,
                               sensitivity = sensitivity_03,
                               specificity = specificity_03,
                               precision = precision_03)

results %>% knitr::kable()
#|method                                               |     accuracy| sensitivity| specificity|    precision|
#|:----------------------------------------------------|------------:|-----------:|-----------:|------------:|
#|h2o logistic regression                              | 0.6685714286|       0.672|        0.66| 0.8316831683|
#|h2o neural network/deep learning (6,3) hidden layers | 0.6742857143|       0.696|        0.62| 0.8207547170|
#|h2o random forests model                             | 0.6057142857|       0.536|        0.78| 0.8589743590|



##########
## 4.4 Build gradient boosting machine (gbm) model h2o

# Fitting gbm model to the training set
# dependent variable disease {col 12}
# exclude y int {col 11}
set.seed(1)

gbm_h2o <- h2o.gbm(x=c(1:10),
                   y=12,
                   training_frame=as.h2o(train_set),
                   seed=1,
                   nfolds = 10,
                   distribution="bernoulli",
                   stopping_metric = "AUTO",
                   categorical_encoding = "AUTO",
                   verbose=TRUE)

h2o.performance(gbm_h2o)

# Predicting the test set results
# exclude y int {col 11} and factor disease {col 12}
prob_pred.gbm.h2o = as.data.frame(h2o.predict(gbm_h2o, type='raw', newdata=as.h2o(test_set[-c(11,12)])))

# Making the Confusion Matrix
cm <- confusionMatrix(data=prob_pred.gbm.h2o$predict, reference=test_set$disease)

cm$table
#          Reference
# Prediction   N  Y
#           N 99 29
#           Y 26 21

# accuracy = (TruePositives + TrueNegatives)/SampleSize
accuracy_04 <- cm$overall["Accuracy"][[1]]; accuracy_04
# 0.6857142857

# Sensitivity = TP/(TP + FN)
sensitivity_04 <- cm$byClass["Sensitivity"][[1]]; sensitivity_04
# 0.792

# Specificity = TN/(TN + FP)
specificity_04 <- cm$byClass["Specificity"][[1]]; specificity_04
# 0.42

# Precision = TP/(TP + FP)
precision_04 <- cm$byClass["Precision"][[1]]; precision_04
# 0.7734375

# add to the results dataframe previously created
results <- results %>% add_row(method = "h2o gradient boosting machine (gbm) model",
                               accuracy = accuracy_04,
                               sensitivity = sensitivity_04,
                               specificity = specificity_04,
                               precision = precision_04)

results %>% knitr::kable()
#|method                                               |     accuracy| sensitivity| specificity|    precision|
#|:----------------------------------------------------|------------:|-----------:|-----------:|------------:|
#|h2o logistic regression                              | 0.6685714286|       0.672|        0.66| 0.8316831683|
#|h2o neural network/deep learning (6,3) hidden layers | 0.6742857143|       0.696|        0.62| 0.8207547170|
#|h2o random forests model                             | 0.6057142857|       0.536|        0.78| 0.8589743590|
#|h2o gradient boosting machine (gbm) model            | 0.6857142857|       0.792|        0.42| 0.7734375000|


##########
## 4.5 Build support vector machine (SVM) model h2o

# Fitting SVM model to the training set
# dependent variable disease {col 12}
# exclude y int {col 11}
set.seed(1)

svm_h2o <- h2o.psvm(x=c(1:10),
                    y=12,
                    training_frame=as.h2o(train_set),
                    gamma = 0.01, rank_ratio = 0.1,
                    disable_training_metrics = FALSE)

h2o.performance(svm_h2o)

# Predicting the test set results
# exclude y int {col 11} and factor disease {col 12}
prob_pred.svm.h2o = as.data.frame(h2o.predict(svm_h2o, type='raw', newdata=as.h2o(test_set[-c(11,12)])))

# Making the Confusion Matrix
cm <- confusionMatrix(data=prob_pred.svm.h2o$predict, reference=test_set$disease)

cm$table
#          Reference
# Prediction   N  Y
#         N 120  37
#         Y   5  13

# accuracy = (TruePositives + TrueNegatives)/SampleSize
accuracy_05 <- cm$overall["Accuracy"][[1]]; accuracy_05
# 0.76

# Sensitivity = TP/(TP + FN)
sensitivity_05 <- cm$byClass["Sensitivity"][[1]]; sensitivity_05
# 0.96

# Specificity = TN/(TN + FP)
specificity_05 <- cm$byClass["Specificity"][[1]]; specificity_05
# 0.26

# Precision = TP/(TP + FP)
precision_05 <- cm$byClass["Precision"][[1]]; precision_05
# 0.7643312102

# add to the results dataframe previously created
results <- results %>% add_row(method = "h2o support vector machine (svm) model",
                               accuracy = accuracy_05,
                               sensitivity = sensitivity_05,
                               specificity = specificity_05,
                               precision = precision_05)

results %>% knitr::kable()
#|method                                               |     accuracy| sensitivity| specificity|    precision|
#|:----------------------------------------------------|------------:|-----------:|-----------:|------------:|
#|h2o logistic regression                              | 0.6685714286|       0.672|        0.66| 0.8316831683|
#|h2o neural network/deep learning (6,3) hidden layers | 0.6742857143|       0.696|        0.62| 0.8207547170|
#|h2o random forests model                             | 0.6057142857|       0.536|        0.78| 0.8589743590|
#|h2o gradient boosting machine (gbm) model            | 0.6857142857|       0.792|        0.42| 0.7734375000|
#|h2o support vector machine (svm) model               | 0.7600000000|       0.960|        0.26| 0.7643312102|


##########
## 4.6 Build naive bayes (NB) model h2o

# Fitting NB model to the training set
# dependent variable disease {col 12}
# exclude y int {col 11}
set.seed(1)

nb_h2o <- h2o.naiveBayes(x=c(1:10),
                         y=12,
                         training_frame=as.h2o(train_set),
                         nfolds = 10,
                         seed = 1,
                         fold_assignment="Stratified",
                         laplace = 1)

h2o.performance(nb_h2o)

# Predicting the test set results
# exclude y int {col 11} and factor disease {col 12}
prob_pred.nb.h2o = as.data.frame(h2o.predict(nb_h2o, type='raw', newdata=as.h2o(test_set[-c(11,12)])))

# Making the Confusion Matrix
cm <- confusionMatrix(data=prob_pred.nb.h2o$predict, reference=test_set$disease)

cm$table
#          Reference
# Prediction   N  Y
#           N 60  6
#           Y 65 44

# accuracy = (TruePositives + TrueNegatives)/SampleSize
accuracy_06 <- cm$overall["Accuracy"][[1]]; accuracy_06
# 0.5942857143

# Sensitivity = TP/(TP + FN)
sensitivity_06 <- cm$byClass["Sensitivity"][[1]]; sensitivity_06
# 0.48

# Specificity = TN/(TN + FP)
specificity_06 <- cm$byClass["Specificity"][[1]]; specificity_06
# 0.88

# Precision = TP/(TP + FP)
precision_06 <- cm$byClass["Precision"][[1]]; precision_06
# 0.9090909091

# add to the results dataframe previously created
results <- results %>% add_row(method = "h2o naive bayes model",
                               accuracy = accuracy_06,
                               sensitivity = sensitivity_06,
                               specificity = specificity_06,
                               precision = precision_06)

results %>% knitr::kable()
#|method                                               |     accuracy| sensitivity| specificity|    precision|
#|:----------------------------------------------------|------------:|-----------:|-----------:|------------:|
#|h2o logistic regression                              | 0.6685714286|       0.672|        0.66| 0.8316831683|
#|h2o neural network/deep learning (6,3) hidden layers | 0.6742857143|       0.696|        0.62| 0.8207547170|
#|h2o random forests model                             | 0.6057142857|       0.536|        0.78| 0.8589743590|
#|h2o gradient boosting machine (gbm) model            | 0.6857142857|       0.792|        0.42| 0.7734375000|
#|h2o support vector machine (svm) model               | 0.7600000000|       0.960|        0.26| 0.7643312102|
#|h2o naive bayes model                                | 0.5942857143|       0.480|        0.88| 0.9090909091|


## shutdown h2o
h2o.shutdown()


##########
## 4.7 Decision tree: Classification Tree

# Source: Lewis, N.D.: 92 Applied Predictive Modeling Techniques in R (Page 22)  

# Fitting tree model to the training set
# dependent variable disease {col 12}
# exclude y int {col 11}
library(tree)
library(mlbench)

set.seed(1)

# We use all the variables
f <- disease ~ Direct_Bilirubin + Alamine_Aminotransferase + Age + Alkaline_Phosphotase +
  Gender + Total_Bilirubin + Aspartate_Aminotransferase + Total_Protiens + Albumin + Albumin_and_Globulin_Ratio

tree_fit_01 <- tree(formula= f,
                    data=train_set[-c(11)],
                    split="deviance")

summary(tree_fit_01)

plot(tree_fit_01); text(tree_fit_01, cex=0.5)

# # Predicting the test set results
# exclude y int {col 11} and factor disease {col 12}
prob_pred.tree_01 <- predict(tree_fit_01, newdata=test_set[-c(11,12)])
prob_pred.tree_01_class <- as.factor(colnames(prob_pred.tree_01)[max.col(prob_pred.tree_01, ties.method = c("random"))])

# Making the Confusion Matrix
cm <- confusionMatrix(data=prob_pred.tree_01_class, reference=test_set$disease)

cm$table
#          Reference
# Prediction   N  Y
#           N 88 26
#           Y 37 24

# accuracy = (TruePositives + TrueNegatives)/SampleSize
accuracy_07 <- cm$overall["Accuracy"][[1]]; accuracy_07
# 0.64

# Sensitivity = TP/(TP + FN)
sensitivity_07 <- cm$byClass["Sensitivity"][[1]]; sensitivity_07
# 0.704

# Specificity = TN/(TN + FP)
specificity_07 <- cm$byClass["Specificity"][[1]]; specificity_07
# 0.48

# Precision = TP/(TP + FP)
precision_07 <- cm$byClass["Precision"][[1]]; precision_07
# 0.7719298246

# add to the results dataframe previously created
results <- results %>% add_row(method = "classification tree",
                               accuracy = accuracy_07,
                               sensitivity = sensitivity_07,
                               specificity = specificity_07,
                               precision = precision_07)

results %>% knitr::kable()
#|method                                               |     accuracy| sensitivity| specificity|    precision|
#|:----------------------------------------------------|------------:|-----------:|-----------:|------------:|
#|h2o logistic regression                              | 0.6685714286|       0.672|        0.66| 0.8316831683|
#|h2o neural network/deep learning (6,3) hidden layers | 0.6742857143|       0.696|        0.62| 0.8207547170|
#|h2o random forests model                             | 0.6057142857|       0.536|        0.78| 0.8589743590|
#|h2o gradient boosting machine (gbm) model            | 0.6857142857|       0.792|        0.42| 0.7734375000|
#|h2o support vector machine (svm) model               | 0.7600000000|       0.960|        0.26| 0.7643312102|
#|h2o naive bayes model                                | 0.5942857143|       0.480|        0.88| 0.9090909091|
#|classification tree                                  | 0.6400000000|       0.704|        0.48| 0.7719298246|


##########
## 4.8 C5.0 Classification Tree

# Source: Lewis, N.D.: 92 Applied Predictive Modeling Techniques in R (Page 32)  

# Fitting tree model to the training set
# dependent variable disease {col 12}
# exclude y int {col 11}
library(C50)
library(mlbench)

set.seed(1)

# We use all the variables
f <- disease ~ Direct_Bilirubin + Alamine_Aminotransferase + Age + Alkaline_Phosphotase +
  Gender + Total_Bilirubin + Aspartate_Aminotransferase + Total_Protiens + Albumin + Albumin_and_Globulin_Ratio

tree_fit_02 <- C5.0(formula= f,
                    data=train_set[-c(11)])

# important variables
C5imp(tree_fit_02)
#                           Overall
#Aspartate_Aminotransferase  100.00
#Total_Bilirubin              84.41
#Gender                       58.91
#Albumin                      40.10
#Age                          34.90
#Alkaline_Phosphotase         16.58
#Alamine_Aminotransferase     15.84
#Direct_Bilirubin              0.00
#Total_Protiens                0.00
#Albumin_and_Globulin_Ratio    0.00

plot(tree_fit_02); text(tree_fit_02, cex=0.01)

# # Predicting the test set results
# exclude y int {col 11} and factor disease {col 12}
prob_pred.tree_02 <- predict(tree_fit_02, newdata=test_set[-c(11,12)], type="class")

# Making the Confusion Matrix
cm <- confusionMatrix(data=prob_pred.tree_02, reference=test_set$disease)

cm$table
#          Reference
# Prediction    N  Y
#           N 113  40
#           Y  12  10

# accuracy = (TruePositives + TrueNegatives)/SampleSize
accuracy_08 <- cm$overall["Accuracy"][[1]]; accuracy_08
# 0.7028571429

# Sensitivity = TP/(TP + FN)
sensitivity_08 <- cm$byClass["Sensitivity"][[1]]; sensitivity_08
# 0.904

# Specificity = TN/(TN + FP)
specificity_08 <- cm$byClass["Specificity"][[1]]; specificity_08
# 0.20

# Precision = TP/(TP + FP)
precision_08 <- cm$byClass["Precision"][[1]]; precision_08
# 0.7385620915

# add to the results dataframe previously created
results <- results %>% add_row(method = "C5.0 classification tree",
                               accuracy = accuracy_08,
                               sensitivity = sensitivity_08,
                               specificity = specificity_08,
                               precision = precision_08)

results %>% knitr::kable()
#|method                                               |     accuracy| sensitivity| specificity|    precision|
#|:----------------------------------------------------|------------:|-----------:|-----------:|------------:|
#|h2o logistic regression                              | 0.6685714286|       0.672|        0.66| 0.8316831683|
#|h2o neural network/deep learning (6,3) hidden layers | 0.6742857143|       0.696|        0.62| 0.8207547170|
#|h2o random forests model                             | 0.6057142857|       0.536|        0.78| 0.8589743590|
#|h2o gradient boosting machine (gbm) model            | 0.6857142857|       0.792|        0.42| 0.7734375000|
#|h2o support vector machine (svm) model               | 0.7600000000|       0.960|        0.26| 0.7643312102|
#|h2o naive bayes model                                | 0.5942857143|       0.480|        0.88| 0.9090909091|
#|classification tree                                  | 0.6457142857|       0.712|        0.48| 0.7739130435|
#|C5.0 classification tree                             | 0.7028571429|       0.904|        0.20| 0.7385620915|


##########
## 4.9 Evolutionary classification Tree

# Source: Lewis, N.D.: 92 Applied Predictive Modeling Techniques in R (Page 41)  

# Fitting tree model to the training set
# dependent variable disease {col 12}
# exclude y int {col 11}
library(evtree)
library(mlbench)

set.seed(1)

# We use all the variables
f <- disease ~ Direct_Bilirubin + Alamine_Aminotransferase + Age + Alkaline_Phosphotase +
  Gender + Total_Bilirubin + Aspartate_Aminotransferase + Total_Protiens + Albumin + Albumin_and_Globulin_Ratio

tree_fit_03 <- evtree(formula= f,
                      data=train_set[-c(11)],
                      control=evtree.control(maxdepth=7))

plot(tree_fit_03, cex=0.1)

# # Predicting the test set results
# exclude y int {col 11} and factor disease {col 12}
prob_pred.tree_03 <- predict(tree_fit_03, newdata=test_set[-c(11,12)], type="response")

# Making the Confusion Matrix
cm <- confusionMatrix(data=prob_pred.tree_03, reference=test_set$disease)

cm$table
#          Reference
# Prediction    N   Y
#           N 115  43
#           Y  10   7

# accuracy = (TruePositives + TrueNegatives)/SampleSize
accuracy_09 <- cm$overall["Accuracy"][[1]]; accuracy_09
# 0.6971428571

# Sensitivity = TP/(TP + FN)
sensitivity_09 <- cm$byClass["Sensitivity"][[1]]; sensitivity_09
# 0.92

# Specificity = TN/(TN + FP)
specificity_09 <- cm$byClass["Specificity"][[1]]; specificity_09
# 0.14

# Precision = TP/(TP + FP)
precision_09 <- cm$byClass["Precision"][[1]]; precision_09
# 0.7278481013

# add to the results dataframe previously created
results <- results %>% add_row(method = "Evolutionary classification tree",
                               accuracy = accuracy_09,
                               sensitivity = sensitivity_09,
                               specificity = specificity_09,
                               precision = precision_09)

results %>% knitr::kable()
#|method                                               |     accuracy| sensitivity| specificity|    precision|
#|:----------------------------------------------------|------------:|-----------:|-----------:|------------:|
#|h2o logistic regression                              | 0.6685714286|       0.672|        0.66| 0.8316831683|
#|h2o neural network/deep learning (6,3) hidden layers | 0.6742857143|       0.696|        0.62| 0.8207547170|
#|h2o random forests model                             | 0.6057142857|       0.536|        0.78| 0.8589743590|
#|h2o gradient boosting machine (gbm) model            | 0.6857142857|       0.792|        0.42| 0.7734375000|
#|h2o support vector machine (svm) model               | 0.7600000000|       0.960|        0.26| 0.7643312102|
#|h2o naive bayes model                                | 0.5942857143|       0.480|        0.88| 0.9090909091|
#|classification tree                                  | 0.6457142857|       0.712|        0.48| 0.7739130435|
#|C5.0 classification tree                             | 0.7028571429|       0.904|        0.20| 0.7385620915|
#|Evolutionary classification tree                     | 0.6971428571|       0.920|        0.14| 0.7278481013|


##########
## 4.10 Logistic model based recursive partitioning

# Source: Lewis, N.D.: 92 Applied Predictive Modeling Techniques in R (Page 52)  

# Fitting tree model to the training set
# dependent variable disease {col 12}
# exclude y int {col 11}
library(party)

set.seed(1)

# We use Aspartate_Aminotransferase + Total_Bilirubin + Gender + Albumin as logistic regression conditioning variables
# Then use Direct_Bilirubin + Alamine_Aminotransferase + Age + Alkaline_Phosphotase as partitioning variables
f <- disease ~ Aspartate_Aminotransferase + Total_Bilirubin + Gender + Albumin| Direct_Bilirubin +
  Alamine_Aminotransferase + Age + Alkaline_Phosphotase

tree_fit_04 <- mob(formula= disease ~ Aspartate_Aminotransferase + Total_Bilirubin + Gender + 
                     Albumin | Direct_Bilirubin + Alamine_Aminotransferase + Age + 
                     Alkaline_Phosphotase,
                   data=train_set[-c(11)],
                   model=glinearModel,
                   family=binomial())

plot(tree_fit_04)

# # Predicting the test set results
# exclude y int {col 11} and factor disease {col 12}
prob_pred.tree_04 <- predict(tree_fit_04, newdata=test_set[-c(11,12)], type="response")

prob_pred.tree_04_class <- as.factor(ifelse(prob_pred.tree_04 > 0.50, "Y", "N"))

# Making the Confusion Matrix
cm <- confusionMatrix(data=prob_pred.tree_04_class, reference=test_set$disease)

cm$table
#          Reference
# Prediction    N    Y
#            N 114  43
#            Y  11   7


# accuracy = (TruePositives + TrueNegatives)/SampleSize
accuracy_10 <- cm$overall["Accuracy"][[1]]; accuracy_10
# 0.6914285714

# Sensitivity = TP/(TP + FN)
sensitivity_10<- cm$byClass["Sensitivity"][[1]]; sensitivity_10
# 0.912

# Specificity = TN/(TN + FP)
specificity_10 <- cm$byClass["Specificity"][[1]]; specificity_10
# 0.14

# Precision = TP/(TP + FP)
precision_10 <- cm$byClass["Precision"][[1]]; precision_10
# 0.7261146497

# add to the results dataframe previously created
results <- results %>% add_row(method = "Logistic model based recursive partitioning",
                               accuracy = accuracy_10,
                               sensitivity = sensitivity_10,
                               specificity = specificity_10,
                               precision = precision_10)

results %>% knitr::kable()
#|method                                               |     accuracy| sensitivity| specificity|    precision|
#|:----------------------------------------------------|------------:|-----------:|-----------:|------------:|
#|h2o logistic regression                              | 0.6685714286|       0.672|        0.66| 0.8316831683|
#|h2o neural network/deep learning (6,3) hidden layers | 0.6742857143|       0.696|        0.62| 0.8207547170|
#|h2o random forests model                             | 0.6057142857|       0.536|        0.78| 0.8589743590|
#|h2o gradient boosting machine (gbm) model            | 0.6857142857|       0.792|        0.42| 0.7734375000|
#|h2o support vector machine (svm) model               | 0.7600000000|       0.960|        0.26| 0.7643312102|
#|h2o naive bayes model                                | 0.5942857143|       0.480|        0.88| 0.9090909091|
#|classification tree                                  | 0.6457142857|       0.712|        0.48| 0.7739130435|
#|C5.0 classification tree                             | 0.7028571429|       0.904|        0.20| 0.7385620915|
#|Evolutionary classification tree                     | 0.6971428571|       0.920|        0.14| 0.7278481013|
#|Logistic model based recursive partitioning          | 0.6914285714|       0.912|        0.14| 0.7261146497|


##########
## 4.11  Principal Components Analysis (PCA)

# sources:
# https://www.analyticsvidhya.com/blog/2016/03/pca-practical-guide-principal-component-analysis-python/
# http://www.sthda.com/english/articles/31-principal-component-methods-in-r-practical-guide/118-principal-component-analysis-in-r-prcomp-vs-princomp/

# Remember, PCA can be applied only on numerical data. Therefore, if the data has categorical variables
# they must be converted to numerical. 

set.seed(1)

#remove the dependent and identifier variables
train_set[1:10] %>% mutate(Gender=as.numeric(Gender)) %>%
  prcomp(., scale. = T)

#principal component analysis
pc_01 <- train_set[1:10] %>% mutate(Gender=as.numeric(Gender)) %>% prcomp(., scale. = T)

summary(pc_01)

# rotation
pc_01$rotation

dim(pc_01$rotation)
# 10  10

# x = scores = The coordinates of the individuals (observations) on the principal components.
pc_01$x

dim(pc_01$x)
# 404  10


# add a training set with principal components
train.data <- data.frame(y = train_set$y, pc_01$x)


# transform test_set into PCA
# test.data <- predict(pc, newdata = test_set)
# test.data <- as.data.frame(test.data)

test.data <- test_set[1:10] %>% mutate(Gender=as.numeric(Gender)) %>%
  predict(pc_01, newdata = .) %>% as.data.frame()

dim(test.data)
# 175  10


# run a decision tree
# install.packages("rpart")
library(rpart)
rpart.model <- rpart(y ~ ., data = train.data, method = "anova")
rpart.model


#make prediction on test data
rpart.prediction <- predict(rpart.model, test.data) %>% as.data.frame()

dim(rpart.prediction)
# 175 1

# convert prediction into categorical N or Y
pred <- rpart.prediction %>%
  mutate(class_pred = ifelse(rpart.prediction >= 1.5, "Y", "N")) %>%
  pull(class_pred) %>% as.vector() %>% as.factor()

cm <- confusionMatrix(data = pred, reference=test_set$disease)

cm$table
#          Reference
# Prediction    N    Y
#            N 98 34
#            Y 27 16

# accuracy = (TruePositives + TrueNegatives)/SampleSize
accuracy_11 <- cm$overall["Accuracy"][[1]]; accuracy_11
# 0.6514285714

# Sensitivity = TP/(TP + FN)
sensitivity_11<- cm$byClass["Sensitivity"][[1]]; sensitivity_11
# 0.784

# Specificity = TN/(TN + FP)
specificity_11 <- cm$byClass["Specificity"][[1]]; specificity_11
# 0.32

# Precision = TP/(TP + FP)
precision_11 <- cm$byClass["Precision"][[1]]; precision_11
# 0.7424242424

# add to the results dataframe previously created
results <- results %>% add_row(method = "Principal components analysis (PCA)",
                               accuracy = accuracy_11,
                               sensitivity = sensitivity_11,
                               specificity = specificity_11,
                               precision = precision_11)

results %>% knitr::kable()
#|method                                               |     accuracy| sensitivity| specificity|    precision|
#|:----------------------------------------------------|------------:|-----------:|-----------:|------------:|
#|h2o logistic regression                              | 0.6685714286|       0.672|        0.66| 0.8316831683|
#|h2o neural network/deep learning (6,3) hidden layers | 0.6742857143|       0.696|        0.62| 0.8207547170|
#|h2o random forests model                             | 0.6057142857|       0.536|        0.78| 0.8589743590|
#|h2o gradient boosting machine (gbm) model            | 0.6857142857|       0.792|        0.42| 0.7734375000|
#|h2o support vector machine (svm) model               | 0.7600000000|       0.960|        0.26| 0.7643312102|
#|h2o naive bayes model                                | 0.5942857143|       0.480|        0.88| 0.9090909091|
#|classification tree                                  | 0.6457142857|       0.712|        0.48| 0.7739130435|
#|C5.0 classification tree                             | 0.7028571429|       0.904|        0.20| 0.7385620915|
#|Evolutionary classification tree                     | 0.6971428571|       0.920|        0.14| 0.7278481013|
#|Logistic model based recursive partitioning          | 0.6914285714|       0.912|        0.14| 0.7261146497|
#|Principal components analysis (PCA)                  | 0.6514285714|       0.784|        0.32| 0.7424242424|


##########
## 4.12 Principal components analysis - singular value decomposition method (PCA-SVD)

# https://rdrr.io/bioc/pcaMethods/man/pca.html
# https://rdrr.io/bioc/pcaMethods/man/predict-methods.html

# install pcaMethods
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("pcaMethods")

set.seed(1)

# using PcaMethods
# SVD: a fast method which is also the standard method in R but which is not applicable for data with missing values.
# https://github.com/hredestig/pcamethods
# Fitting tree model to the training set
# dependent variable disease {col 12}
# exclude y int {col 11}
library(pcaMethods)
# listPcaMethods()
# One of "UV" (unit variance a=a/σ_{a}) "vector" (vector normalisation b=b/||b||),
# "pareto" (sqrt UV) or "none" to indicate which scaling should be used to scale the matrix with a variables and b samples.

# fit on train_set
# all numeric
train <- train_set[1:11] %>% mutate(Gender=as.numeric(Gender))

# Perform PCA on a numeric matrix
# method = singular value decomposition (svd)
# source: https://www.bioconductor.org/packages/release/bioc/manuals/pcaMethods/man/pcaMethods.pdf
pc_02 <- pca(train,
             method = "svd",
             nPcs = 10,     # number of principal components to calculate.
             scale = "uv",  # unit variance (or could be "vector" (vector normalisation), "pareto" (sqrt UV) or "none")
             completeObs = FALSE,
             center=TRUE)

# x = scores = The coordinates of the individuals (observations) on the principal components.
summary(pc_02)
# svd calculated PCA
# Importance of component(s):
#                  PC1    PC2    PC3    PC4    PC5    PC6     PC7     PC8     PC9    PC10
# R2            0.2633 0.1798 0.1246 0.0959 0.0855 0.0748 0.07013 0.05923 0.02588 0.01634
# Cumulative R2 0.2633 0.4431 0.5678 0.6637 0.7492 0.8240 0.89409 0.95332 0.97920 0.99554

# will use the first 6 PCs

# x = scores = The coordinates of the individuals (observations) on the principal components
dim(scores(pc_02))
# 404  10

dim(loadings(pc_02))
# 11 10


# predicting data using pca model (pcaMethods)
# source: https://rdrr.io/bioc/pcaMethods/man/predict-methods.html
# predict using test_set but set y = NA
test <- test_set[1:10] %>% mutate(Gender=as.numeric(Gender),
                                  y = NA)

pc_test_02 <- predict(pc_02,
                      newdata = test,  # new data with same number of columns as the used to compute
                      pcs=6,           # number of PCs
                      pre=TRUE)        # pre-process newdata based on the pre-processing chosen for the PCA model

# the output is a list with the following components
#   scores - predicted scores
#   x - predicted data

# predicted scores
pc_test_02$scores
dim(pc_test_02$scores)
# 175   6

# predicted data
pc_test_02$x
dim(pc_test_02$x)
# 175  11

# the predicted dependent variable y is now populated at col 11
pc_test_02$x[, 11]

# convert the predicted values into categorical "N" or "Y"
# "N" if x < 1.50
# "Y" if x >= 1.50
# ifelse(pc_test_03$x[, 11] >= 1.5, "Y", "N")

pred <- data.frame(pc_test_02$x[, 11]) %>%
  mutate(class_pred = ifelse(pc_test_02$x[, 11] >= 1.50, "Y", "N")) %>%
  pull(class_pred) %>% as.vector() %>% as.factor()

# confusion matrix
cm <- confusionMatrix(data = pred, reference=test_set$disease)

cm$table
#          Reference
# Prediction    N    Y
#            N 103  41
#            Y  22   9

# accuracy = (TruePositives + TrueNegatives)/SampleSize
accuracy_12 <- cm$overall["Accuracy"][[1]]; accuracy_12
# 0.64

# Sensitivity = TP/(TP + FN)
sensitivity_12<- cm$byClass["Sensitivity"][[1]]; sensitivity_12
# 0.824

# Specificity = TN/(TN + FP)
specificity_12 <- cm$byClass["Specificity"][[1]]; specificity_12
# 0.18

# Precision = TP/(TP + FP)
precision_12 <- cm$byClass["Precision"][[1]]; precision_12
# 0.7152777778

# add to the results dataframe previously created
results <- results %>% add_row(method = "PCA - singular value decomposition",
                               accuracy = accuracy_12,
                               sensitivity = sensitivity_12,
                               specificity = specificity_12,
                               precision = precision_12)

results %>% knitr::kable()

#|method                                               |     accuracy| sensitivity| specificity|    precision|
#|:----------------------------------------------------|------------:|-----------:|-----------:|------------:|
#|h2o logistic regression                              | 0.6685714286|       0.672|        0.66| 0.8316831683|
#|h2o neural network/deep learning (6,3) hidden layers | 0.6514285714|       0.632|        0.70| 0.8404255319|
#|h2o random forests model                             | 0.6057142857|       0.536|        0.78| 0.8589743590|
#|h2o gradient boosting machine (gbm) model            | 0.6857142857|       0.792|        0.42| 0.7734375000|
#|h2o support vector machine (svm) model               | 0.7600000000|       0.960|        0.26| 0.7643312102|
#|h2o naive bayes model                                | 0.5942857143|       0.480|        0.88| 0.9090909091|
#|classification tree                                  | 0.6400000000|       0.704|        0.48| 0.7719298246|
#|C5.0 classification tree                             | 0.7028571429|       0.904|        0.20| 0.7385620915|
#|Evolutionary classification tree                     | 0.6971428571|       0.920|        0.14| 0.7278481013|
#|Logistic model based recursive partitioning          | 0.6914285714|       0.912|        0.14| 0.7261146497|
#|Principal components analysis (PCA)                  | 0.6514285714|       0.784|        0.32| 0.7424242424|
#|PCA - singular value decomposition                   | 0.6400000000|       0.824|        0.18| 0.7152777778|


#######################################

### Annex 1 - Script to quickly and automatically install packages not currently installed

##### Script to quickly automatically install packages not currently installed:
# source: https://vbaliga.github.io/verify-that-r-packages-are-installed-and-loaded/
# Specify packages used in this project:
packages = c("tidyverse", "caret", "data.table", "tinytex", "tidyr", "devtools", "devtools", "mice", "earth",
             "knitr", "h2o", "mlbench", "C50", "evtree", "party", "e1071", "ModelMetrics", "car", "NeuralNetTools",
             "lmtest", "outliers", "BiocManager", "reshape2")

# Install packages if not present
package.check <- lapply(
  packages,
  FUN = function(x) {
    if (!require(x, character.only = TRUE)) {
      install.packages(x, dependencies = TRUE)
      library(x, character.only = TRUE)
    }
  }
)

# separately install xda
library(devtools)
install_github("ujjwalkarn/xda")

# install pcaMethods
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("pcaMethods")

#######################################


