#PART A: Logistic regression analysis the dataset. Validate assumptions, evaluate with a confusion matrix and ROC curve, and interpret the results. 


# Load necessary libraries
library(tidyverse)
library(caret)
library(pROC)
library(rpart)
library(rpart.plot)

# Read the data
df <- read.csv('/Users/sarathsabu/Desktop/scma/datasets/framingham.csv')

# Remove rows with missing values
df_clean <- na.omit(df)

# Split the data into features (X) and target variable (y)
X <- df_clean %>% select(-TenYearCHD)
y <- df_clean$TenYearCHD

# Split the data into training and testing sets
set.seed(42)
train_indices <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_indices, ]
X_test <- X[-train_indices, ]
y_train <- y[train_indices]
y_test <- y[-train_indices]

# Fit logistic regression model
logistic_model <- glm(TenYearCHD ~ ., data = cbind(X_train, TenYearCHD = y_train), family = 'binomial')

# Print summary of the logistic regression model
print(summary(logistic_model))

# Make predictions on the test set
y_pred_proba <- predict(logistic_model, newdata = X_test, type = 'response')
y_pred <- ifelse(y_pred_proba > 0.5, 1, 0)

# Create confusion matrix
conf_matrix <- confusionMatrix(factor(y_pred), factor(y_test))
print(conf_matrix)

# Plot ROC curve
roc_curve <- roc(y_test, y_pred_proba)
plot(roc_curve, main = 'ROC Curve for Logistic Regression')
auc_value <- auc(roc_curve)
print(paste('AUC:', auc_value))

# Fit decision tree model
tree_model <- rpart(TenYearCHD ~ ., data = cbind(X_train, TenYearCHD = y_train), method = 'class')

# Plot decision tree
rpart.plot(tree_model, main = 'Decision Tree for CHD Prediction')

# Make predictions using the decision tree
y_pred_tree <- predict(tree_model, newdata = X_test, type = 'class')

# Create confusion matrix for decision tree
conf_matrix_tree <- confusionMatrix(factor(y_pred_tree), factor(y_test))
print(conf_matrix_tree)

# Calculate ROC curve for decision tree
y_pred_proba_tree <- predict(tree_model, newdata = X_test, type = 'prob')[,2]
roc_curve_tree <- roc(y_test, y_pred_proba_tree)
plot(roc_curve_tree, main = 'ROC Curve for Decision Tree')
auc_value_tree <- auc(roc_curve_tree)
print(paste('AUC (Decision Tree):', auc_value_tree))


# 2 Perform a probit regression on "NSSO68.csv" to identify non-vegetarians. 

# Load the dataset
data_nss <- read.csv("/Users/sarathsabu/Desktop/scma/datasets/NSSO68.csv")
# Create a binary variable for chicken consumption
data_nss$chicken_q <- ifelse(data_nss$chicken_q > 0, 1, 0)

# Verify the creation of 'chicken_binary'
table(data_nss$chicken_q)

# Probit regression model
probit_model <- glm(chicken_q ~ Age + Marital_Status + Education, data = data_nss, family = binomial(link = "probit"))

# Summary of the probit regression model
summary(probit_model)

#3 
# Load necessary libraries
library(dplyr)
library(haven)
library(maxLik)

# Load the data
data <- read.csv('/Users/sarathsabu/Desktop/scma/datasets/NSSO68.csv', stringsAsFactors = FALSE)

# Subset data for state 'KA'
df <- data %>%
  select(MPCE_URP, Whether_owns_any_land, hhdsz, Religion, Social_Group, Regular_salary_earner)

# Check for missing values
cat("Missing values in MPCE_URP:", sum(is.na(df$MPCE_URP)), "\n")
cat("Missing values in Whether_owns_any_land:", sum(is.na(df$Whether_owns_any_land)), "\n")
cat("Missing values in hhdsz:", sum(is.na(df$hhdsz)), "\n")
cat("Missing values in Religion:", sum(is.na(df$Religion)), "\n")
cat("Missing values in Social_Group:", sum(is.na(df$Social_Group)), "\n")
cat("Missing values in Regular_salary_earner:", sum(is.na(df$Regular_salary_earner)), "\n")

# Impute missing values for selected columns
columns_to_impute <- c('Whether_owns_any_land', 'Religion', 'Social_Group', 'Regular_salary_earner')

# Assuming using mode for imputation for categorical variables
for (col in columns_to_impute) {
  mode_value <- names(sort(table(df[[col]]), decreasing = TRUE))[1]
  df[[col]][is.na(df[[col]])] <- mode_value
}

# Drop rows with any remaining NaN values
df <- na.omit(df)

# Check for missing values again
cat("Missing values after imputation and omitting rows:\n")
cat("Missing values in MPCE_URP:", sum(is.na(df$MPCE_URP)), "\n")
cat("Missing values in Whether_owns_any_land:", sum(is.na(df$Whether_owns_any_land)), "\n")
cat("Missing values in hhdsz:", sum(is.na(df$hhdsz)), "\n")
cat("Missing values in Religion:", sum(is.na(df$Religion)), "\n")
cat("Missing values in Social_Group:", sum(is.na(df$Social_Group)), "\n")
cat("Missing values in Regular_salary_earner:", sum(is.na(df$Regular_salary_earner)), "\n")

# Convert the target variable to binary based on the specified condition
df$MPCE_URP <- ifelse(df$MPCE_URP < 420, 0, 1)

# Convert categorical variables to factors and then to numeric
df$Whether_owns_any_land <- as.numeric(as.factor(df$Whether_owns_any_land))
df$Religion <- as.numeric(as.factor(df$Religion))
df$Social_Group <- as.numeric(as.factor(df$Social_Group))
df$Regular_salary_earner <- as.numeric(as.factor(df$Regular_salary_earner))

# Define the independent variables (X) and the dependent variable (y)
X <- df %>%
  select(Whether_owns_any_land, hhdsz, Religion, Social_Group, Regular_salary_earner)
X <- cbind(1, X)  # Add a constant term for the intercept
y <- df$MPCE_URP

# Ensure all columns in X are numeric
X <- as.matrix(sapply(X, as.numeric))

# Define the Tobit model function
tobit_loglike <- function(params) {
  beta <- params[1:(length(params)-1)]
  sigma <- params[length(params)]
  XB <- as.matrix(X) %*% beta
  cens <- (y == 0) + (y == 1)
  uncens <- 1 - cens
  ll <- numeric(length(y))
  
  ll[cens == 1] <- log(dnorm(y[cens == 1], mean = XB[cens == 1], sd = sigma))
  ll[uncens == 1] <- log(dnorm(y[uncens == 1], mean = XB[uncens == 1], sd = sigma))
  
  return(-sum(ll))
}

# Initial parameter guesses
start_params <- c(rep(0, ncol(X)), 1)

# Fit the Tobit model
tobit_results <- maxLik(tobit_loglike, start = start_params, method = "BFGS")

# Print the summary of the model
summary(tobit_results)
