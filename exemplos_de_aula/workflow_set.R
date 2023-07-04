# Pacotes ------------------------------------------------------------------

library(tidymodels)
library(ISLR)
library(tidyverse)
library(modeldata)
library(pROC)
library(vip)

# Base de treino e teste --------------------------------------------------

set.seed(1)
credit_initial_split <- initial_split(credit_data, strata = "Status", prop = 0.75)

credit_train <- training(credit_initial_split)
credit_test  <- testing(credit_initial_split)

# Reamostragem ------------------------------------------------------------

credit_resamples <- vfold_cv(credit_train, v = 5, strata = "Status")

# Recipes -----------------------------------------------------------------

credit_lr_input_linear <- recipe(Status ~ ., data = credit_train) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_novel(all_nominal_predictors()) %>%
  step_impute_linear(Income, Assets, Debt, impute_with = imp_vars(Expenses)) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_poly(all_numeric_predictors(), degree = 9) %>%
  step_dummy(all_nominal_predictors()) |>
  step_interact(~starts_with("Seniority"):starts_with("Assets"))

credit_lr_input_median <- recipe(Status ~ ., data = credit_train) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_novel(all_nominal_predictors()) %>%
  step_impute_median(Income, Assets, Debt) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_poly(all_numeric_predictors(), degree = 9) %>%
  step_dummy(all_nominal_predictors())

especificao_logistica <- logistic_reg(penalty = tune(), mixture = 1) |>
  set_mode("classification") |>
  set_engine("glmnet")

lista_de_recipes <- list(
  linear = credit_lr_input_linear,
  median = credit_lr_input_median
)

todos_os_workflows <- workflow_set(
  lista_de_recipes,
  list(LOGIT = especificao_logistica),
  cross = TRUE
)

todos_os_workflows$info[[1]]$preproc <- "input_linear"
todos_os_workflows$info[[2]]$preproc <- "input_mediana"

workflows_ajustados <- todos_os_workflows |>
  workflow_map(resamples = credit_resamples,
               verbose = TRUE, grid = 20)

autoplot(workflows_ajustados, select_best = FALSE)

workflows_ajustados |>
  extract_workflow_set_result("linear_LOGIT")
