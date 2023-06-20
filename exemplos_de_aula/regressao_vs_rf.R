# Pacotes ------------------------------------------------------------------

library(tidymodels)
library(ISLR)
library(tidyverse)
library(modeldata)
library(pROC)
library(vip)


# Bases de dados ----------------------------------------------------------

data("credit_data")
help(credit_data)
glimpse(credit_data) # German Risk

credit_data %>% count(Status)

# Base de treino e teste --------------------------------------------------

set.seed(1)
credit_initial_split <- initial_split(credit_data, strata = "Status", prop = 0.75)

credit_train <- training(credit_initial_split)
credit_test  <- testing(credit_initial_split)

# Reamostragem ------------------------------------------------------------

credit_resamples <- vfold_cv(credit_train, v = 5, strata = "Status")

# Exploratória ------------------------------------------------------------

# skimr::skim(credit_train)
# visdat::vis_miss(credit_train)
# credit_train %>%
#   select(where(is.numeric)) %>%
#   cor(use = "pairwise.complete.obs") %>%
#   corrplot::corrplot()


# Regressão Logística -----------------------------------------------------

## Data prep

credit_lr_recipe <- recipe(Status ~ ., data = credit_train) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_novel(all_nominal_predictors()) %>%
  step_impute_linear(Income, Assets, Debt, impute_with = imp_vars(Expenses)) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_poly(all_numeric_predictors(), degree = 9) %>%
  step_dummy(all_nominal_predictors())

## Modelo

credit_lr_model <- logistic_reg(penalty = tune(), mixture = 1) %>%
  set_mode("classification") %>%
  set_engine("glmnet")

## Workflow

credit_lr_wf <- workflow() %>%
  add_model(credit_lr_model) %>%
  add_recipe(credit_lr_recipe)

## Tune

grid_lr <- grid_regular(
  penalty(range = c(-4, -1)),
  levels = 20
)

credit_lr_tune_grid <- tune_grid(
  credit_lr_wf,
  resamples = credit_resamples,
  grid = grid_lr,
  metrics = metric_set(roc_auc, accuracy, recall)
)

autoplot(credit_lr_tune_grid)
show_best(credit_lr_tune_grid)

# Árvore de decisão -------------------------------------------------------

## Data prep

# credit_dt_recipe <- recipe(Status ~ ., data = credit_train) %>%
#   step_novel(all_nominal_predictors()) %>%
#   step_zv(all_predictors())

credit_rf_recipe <- recipe(Status ~ ., data = credit_train) |>
  step_impute_linear(Income, Assets, Debt, impute_with = imp_vars(Expenses)) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_novel(all_nominal_predictors()) |>
  step_zv(all_predictors())

## Modelo

# credit_dt_model <- decision_tree(
#   cost_complexity = tune(),
#   tree_depth = tune(),
#   min_n = tune()
# ) %>%
#   set_mode("classification") %>%
#   set_engine("rpart")

credit_rf_model <- rand_forest(
  mtry = tune()
) |>
  set_mode("classification") |>
  set_engine("ranger", importance = "impurity_corrected")

## Workflow

# credit_dt_wf <- workflow() %>%
#   add_model(credit_dt_model) %>%
#   add_recipe(credit_dt_recipe)

credit_rf_wf <- workflow() %>%
  add_model(credit_rf_model) %>%
  add_recipe(credit_rf_recipe)

## Tune

# grid_dt <- grid_regular(
#   cost_complexity(c(-15, -6)),
#   tree_depth(range = c(8, 17)),
#   min_n(range = c(20, 30)),
#   levels = 2
# )

grid_rf <- grid_regular(
  mtry(c(1, 5)),
  levels = 5
)

#doParallel::registerDoParallel(4)

# credit_dt_tune_grid <- tune_grid(
#   credit_dt_wf,
#   resamples = credit_resamples,
#   grid = grid_dt,
#   control = control_grid(verbose = TRUE),
#   metrics = metric_set(roc_auc, accuracy, recall)
# )

credit_rf_tune_grid <- tune_grid(
  credit_rf_wf,
  resamples = credit_resamples,
  grid = grid_rf,
  control = control_grid(verbose = TRUE),
  metrics = metric_set(roc_auc, accuracy, recall)
)

#doParallel::stopImplicitCluster()

autoplot(credit_rf_tune_grid)
show_best(credit_rf_tune_grid, "accuracy") |> View()
show_best(credit_lr_tune_grid, "accuracy") |> View()

show_best(credit_rf_tune_grid, "recall") |> View()
show_best(credit_lr_tune_grid, "recall") |> View()

show_best(credit_rf_tune_grid, "roc_auc") |> View()
show_best(credit_lr_tune_grid, "roc_auc") |> View()

collect_metrics(credit_rf_tune_grid)


# Desempenho dos modelos finais ----------------------------------------------

credit_lr_best_params <- select_best(credit_lr_tune_grid, "roc_auc")
credit_lr_wf <- credit_lr_wf %>% finalize_workflow(credit_lr_best_params)
credit_lr_last_fit <- last_fit(credit_lr_wf, credit_initial_split)

credit_rf_best_params <- select_best(credit_rf_tune_grid, "roc_auc")
credit_rf_wf <- credit_rf_wf %>% finalize_workflow(credit_rf_best_params)
credit_rf_last_fit <- last_fit(credit_rf_wf, credit_initial_split)


credit_test_preds <- bind_rows(
  collect_predictions(credit_lr_last_fit) %>% mutate(modelo = "lr"),
  collect_predictions(credit_rf_last_fit) %>% mutate(modelo = "rf")
)

## roc
credit_test_preds %>%
  group_by(modelo) %>%
  roc_curve(Status, .pred_bad) %>%
  autoplot()

## lift
credit_test_preds %>%
  group_by(modelo) %>%
  lift_curve(Status, .pred_bad) %>%
  autoplot()

# Variáveis importantes
credit_lr_last_fit_model <- credit_lr_last_fit$.workflow[[1]]$fit$fit
vip(credit_lr_last_fit_model)

credit_rf_last_fit_model <- credit_rf_last_fit$.workflow[[1]]$fit$fit
vip(credit_rf_last_fit_model)

# Guardar tudo ------------------------------------------------------------

write_rds(credit_lr_last_fit, "credit_lr_last_fit.rds")
write_rds(credit_lr_model, "credit_lr_model.rds")

# Modelo final ------------------------------------------------------------

credit_final_lr_model <- credit_lr_wf %>% fit(credit_data)
