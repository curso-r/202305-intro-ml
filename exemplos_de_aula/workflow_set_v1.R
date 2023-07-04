# Pacotes ------------------------------------------------------------------

library(ggplot2)
library(tidymodels)
library(ISLR2)

# Dados -------------------------------------------------------------------
data("Hitters")
#Hitters <- na.omit(Hitters)
help(Hitters)

# base treino e teste -----------------------------------------------------
set.seed(123)
hitters_initial_split <- Hitters %>% drop_na() |> initial_split(3/4)

hitters_train <- training(hitters_initial_split)
hitters_test <- testing(hitters_initial_split)

# Reamostragem ------------------------------------------------------------

hitters_resamples <- vfold_cv(hitters_train, v = 5)

# Recipes -----------------------------------------------------

## Data prep

hitters_lr_recipe <- recipe(Salary ~ ., data = hitters_train) %>%
  step_normalize(all_numeric(), -Salary) |>
  step_dummy(all_nominal()) |>
  step_interact(~starts_with("League"):starts_with("Division")) |>
  step_interact(~starts_with("Division"):Hits)

hitters_lr_recipe_2 <- recipe(Salary ~ ., data = hitters_train) %>%
  step_normalize(all_numeric(), -Salary) |>
  step_dummy(all_nominal()) |>
  step_bs(HmRun, degree= 3)

hitters_recipe_svm <- recipe(Salary ~ ., data = hitters_train) %>%
  step_pca(all_numeric(), -Salary) |>
  #step_normalize(all_numeric(), -Salary) |>
  step_dummy(all_nominal())

hitters_rf_recipe <- recipe(Salary ~ ., data = hitters_train) |>
  step_novel(all_nominal_predictors()) |>
  step_zv(all_predictors())

## Modelos

hitters_lr_model <- linear_reg(penalty = tune(), mixture = 1) %>%
  set_mode("regression") %>%
  set_engine("glmnet")

hitter_lr_svm <- svm_linear(cost = tune(), margin = tune()) %>%
  set_mode("regression") %>%
  set_engine("LiblineaR")

hitters_rf_model <- rand_forest(mtry = tune()) %>%
  set_mode("regression") %>%
  set_engine("ranger")

recipes <- list(SP = hitters_lr_recipe,
                CP = hitters_lr_recipe_2,
                PCA = hitters_recipe_svm
                #receita_arvore = hitters_rf_recipe
                )

modelos <- list(
  GAU = hitters_lr_model,
  GAU = hitters_lr_model,
  SVM = hitter_lr_svm
  )

todos_os_workflows <- workflow_set(
  preproc = recipes,
  modelos, cross = FALSE)

ajustado <- todos_os_workflows |>
  workflow_map(resamples = hitters_resamples, grid = 10, verbose = TRUE)

library(ggrepel)

autoplot(ajustado, select_best = TRUE) +
  geom_label_repel(aes(label = wflow_id))

extract_workflow_set_result(ajustado, "sem_polinomio_regressao") |>
  collect_metrics()

extract_workflow_set_result(ajustado, "com_polinomio_regressao") |>
  collect_metrics()
