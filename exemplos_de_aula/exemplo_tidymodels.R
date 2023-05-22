# Pacotes ------------------------------------------------------------------

library(ggplot2)
library(skimr)
library(tidymodels)

# Dados -------------------------------------------------------------------
data("diamonds")

# ---------------------------------------------------------------------
glimpse(diamonds)
skim(diamonds)

qplot(x, log(price), data = diamonds)
qplot(price, data = diamonds, geom = "histogram")
qplot(x, data = diamonds, geom = "histogram")

# Precisamos passar pro R:
# 1. A f que queremos usar
# 2. Ajustar a f para um conjunto de dados

# ----------------------------------------------------------------------
# Passo 1: Especificações de
# a) a f (a hipótese) com seus respectivos hiperparâmetros;
# b) o pacote 'motor' (engine);
# c) a tarefa/modo ("regression" ou "classification").

# Porque o tidymodels é legal?

# no dplyr faríamos:

# dados %>%
# mutate(nova_coluna = coluna_anterior+1) %>%
# group_by(grupo) %>%
# summarise(media_coluna = mean(nova_coluna))

# Essas funções sao do parsnip.
especificacao_modelo <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")

# essas definições são redundantes, mas servem pra deixar
# claro o que estamos fazendo.

especificacao_modelo

# Outros exemplos...

# especificacao_modelo <- decision_tree() %>%
#    set_engine("rpart") %>%
#    set_mode("regression")
#
# especificacao_modelo <- rand_forest() %>%
#   set_engine("ranger") %>%
#   set_mode("regression")


# --------------------------------------------------------------------
# Passo 2: Ajuste do modelo

diamonds <- diamonds %>% mutate(
  log_price = log(price),
  flag_maior_que_8 = ifelse(x > 8, "sim", "nao")
)

modelo <- especificacao_modelo %>%
  fit(log_price ~ x, data = diamonds)
# essa é a f

print(modelo)

# esses dois são iguais
reg1 <- extract_fit_engine(modelo)
reg2 <- lm(price ~ x, data = diamonds)

# --------------------------------------------------------------------
# Passo 3: Analisar as previsões

novo_x <- data.frame(x = 7)
predict(modelo, new_data = novo_x)
# ^ esse é o "programa que se programou sozinho"

so_o_x <- diamonds %>% select(x, flag_maior_que_8)
pred <- predict(modelo, new_data =  so_o_x)


# aqui só vamos usar tidyverse p/ baixo

diamonds_com_previsao <- diamonds %>%
  add_column(pred) %>%
  mutate(.pred_price = exp(.pred))

# Pontos observados + curva da f
diamonds_com_previsao %>%
  filter(x > 0) %>%
  sample_n(1000) %>%
  ggplot() +
  geom_point(aes(x, price), alpha = 0.3) +
  geom_point(aes(x, .pred_price), color = "red") +
  theme_bw()

# Observado vs Esperado
diamonds_com_previsao %>%
  filter(x > 0) %>%
  ggplot() +
  geom_point(aes(.pred_price, price)) +
  geom_abline(slope = 1, intercept = 0, colour = "purple", size = 1) +
  theme_bw()

diamonds_com_previsao %>%
  filter(x > 0) %>%
  sample_n(1000) %>%
  ggplot() +
  geom_point(aes(price, price - .pred_price))

# Como quantificar a qualidade de um modelo?

library(yardstick)

metrics <- metric_set(rmse, mae, rsq)

# Métricas de erro
diamonds_com_previsao %>%
  metrics(truth = price, estimate = .pred_price)


diamonds_com_previsao %>%
  mae(truth = price, estimate = .pred_price)
diamonds_com_previsao %>%
  rsq(truth = price, estimate = .pred_price)
diamonds_com_previsao %>%
  rmse(truth = price, estimate = .pred_price)


