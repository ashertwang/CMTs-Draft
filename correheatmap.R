require(ggplot2)
require(reshape2)
require(DescTools)  # Cramér's V
require(dplyr)

data <- read.csv("/heatmap_main.csv", header = TRUE)

data$ER.Status <- as.factor(data$ER.Status)
data$HER2.Status <- as.factor(data$HER2.Status)
data$Pena.Grade..PG. <- as.factor(data$Pena.Grade..PG.)
data$Metabolic.Grade..MG. <- as.factor(data$Metabolic.Grade..MG.)
data$Tumor.State <- as.factor(data$Tumor.State)

impute_missing_values <- function(df) {
  # Impute numerical variables with the mean
  df <- df %>%
    mutate(across(where(is.numeric), ~ifelse(is.na(.), mean(., na.rm = TRUE), .)))
  
  # Impute categorical with mode
  mode_impute <- function(x) {
    ux <- unique(x)
    ux[which.max(tabulate(match(x, ux)))]
  }
  df <- df %>%
    mutate(across(where(is.factor), ~ifelse(is.na(.), mode_impute(.), .)))
  
  return(df)
}

data <- impute_missing_values(data)

cor_matrix <- matrix(NA, ncol = ncol(data), nrow = ncol(data))
rownames(cor_matrix) <- colnames(data)
colnames(cor_matrix) <- colnames(data)

# Calculate correlations based on data type
for (i in 1:ncol(data)) {
  for (j in 1:ncol(data)) {
    if (is.numeric(data[, i]) && is.numeric(data[, j])) {
      cor_matrix[i, j] <- cor(data[, i], data[, j], use = "complete.obs")
    } else if (is.factor(data[, i]) && is.factor(data[, j])) {
      cor_matrix[i, j] <- CramerV(table(data[, i], data[, j]))
    }
  }
}

# Filter weak correlation (abs < 0.1)
cor_matrix[abs(cor_matrix) < 0.1] <- NA

melted_cor_matrix <- melt(cor_matrix, na.rm = TRUE)

ggplot(data = melted_cor_matrix, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white",
                       midpoint = 0, limit = c(-1, 1), name = "Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 8),
        axis.text.y = element_text(size = 8),
        axis.title = element_blank(),
        panel.border = element_rect(fill = NA, color = "black", size = 1),
        plot.margin = unit(c(1, 1, 1, 1), "cm")) +
  coord_fixed() +
  labs(title = "CMTs Correlation Heatmap") +
  theme(plot.title = element_text(size = 15, face = "bold", hjust = 0.5))