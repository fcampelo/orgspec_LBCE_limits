library(dplyr)
library(ggplot2)
library(ggforce)

# Load consolidated results
myperf <- readRDS("../output/performance.rds") %>%
  filter(!(Metric %in% c("F1", "SPEC", "ACC"))) %>%
  rename(Organism = Info_organism,
         Peptides = Info_size, 
         Training = Info_Type) %>%
  mutate(Training = factor(Training, 
                           levels = unique(Training), 
                           labels = c("Hybrid - A",
                                      "Hybrid - B",
                                      "Organism-specific")),
         Metric = gsub("BALACC", "BAL.ACC", Metric))

# Load comparison results
cmp_res <- readRDS("../output/baseline_performances.rds") %>%
  filter(!(Metric %in% c("F1", "SPEC", "ACC"))) %>%
  mutate(Metric = gsub("BALACC", "BAL.ACC", Metric))

cmp_models <- filter(cmp_res, grepl("RF-", Method)) %>%
  mutate(Method = factor(Method, levels = unique(Method),
                         labels = c("OS-full", "OS-full+6k", "Heter 6k")))

cmp_res <- cmp_res %>% 
  filter(!grepl("RF-", Method)) %>%
  mutate(Method = factor(Method, 
                         levels = unique(Method), 
                         labels = c("ABCPred", "Bepipred2", "iBCE-EL", "LBtope")))

# ============================================================================ #

tiff(filename = "../figures/res-all.tif",
     width = 10, height = 14, units = "in", res = 150)

# Plot
myperf %>%
  #
  # Basic plot object
  ggplot(aes(x = Peptides, y = Value, 
             ymin = Value - StdErr,
             ymax = Value + StdErr)) +
  #
  # Point estimates/error bars and connecting dotted lines
  geom_pointrange(aes(colour = Training, shape = Training),
                  size = .5) +
  geom_line(aes(colour = Training), alpha = .5, lty = 3,
            show.legend = FALSE) +
  # 
  # Performance baselines - line segments
  geom_segment(data = cmp_res,
               aes(y = Value, yend = Value, 
                   x = -100, xend = 490),
               alpha = .5, lwd = .5, 
               col = "#00888888") +
  geom_segment(data = cmp_models,
               aes(y = Value, yend = Value, 
                   x = 0, xend = 600),
               alpha = .5, lwd = .5, 
               col = "#88008888") +
  #
  # Performance baselines - text
  geom_text(data = cmp_res,
            aes(x = 495, y = Value, label = Method),
            vjust = .5, hjust = 0, size = 2, 
            col = "#008888") +
  geom_text(data = cmp_models,
            aes(x = -5, y = Value, label = Method),
            vjust = .5, hjust = 1, size = 2, 
            col = "#880088") +
  #
  # Graphical adjustments:
  theme_light() +
  theme(strip.text = element_text(colour = "black", face = "bold"),
        panel.border = element_rect(color = "black", fill = NA, size = .5),
        legend.position = "top", 
        legend.title = element_blank(),
        axis.text.x = element_text(size = 8)) + 
  guides(colour = guide_legend(override.aes = list(size=.75))) +
  labs(x = "Number of organism-specific peptides in training set",
       y = "Estimated performance on hold-out set") +
  #
  # Scale adjustments
  scale_x_continuous(breaks = 50*(0:10), 
                     minor_breaks = NULL,
                     guide = guide_axis(check.overlap = TRUE),
                     expand = expansion(add = c(0, 0)),
                     limits = c(-100, 600)) + # <- to get space for the text on the sides
  scale_shape_manual(values = c(4, 15, 17)) +
  #
  #Facetting
  facet_grid(Metric ~ Organism, scales = "free_y")

dev.off()

#ggsave(filename = "../figures/res-all.png", width = 10, height = 14, units = "in")


# Break in two pages
# Page 1
last_plot() + 
  facet_grid_paginate(Metric ~ Organism, nrow = 3, ncol = 3, 
                      scales = "free_y", page = 1)
ggsave(filename = "../figures/res01.png", width = 10, height = 8, units = "in")

# Page 2
last_plot() + 
  facet_grid_paginate(Metric ~ Organism, nrow = 3, ncol = 3, 
                      scales = "free_y", page = 2)
ggsave(filename = "../figures/res02.png", width = 10, height = 8, units = "in")


# ======
# Generate LaTeX table(s)
library(kableExtra)
library(tidyr)

for (i in seq_along(unique(myperf$Organism))){
  myperf %>%
    filter(Organism == unique(myperf$Organism)[i]) %>%
    mutate(Value = paste0("$", round(Value, 2), "\\pm ", round(StdErr, 3), "$"),
           Training = gsub("\\ ", "", Training),
           Training = gsub("Organism-specific", "OrgSpec", Training)) %>%
    select(-StdErr, -n, -Organism) %>%
    pivot_wider(id_cols = c("Training", "Peptides"), 
                names_from = "Metric", values_from = "Value") %>%
    arrange(desc(Training)) %>%
    kable(format = "latex", escape = FALSE, caption = unique(myperf$Organism)[i]) %>%
    print()
}


  



