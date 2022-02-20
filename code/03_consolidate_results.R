library(dplyr)
library(tidyr)
library(ggplot2)
library(pbapply)
source("utils.R") # load performance calculation function

# All file paths
paths <- dir("../data/Results", pattern = ".csv", 
             full.names = TRUE, recursive = TRUE)

# Read all results
df <- pbapply::pblapply(paths, 
                        FUN = function(fp){
                          ml  <- strsplit(fp, split = "/|_")[[1]]
                          read.csv(fp, header = TRUE) %>%
                            mutate(Organism   = ml[4],
                                   Type       = ml[5],
                                   DS.size    = as.numeric(gsub("peptides", "", ml[6])), 
                                   Replicate  = as.numeric(gsub("replicate", "", ml[7]))) %>%
                            select(Organism, Type, DS.size, Replicate,
                                   starts_with("Info"),
                                   Class = true.test.labels,
                                   pred  = Predictions,
                                   prob  = Probabilities)}, 
                        cl = parallel::detectCores() - 1) %>%
  bind_rows() %>%
  # Consolidate prediction results by peptide
  group_by(Organism, Type, DS.size, Replicate, Info_epitope_id) %>%
  summarise(Class = first(Class),
            Prob = mean(prob),
            Pred = ifelse(Prob > 0.5, 1, -1)) %>%
  as_tibble() %>%
  group_by(Organism, Type, DS.size, Replicate)


# Calculate performance
perf <- df %>%
  group_map(~ calc_perf(.x$Class, .x$Pred, .x$Prob)) %>%
  bind_rows() %>%
  bind_cols(group_keys(df)) %>%
  as_tibble() %>%
  select(Info_organism  = Organism,
         Info_Type      = Type,
         Info_size      = DS.size,
         Info_Replicate = Replicate,
         everything()) %>%
  mutate(Info_size = ifelse(Info_size == 1000, 0, Info_size),
         Info_Type = ifelse(Info_Type == "Heter", "HybridB", Info_Type))

saveRDS(perf, "../output/performance-raw.rds")

# Calculate performance means and standard errors
perf_stats <- perf %>%
  select(-Info_Replicate) %>%
  group_by(Info_organism, Info_Type, Info_size) %>%
  summarise(across(everything(),
                   .fns = list(Value  = mean,
                               StdErr = sd)),
            n = n()) %>%
  mutate(across(ends_with("StdErr"), ~.x/n)) %>%
  ungroup() %>%
  pivot_longer(-c("n", starts_with("Info")), names_to = "Variable", values_to = "Value") %>%
  rowwise() %>%
  mutate(Metric = toupper(strsplit(Variable, split = "_")[[1]][1]),
         Stat   = strsplit(Variable, split = "_")[[1]][2]) %>%
  select(-Variable) %>%
  pivot_wider(names_from = Stat, values_from = Value)

saveRDS(perf_stats, "../output/performance.rds")


# # Get number of replicates for each Org/DS size
# df %>%
#   filter(Type == "OrgSpec") %>%
#   group_by(Organism, DS.size) %>%
#   summarise(Reps = max(Replicate)) %>%
#   print(n = Inf)



# Retrieve results form other predictors AND heterogeneous models
fpath <- dir("../data", pattern = "^[a-zA-Z]", full.names = TRUE)[-4]
fname <- dir("../data", pattern = "^[a-zA-Z]")[-4]

cmp_res <- mapply(
  function(fp, fn){
    readRDS(paste0(fp, "/analysis.rds"))$myperf_pep %>%
      filter(NoLeak == FALSE) %>%
      mutate(Organism = fn,
             Metric = as.character(Metric),
             Metric = gsub("ACCURACY", "ACC", Metric),
             Method = as.character(Method)) %>%
      select(Organism, everything(), -c(Type, NoLeak)) %>%
      dplyr::filter(!grepl("SVM", Method))}, 
  fpath, fname,
  SIMPLIFY = FALSE) %>%
  bind_rows() %>%
  as_tibble()

balacc <- cmp_res %>%
  filter(Metric %in% c("SENS", "SPEC")) %>%
  group_by(Organism, Method) %>%
  summarise(Metric = "BALACC",
            Value  = sum(Value) / 2,
            Mean   = sum(Mean) / 2,
            StdErr = NA)

cmp_res <- bind_rows(cmp_res, balacc)

saveRDS(cmp_res, "../output/baseline_performances.rds")
