# ============================================================================ #
# function to calculate performance values
calc_perf <- function(truth, pred, prob){
  out <- data.frame(sens = NA, spec = NA, ppv = NA, npv = NA, 
                    f1 = NA, mcc = NA, auc = NA)
  if(all(is.na(pred))) return(out)
  
  TN   <- as.numeric(sum(truth == -1 & pred == -1))
  FN   <- as.numeric(sum(truth == 1 & pred == -1))
  FP   <- as.numeric(sum(truth == -1 & pred == 1))
  TP   <- as.numeric(sum(truth == 1 & pred == 1))
  out$acc    <- (TP + TN)/(TP + TN + FP + FN)
  out$sens   <- TP/(TP + FN + 1e-8)
  out$spec   <- TN/(TN + FP + 1e-8)
  out$ppv    <- TP/(TP + FP + 1e-8)
  out$npv    <- TN/(TN + FN + 1e-8)
  out$f1     <- 2 * TP/(2 * TP + FP + FN + 1e-8)
  out$balacc <- (out$sens + out$spec) / 2
  out$mcc  <- (TP * TN - FP * FN) / sqrt(1e-8 + ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
  
  # Calculate AUC
  tr <- sort(unique(prob), decreasing = TRUE)
  x  <- lapply(tr, 
               function(thres, prob, truth){
                 pred <- ifelse(prob >= thres, 1, -1)
                 data.frame(tpr = sum(pred == 1 & truth == 1) / sum(truth == 1), 
                            fpr = sum(pred == 1 & truth == -1) / sum(truth == -1))
               }, 
               prob = prob, truth = truth) %>%
    bind_rows() %>%
    summarise(auc = sum(.5 * (fpr - lag(fpr)) * (tpr + lag(tpr)), na.rm = TRUE))
  
  out$auc <- x$auc
  return(out)
}
# ============================================================================ #