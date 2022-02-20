library(dplyr)
library(pbapply)
set.seed(1234)

fpath <- dir("../data", pattern = "^[a-zA-Z]", full.names = TRUE)
fname <- dir("../data", pattern = "^[a-zA-Z]")

# Set desired data set sizes
DS_sizes <- c(20*(1:5), 150, 200, 250, 300, 400, 500)

# Generate mutually exclusive data sets for each organism and each size
mapply(fp = fpath, fn = fname, 
       MoreArgs = list(DS_sizes = DS_sizes),
       FUN = function(fp, fn, DS_sizes){
         # Read organism data
         X <- readRDS(paste0(fp, "/01_training.rds")) %>%
           as_tibble()
         
         # Extract unique peptide ids and reshuffle their order
         ids <- X %>%
           group_by(Info_epitope_id) %>%
           summarise(Class = first(Class)) %>%
           slice_sample(prop = 1)
         
         # Get class balance
         bal <- as.numeric(table(ids$Class) / sum(table(ids$Class)))
         
         for (i in seq_along(DS_sizes)){
           message("Making datasets for ", fn, ": ", DS_sizes[i], " peptides")
           
           if(!dir.exists(sprintf("%s/splits/%03dpeptides", fp, DS_sizes[i]))){
             dir.create(sprintf("%s/splits/%03dpeptides", fp, DS_sizes[i]), 
                        recursive = TRUE)
           }
           
           nneg = round(DS_sizes[i] * bal[1])
           npos = DS_sizes[i] - nneg
           k <- kpos <- kneg <- 1
           while(TRUE){
             if (k==1 || !(k%%10)) message("Building replicate: ", k)
             myIds <- rbind(
               filter(ids, Class == 1)[kpos:(kpos + npos - 1), ],
               filter(ids, Class == -1)[kneg:(kneg + nneg - 1), ])
             myRep <- X %>% filter(Info_epitope_id %in% myIds$Info_epitope_id)
             saveRDS(myRep, 
                     sprintf("%s/splits/%03dpeptides/replicate%03d.rds", 
                             fp, DS_sizes[i], k))
             
             kpos <- kpos + npos
             kneg <- kneg + nneg
             k    <- k + 1
             
             # If there are not enough ids anymore, break
             noPos <- (kpos + npos - 1) > sum(ids$Class == 1)
             noNeg <- (kneg + nneg - 1) > sum(ids$Class == -1)
             if(noPos || noNeg) break
           }
         }
         return(TRUE)
       })


# Generate Hybrid datasets based on the organism-specific ones above
# (note: this could have been done in the same mapply call above, but
#  I only thought of it afterwards, so... )
set.seed(1234)
lapply(fpath, 
       FUN = function(fp){
         # Read heterogeneous data
         Y <- readRDS(paste0(fp, "/df_heterogeneous.rds")) %>%
           as_tibble()
         files <- dir(paste0(fp, "/splits"), pattern = ".rds", 
                      full.names = TRUE, recursive = TRUE)
         
         pbapply::pblapply(files,
                           function(myfile, Y){
                             x    <- readRDS(myfile)
                             npep <- length(unique(x$Info_epitope_id))
                             idx1 <- sample(unique(Y$Info_epitope_id), size = npep)
                             idx2 <- sample(unique(Y$Info_epitope_id), size = 1000 - npep)
                             idx3 <- sample(unique(Y$Info_epitope_id), size = 1000)
                             
                             y1 <- Y %>%
                               filter(Info_epitope_id %in% idx1) %>%
                               rbind(x)
                             y2 <- Y %>%
                               filter(Info_epitope_id %in% idx2) %>%
                               rbind(x)
                             
                             mfp <- gsub("/replicate[0-9]+\\.rds", "", myfile)
                             if(!dir.exists(gsub("splits", "hybrid/double_peps", mfp))){
                               dir.create(gsub("splits", "hybrid/double_peps", mfp), recursive = TRUE)
                             }
                             if(!dir.exists(gsub("splits", "hybrid/1000_peps", mfp))){
                               dir.create(gsub("splits", "hybrid/1000_peps", mfp), recursive = TRUE)
                             }
                             if(!dir.exists(gsub("splits", "heter", mfp))){
                               dir.create(gsub("splits", "heter", mfp), recursive = TRUE)
                             }
                             saveRDS(y1, gsub("splits", "hybrid/double_peps", myfile))
                             saveRDS(y2, gsub("splits", "hybrid/1000_peps", myfile))
                             
                             # y3 <- Y %>% 
                             #   filter(Info_epitope_id %in% idx3)
                             # saveRDS(y3, gsub("splits", "heter", myfile))
                             return(TRUE)
                           }, 
                           Y = Y, cl = parallel::detectCores() - 1)
         return(TRUE)
       })
