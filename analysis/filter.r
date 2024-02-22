library(readr)
library(dplyr)


filter <- function(filename)
{
  data <- read_tsv(paste("../data/", filename, ".tsv", sep=""))
  
  data <- data %>% select(prompt, mood, mood_idx, sentiment, context)
  write.table(data, file = paste("../data/", filename, "-clean.tsv", sep=""), sep = "\t", 
              quote = FALSE, row.names = FALSE)
}


filter("test-emotion")
filter("train-emotion")
filter("val-emotion")


