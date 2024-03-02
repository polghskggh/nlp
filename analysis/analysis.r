library(dplyr)
library(ggplot2)

read <- function(filename)
{
    data <- read.csv(paste("../data/", filename, ".csv", sep=""), sep=";")
    ggplot(data, aes(x = emotion)) + geom_bar() + y_lim(min = 0, max = count(data))
}




read("train")


