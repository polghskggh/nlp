library(dplyr)
library(ggplot2)
library(ggpubr)


metrics <- read.csv("../classifier/metrics.csv", sep=",")
metrics$epoch <- c(1:3)

f1 <- ggplot(head(metrics, 3),aes(x = epoch, y = f1)) + geom_line(color = "red") +
  scale_x_continuous(breaks = 1:3) + labs(y = "F1 score") +
    scale_y_continuous(limits = c(0.78,0.83)) +
  theme(axis.title.x=element_blank(), axis.text.x=element_blank()) +
  theme(axis.text.y=element_blank())



acc <- ggplot(head(metrics, 3),aes(x = epoch, y = accuracy)) + geom_line(color = "red") +
  scale_x_continuous(breaks = 1:3) + labs(y = "Accuracy") +
    scale_y_continuous(limits = c(0.78,0.83)) +
  theme(axis.title.x=element_blank(), axis.text.x=element_blank())


prec <- ggplot(head(metrics, 3),aes(x = epoch, y = precision)) + geom_line(color = "red") +
  scale_x_continuous(breaks = 1:3) + labs(y = "Precision") +
    scale_y_continuous(limits = c(0.78,0.83))


rec <- ggplot(head(metrics, 3),aes(x = epoch, y = recall)) + geom_line(color = "red") +
  scale_x_continuous(breaks = 1:3) + labs(y = "Recall") +
  scale_y_continuous(limits = c(0.78,0.83)) +
  theme(axis.text.y=element_blank())


ggarrange(acc, f1, prec, rec, nrow = 2, ncol = 2)



metrics2 <- read.csv("../seq2seq/metrics.csv", sep=",")
ggplot(head(metrics2, 2), aes(x = epoch, y = eval_loss)) + geom_line(color = "red") +
  scale_x_continuous(breaks = 1:2)
