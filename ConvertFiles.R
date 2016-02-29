library(xtable)
setwd("~/TarekGhaddarMastersWork")
opp_no_iter <- read.csv("opp_no_iter.csv",stringsAsFactors=FALSE)
print(xtable(opp_no_iter), type='latex')

opp_iter <- read.csv("opp_iter.csv",stringsAsFactors = FALSE)
print(xtable(caption = "The metric behavior of the first test case after 10 load balancing iterations.",opp_iter),type = 'latex')

opp_diff <- read.csv("opp_diff.csv",stringsAsFactors = FALSE)
print(xtable(caption = "The difference in metric behavior between no iteration and 10 iterations. The closer the z-value to zero, the better the improvement.",opp_diff),type = 'latex')

same_no_iter <- read.csv("same_no_iter.csv",stringsAsFactors=FALSE)
print(xtable(caption = "The metric behavior of the second test case after no load balancing iterations.",same_no_iter), type='latex')

same_iter <- read.csv("same_iter.csv",stringsAsFactors=FALSE)
print(xtable(caption= "The metric behavior of the second test case after 10 load balancing iterations.",same_iter),type = 'latex')

same_diff <- read.csv("same_diff.csv",stringsAsFactors = FALSE)
print(xtable(caption = "The difference in metric behavior between no iteration and 10 iterations. The closer the z-value to zero, the better the improvement.",same_diff),type = 'latex')

lattice_no_iter <- read.csv("lattice_no_iter.csv", stringsAsFactors = FALSE)
print(xtable(caption = "The metric behavior of the third test case after no load balancing iterations.",lattice_no_iter), type='latex')

lattice_iter <- read.csv("lattice_iter.csv", stringsAsFactors = FALSE)
print(xtable(caption= "The metric behavior of the third test case after 10 load balancing iterations.",lattice_iter),type = 'latex')

lattice_diff <- read.csv("lattice_diff.csv", stringsAsFactors = FALSE)
print(xtable(caption = "The difference in metric behavior between no iteration and 10 iterations. The closer the z-value to zero, the better the improvement.",lattice_diff),type = 'latex')
