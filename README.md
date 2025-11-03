# Bayesian Prediction of Online Shoppers’ Purchasing Intention

**Mini project for the Bayesian Data Analysis course**

Author: *Liang-Jen Huang*

---

This project applies **Bayesian logistic regression** to predict whether an online shopping session results in a purchase, based on the *Online Shoppers Purchasing Intention Dataset*.
Several models were compared, including pooled and hierarchical versions with **Bayesian LASSO** shrinkage.
Model performance was evaluated using **PSIS-LOO**, and the best results came from the hierarchical model with *Region* and *TrafficType* effects.

---

**Tools:** R, Stan (RStan), loo

**Dataset:** `online_shoppers_intention.csv` [UCI Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)

**Code:** `analysis.R`

**Report:** `BSDA_LiangJen.pdf`

**Slides:** `BSDA_LiangJen_Slide.pdf`
