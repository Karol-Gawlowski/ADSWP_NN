**ADSWP_NN**

Here you can find the code to the article "Gladiators Ready" from Gawlowski K., Condon J., Harrington J., Ruffini D. - published in "The Actuary" in March, available from [link](http://example.com "The Actuary")

**Repository Structure:**

1) the_actuary.R script references the other analysis from the repository to run the modelling pipeline. 

2) models/ folder contains three scripts with unifying training functions for CANN, LocalGLMnet and a neural net.

3) other

    * "init": contains the data work and a set of functions for encoding and results analysis
    * "init_py": used to access a Python environment with Tensorflow library
    * final_results.rds - contains the results reported in the article
    * freMTPL2freq.csv - dataset used.
