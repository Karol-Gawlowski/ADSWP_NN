# R - PY connection
use_condaenv(condaenv = "ADSWPNL",required = T) 
set_random_seed(seed, disable_gpu = FALSE)
py_module_available("tensorflow")
is_keras_available()

py_config()
tensorflow::tf_version()
