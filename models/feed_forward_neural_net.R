train_ff_nn = function(dt,
                       y,
                       vdt, #validation
                       n_layers = 2, #layers
                       n_units = c(16,4), #units
                       n_activ = c("sigmoid","sigmoid"), #activation functions
                       n_dropout = c(F,F), #dropout
                       lr = 0.001, #learning rate
                       bs = 2^13, #batch size
                       ep = 500, #epochs
                       verbo = 0 #verbose
){
  
  tensorflow::set_random_seed(1,disable_gpu = F)
  
  if(var(c(n_layers,length(n_units),length(n_activ))) != 0){stop("inconsistent n_layers, n_activ, n_units")}
  
  model <- keras_model_sequential(input_shape = ncol(dt))
  
  for (i in 1:n_layers){
    
    model = model %>% 
      layer_dense(units = n_units[i], 
                  activation = n_activ[i],
                  kernel_initializer = keras::initializer_random_normal(stddev = 2))
    
    if(n_dropout[i]){model = model %>% layer_dropout(rate = 0.2)}
    
  }
  
  model = model %>%  layer_dense(units = 1, activation = 'relu')
  
  model %>%
    compile(
      loss = custom_poisson,
      metrics = c("poisson"),
      optimizer = optimizer_adam(learning_rate = lr)
    ) %>%
    fit(
      dt,
      y,
      validation_data = vdt,
      batch_size = bs,
      epochs = ep,
      shuffle = T,
      callbacks = callback_early_stopping(monitor = "val_loss", 
                                          patience = 8,
                                          restore_best_weights=TRUE,
                                          verbose = verbo))
  
  return(model)
  
}
