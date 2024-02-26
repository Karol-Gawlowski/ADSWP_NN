train_LocalGLMnet = function(dt,
                             y,
                             vdt, #validation set
                             n_layers = 3, #Layers excluding the input layer, the output layer and the GLM.
                             n_units = c(20, 15, 10), #dim of the Layers excluding the input layer, the output layer and the GLM.
                             n_activ = c('tanh','tanh','tanh'),
                             n_dropout = c(F,F),
                             lr = 0.001, #learning rate
                             bs = 2^14, #batch size
                             ep = 500, #epochs
                             verbo = 0 #verbose
){
  
  log_size_hom <- log(mean(y))
  
  if(var(c(n_layers,length(n_units),length(n_activ))) != 0){stop("inconsistent n_h_layers, n_activ, n_units")}
  
  # neural network structure
  Design  <- layer_input(shape = c(dim(dt)[2]), dtype = 'float32', name = 'design') #Input layer
  
  # "Hidden layers" - all the layer excluding the input layer, the output layer and the glm
  for (i in 1:length(n_units)) {
    
    if (i==1) {
      Attention <- Design %>%    
        layer_dense(units = n_units[i], activation = n_activ[i], name=paste0('layer', (i)))
    }
    else if (i>1) {
      Attention = Attention %>%    
        layer_dense(units = n_units[i], activation = n_activ[i], name=paste0('layer', (i)))
    }
    
    if(n_dropout[i]){Attention = Attention %>% layer_dropout(rate = 0.2)}
    
  }
  
  #output layer definition
  Attention =Attention %>%
    layer_dense(units=c(dim(dt)[2]), activation='linear', name='attention')
  
  #GLM with fixed loglink function
  Output <- list(Design, Attention) %>% layer_dot(name='LocalGLM', axes=1) %>% 
    layer_dense(
      units=1, activation='exponential', name='output',
      weights=list(array(0, dim=c(1,1)), array(log_size_hom, dim=c(1)))
    )
  
  model_localGLMnet <- keras_model(inputs = list(Design), outputs = c(Output))

  model_localGLMnet %>% 
    compile(
    loss = custom_poisson,
    optimizer= optimizer_adam(learning_rate = lr)
  ) 
  
  fit_LocalGLMnet <- model_localGLMnet %>% 
    fit(
      list(as.matrix(dt)), list(as.matrix(as.numeric(y))),
      validation_data = vdt,
      batch_size = bs,
      epochs = ep,
      shuffle = T,
      callbacks = callback_early_stopping(monitor = "val_loss", 
                                          patience = 15,
                                          restore_best_weights=TRUE,
                                          verbose = verbo)
    )
  
  return(model_localGLMnet)
  
}
