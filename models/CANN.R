train_CANN = function(dt,
                      y,
                      vdt,
                      w, # weights
                      n_layers = 2, # Layers excluding the input layer, the output layer and the GLM.
                      n_units = c(20, 15), # dim of the Layers excluding the input layer, the output layer and the GLM.
                      n_activ = c('relu','relu'),
                      lr = 0.001, # parameter not used
                      bs = 2^12,
                      ep=10,
                      # loss_func='poisson',
                      underweigthing=0,
                      verbo=10
                      
){
  
  w_learn_exp=exp(w) #returning to glm_fit on traning set.
  size_hom=sum(y)/sum(w_learn_exp)
  
  
  if(var(c(n_layers,length(n_units),length(n_activ))) != 0){stop("inconsistent n_h_layers, n_activ, n_units")}
  
  # neural network structure
  Design  <- layer_input(shape = c(dim(dt)[2]), dtype = 'float32', name = 'design') #Input layer
  LogVol <- layer_input(shape = c(1), dtype = 'float32', name = 'LogVol')
  
  # "Hidden layers" - all the layer excluding the input layer, the output layer and the glm
  for (i in 1:length(n_units)) {
    
    if (i==1) {
      Network <- Design %>%    
        layer_dense(units = n_units[i], activation = n_activ[i], name=paste0('layer', (i)))
    }
    else {
      Network <-Network %>%    
        layer_dense(units = n_units[i], activation = n_activ[i], name=paste0('layer', (i)))
    }
    
    
    if (i==length(n_units)) {
      Network <- Network %>%  
        layer_dense(units = 1, activation = 'linear', name = 'Network',
                    weights = list(array(0, dim = c(n_units[i], 1)), array(log(size_hom), dim = 1)))
    }
  }
  
  Response <- list(Network, LogVol) %>%
    layer_add%>% 
    layer_dense(units = 1, activation = k_exp, name = 'Response', trainable = FALSE,
                # weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))
                weights = list(array(1, dim = c(1, 1)), array(0, dim = 1)))
  
  model_cann <- keras_model(inputs = c(Design, LogVol), outputs = c(Response))
  
  
  model_cann %>% compile(
    loss = custom_poisson,
    optimizer= optimizer_adam(learning_rate = lr))
    
    fit <- model_cann %>% fit(list(as.matrix(dt),w), as.matrix(as.numeric(y)),
                              validation_data= vdt,
                              shuffle = T,
                              epochs = ep, 
                              batch_size  = bs,
                              verbose = verbo,
                              class_weigth=class_weigth_in,
                              callbacks = callback_early_stopping(monitor = "val_loss", patience = 10, restore_best_weights=TRUE)
    )
    
    plot(fit)
  
  return(model_cann) 
  
}
