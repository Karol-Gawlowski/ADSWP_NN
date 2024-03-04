library(tidyverse)
library(xgboost)
library(reticulate)
library(tensorflow)
library(keras)
library(mlrMBO)

slice = dplyr::slice
rename = dplyr::rename

seed = 2
set.seed(seed)

# load data objects
dt_list = list()

# Data
dt_list$fre_mtpl2_freq = read.csv("freMTPL2freq.csv") %>%  

  mutate(  Exposure = pmin(1, Exposure),
           ClaimNb = pmin(15, ClaimNb / Exposure)
           ) %>% 
  slice(sample(1:nrow(.),replace = F))

#Poisson Deviance - Loss function
poiss_loss = function(y_true,y_pred){
  y_true*log(y_true/y_pred)-(y_true-y_pred)
}

#Encoding Function
preproc = function(
    dt_frame,       # a dataframe
    y,              # target colname required if present in x or for target enc
    num = NULL,     # type of encoding for numericals - normalize, standardize: c("norm","stand") 
    cat = NULL,     # type of encoding for categoricals - one hot, target encoding, entity embedding, joint embedding: c("ohe","targ","entem","joint")
    bypass = NULL,  # string vector of column names to bypass preprocessing
    exclude = NULL, # columns to remove
    verbose = T
){
  
  dt_frame = dt_frame[,which(!(colnames(dt_frame) %in% exclude))]
  
  num_cols = setdiff(dt_frame %>% select_if(is.numeric)  %>% colnames() ,bypass)
  cat_cols = setdiff(colnames(dt_frame),c(num_cols,bypass))
  num_cols = num_cols[num_cols != y]
  
  num_enc_dt = dt_frame %>% select_at(num_cols)
  cat_enc_dt = dt_frame %>% select_at(cat_cols)
  
  
  if(is.null(num)){
    
    if(verbose==T){message("Numerical encoding bypassed")}
    
    num_encoder = function(input){NULL}
    
  }else if(num=="stand"){
    
    num_encoder = function(input){
      
      # ensures statistics are taken from the training dataset 
      stats = apply(num_enc_dt,2,function(x){list(m = mean(x,na.rm = T),s = sqrt(var(x,na.rm = T)))})
      
      toreturn = lapply(num_cols,function(x){data.frame((pull(input,x) - stats[[x]]$m)/stats[[x]]$s) %>% set_names(x)})
      
      return(data.frame(toreturn))
      
    }
    
  }else if(num=="norm"){
    
    num_encoder = function(input){
      
      # ensures statistics are taken from the training dataset 
      stats = apply(num_enc_dt,2,function(x){list(min = min(x,na.rm = T),max = max(x,na.rm = T))})
      toreturn = lapply(num_cols,function(x){data.frame((pull(input,x) - stats[[x]]$min)/(stats[[x]]$max - stats[[x]]$min)) %>% set_names(x)})
      
      return(data.frame(toreturn))
      
    }
    
  }else{
    stop("unrecognized numerical encoding")
  }
  
  
  if(is.null(cat)){
    
    if(verbose==T){message("Categorical encoding bypassed")}
    
    cat_encoder = function(input){NULL}
    
  }else if(cat=="ohe"){
    
    # we have to ensure  levels of categorical vars in test are a subset of  levels in train
    cat_encoder = function(input){
      
      unq_cat = lapply(cat_cols,function(x){sort(unique(cat_enc_dt[,x]))})
      names(unq_cat) = cat_cols
      
      lapply(cat_cols,function(x){
        
        grid = matrix(rep(unq_cat[[x]],length(input[,x])),
                      nrow = length(input[,x]),
                      byrow = T)
        
        dat = matrix(rep(input[,x],length(unq_cat[[x]])),
                     byrow = F,
                     ncol = length(unq_cat[[x]]),
                     dimnames = list(row_names = NULL,col_names = paste0(x,"_",unq_cat[[x]])))
        
        return(data.frame((dat == grid)*1))
        
      }) %>% 
        data.frame() %>% 
        return()
      
    }
    
  }else if(cat=="targ"){
    
    cat_encoder = function(input){
      
      # create lookup
      lkp = apply(cat_enc_dt,
                  2,
                  function(x){
                    
                    data.frame(original = x,
                               target = dt_frame[,y]) %>% 
                      group_by(original) %>% 
                      mutate(enc = mean(target)) %>% 
                      ungroup() %>% 
                      select(-target) %>% 
                      distinct() %>% 
                      arrange(original) %>% 
                      data.frame()
                    
                  })
      
      # replace entries from input according to lkp
      lapply(cat_cols,
             function(x){
               
               plyr::mapvalues(x = input %>% pull(x),
                               from = lkp[[x]]$original,
                               to = lkp[[x]]$enc) %>% 
                 as.numeric() %>% 
                 data.frame() %>% 
                 set_names(x)
               
             }) %>% 
        data.frame() %>% 
        return()
      
    }
    
  }else if(cat=="entem"){
    
  }else if(cat=="joint"){
    
    if(verbose==T){message("Joint embedding is trained...")}
    
    # OHE the categoricals - reference self with categoricals = cat_cols 
    give_ohe = preproc(dt_frame = dt_frame[,cat_cols],y = y,cat = "ohe",num = NULL,verbose = F)
    
    # just the target is going to be present
    ohe_dt = give_ohe(dt_frame)[,-1]
    
    # train the encoder on OHE categoricals
    no_neurons=16
    epoch=200
    batch_size=1000
    learning_rate=0.001
    
    #Network for the autoencoder
    Input = layer_input(shape = c(ncol(ohe_dt)))
    
    Output = Input %>% 
      layer_dense(units=no_neurons, activation='linear', use_bias=FALSE,name="encoder") %>% 
      layer_dense(units=ncol(ohe_dt), activation='softmax', use_bias=TRUE)
    
    model_ae=keras_model(inputs=Input,outputs=Output)
    
    #Optimize the cross entropy
    model_ae %>% 
      compile(optimizer=optimizer_nadam(lr=learning_rate),
              loss="categorical_crossentropy")  %>% 
      fit(ohe_dt,ohe_dt,
          epochs=epoch,
          batch_size=batch_size,
          verbose=0,
          validation_data=list(ohe_dt, ohe_dt),
          callbacks=list(callback_early_stopping(monitor="val_loss", 
                                                 min_delta=0,
                                                 patience=15, 
                                                 verbose=0, 
                                                 mode=c("min"),
                                                 restore_best_weights=TRUE)))
    
    #Recover the representation from the AE
    joint_embedding = keras_model(inputs=model_ae$input, outputs=get_layer(model_ae, "encoder")$output)
    
    cat_encoder = function(input){
      
      joint_embedding %>% 
        predict(give_ohe(input)) %>% 
        return()
      
    }
    
  }else{
    stop("unrecognized categorical encoding")
  }
  
  # combine encoders  
  encoder = function(x){
    
    x = x[,which(!(colnames(x) %in% exclude))]
    
    if(is.null(bypass)){
      bypassed = NULL
    }else if(length(bypass)==1){
      bypassed = x[bypass]
    }else{
      bypassed = x[,bypass]
    }
    
    if(!(y %in% colnames(x))){
      target = NULL
    }else{
      target = x[y]
    }
    
    temp_list = list(target,
                     bypassed,
                     num_encoder(input = x[,num_cols]),
                     cat_encoder(input = x[,cat_cols]))
    
    toreturn = do.call(cbind,
                       temp_list[!unlist(lapply(temp_list,is.null))])
    
    rownames(toreturn) = NULL
    
    return(data.matrix(toreturn))
    
  }
  
  return(encoder)
  
}


# Poisson Deviance 
custom_poisson <- function( y_true, y_pred ) {
  # Mario V. Wüthrich , Michael Merz
  # Statistical Foundations of Actuarial Learning and its Applications
  # Table 4.1 page 87
  
  # 2 * (y_pred - y_true - y_true*log(y_pred/y_true))
  # 2 (μ − y − ylog(μ/y))
  
  K <- backend()
  
  K$mean(2 * (y_pred - y_true - y_true * K$log((y_pred+10^-7)/(y_true+10^-7))))
  
}


poisson_deviance = function(y_true,y_pred,keras=F,correction = +10^-7){
  
  # stopifnot(length(true)!=length(pred),"different input lengths!")
  
  if(keras){
    # keras:
    # pd = y_pred - y_true * log(y_pred)
    
  }else{
    
    pd =  mean((y_pred - y_true - y_true * log((y_pred+correction)/(y_true+correction))))
    
  }
  
  return(2 * pd)
  
}


multiple_lift = function(y_true,
                          y_pred_df,
                          tiles = 10){
  
  tiles_list = list()
  
  for (i in colnames(y_pred_df)){
    
    tiles_list[[i]] = data.frame(model = y_pred_df[[i]],
                                 actual = y_true) %>% 
      mutate(tiles = ntile(model,tiles)) %>%
      group_by(tiles) %>% 
      summarise(model = mean(model)) %>% 
      pull(model)
  }
  
  bind_cols(tiles_list) %>% 
    mutate(t = 1:tiles) %>% 
    set_names(c(colnames(y_pred_df),"tiles")) %>% 
    pivot_longer(cols = !tiles) %>% 
    ggplot(aes(x = tiles,y=value,group=name,color=name,linetype=name))+
    geom_point()+
    geom_line()
  
}
