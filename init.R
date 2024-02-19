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
# https://paperswithcode.com/paper/model-agnostic-interpretable-and-data-driven/review/

# load data objects
dt_list = list()

dt_list$fre_mtpl2_freq = read.csv("pricing/data/freMTPL2freq.csv") %>%  
  # mutate(ClaimNb = pmin(ClaimNb/Exposure,15)) %>% 
  # mutate(ClaimNb = pmin(ClaimNb,3)) %>% 
  
  mutate(  Exposure = pmin(1, Exposure),
           ClaimNb = pmin(15, ClaimNb / Exposure)
           # VehPower = pmin(12, VehPower),
           # VehAge = pmin(20, VehAge),
           # VehGas = factor(VehGas),
           # DrivAge = pmin(85, DrivAge),
           # logDensity = log(Density)
           ) %>% 
  slice(sample(1:nrow(.),replace = F))

dt_list$fre_mtpl1_freq = read.csv("pricing/data/freMTPL/freMTPLfreq.csv") %>% 
  mutate(Brand = tolower(
    substr(Brand,
           start = 0,
           stop = replace_na(stringi::stri_locate_first(Brand,regex = " "),
                             length(Brand))[,2]-2)
  ),
  Region =  tolower(
    unlist(lapply(str_extract_all(Region, "[[:upper:]]"),
                  paste,
                  collapse="")))) %>% 
  rename(IDpol=PolicyID)

# dt_list$fre_mtpl1_sev = read.csv("pricing/data/freMTPL/freMTPLsev.csv")

dt_list$data_car_freq = read.csv("pricing/data/datacar.csv")[,-1] %>%
  mutate(ClaimNb = numclaims/exposure) %>%
  select(-X_OBSTAT_,-claimcst0,-clm,-numclaims) %>% 
  rename(Exposure=exposure)

dt_list$data_car_sev = read.csv("pricing/data/datacar.csv")[,-1] %>%
  mutate(Sev = claimcst0/exposure) %>%
  select(-X_OBSTAT_,-claimcst0,-clm,-numclaims) 

dt_list$allstate_sev = read.csv("pricing/data/Allstate/train.csv") %>% 
  mutate_at(.vars = paste0("cat",1:72),.funs = function(x){(x=="A")*1})

dt_list$workcomp = NULL

lapply(dt_list, summary)

# old
# poiss_loss = function(y_true,y_pred){
#   y_pred - y_true * log(y_pred,base = exp(1))
# }

poiss_loss = function(y_true,y_pred){
  y_true*log(y_true/y_pred)-(y_true-y_pred)
}

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
    # THINK IF IT SHOULDN"T HAVE NOISE IN TRAINING DATA OR HOLD OUT FOR VALIDATION / BACK TO THE PAPER
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


bayes_wrapper = function(k,
                         FUN = train_XGBoost,
                         PARS = makeParamSet(
                           makeNumericParam("eta",                    lower = 0.005, upper = 0.1),
                           makeIntegerParam("gamma",                  lower = 1,     upper = 5),
                           makeIntegerParam("max_depth",              lower= 2,      upper = 10),
                           makeIntegerParam("min_child_weight",       lower= 100,    upper = 2000),
                           makeNumericParam("subsample",              lower = 0.20,  upper = 1.0),
                           makeNumericParam("colsample_bytree",       lower = 0.20,  upper = 1.0)
                         ),
                         bayes_setup = list(rand_runs = 30,
                                            bayes_runs = 30,
                                            correction = 10^-7)){
  
  # sink("other/sink.txt")
  
  fn_type = substitute(FUN)
  
  param_helper = function(pars,fn = fn_type){
    
    if(fn == "train_XGBoost"){
      
      toreturn = as.list(as.numeric(pars))
      
      names(toreturn) = names(pars)
      
    }else{
      
        toreturn = list(lr = as.numeric(pars["lr"]),
          
          n_layers = as.integer(pars["n_layers"]),
          
          n_units = 2^as.integer(c(pars["units_1"],
                                   pars["units_2"],
                                   pars["units_3"],
                                   pars["units_4"],
                                   pars["units_5"]))[1:as.integer(pars["n_layers"])],
          
          n_activ = c(pars["activ_1"],
                      pars["activ_2"],
                      pars["activ_3"],
                      pars["activ_4"],
                      pars["activ_5"])[1:as.integer(pars["n_layers"])],
          
          n_dropout = c(pars["drop_1"],
                        pars["drop_2"],
                        pars["drop_3"],
                        pars["drop_4"],
                        pars["drop_5"])[1:as.integer(pars["n_layers"])]) 
          
    }
    
    return(toreturn)
    
  }
  
  
  obj.fun <- smoof::makeSingleObjectiveFunction(
    
    fn =   function(x){
      
      set.seed(42)
      
      iter = do.call(FUN,
                     append(list(dt = k$dt_train,
                                 y = k$dt_train_target + bayes_setup$correction,
                                 vdt = list(x_val = k$dt_test, 
                                            y_val = k$dt_test_target + bayes_setup$correction)),
                            param_helper(pars = x)))
      
      # PL = poiss_loss(y_true = k$dt_test_target + bayes_setup$correction,
      #                 y_pred = pmax(predict(iter,k$dt_test),mean(predict(iter,k$dt_test),na.rm=T)))
      
      PL = poiss_loss(y_true = k$dt_test_target + bayes_setup$correction,
                      y_pred = predict(iter,k$dt_test))
      
      if(is.na(mean(PL))|is.nan(mean(PL))){browser()}
      
      # if(run_number == 1 | mean(PL,na.rm=T)<mean(save_best_preds,na.rm=T)){
      if(run_number == 1 | mean(PL)<mean(save_best_preds)){
        
        save_best_preds <<- PL
        
      }
      
      run_number <<- run_number + 1
      
      setTxtProgressBar(pb,run_number)
      
      return(mean(PL))
      
    },
    par.set = PARS,
    minimize = TRUE 
  )
  
  des = generateDesign(n = bayes_setup$rand_runs,
                       par.set = getParamSet(obj.fun),
                       fun = lhs::randomLHS)
  
  control = makeMBOControl() %>%
    setMBOControlTermination(., iters = bayes_setup$bayes_runs)
  
  pb = txtProgressBar(min = 0, max = bayes_setup$bayes_runs+bayes_setup$rand_runs, initial = 0) 
  run_number = 1
  # sink(NULL)
  setTxtProgressBar(pb,run_number)
  # sink("other/sink.txt")
  
  save_best_preds = NA
  
  run = mlrMBO::mbo(fun = obj.fun,
                    design = des,
                    # learner = makeLearner("regr.km", predict.type = "se", covtype = "matern3_2", control = list(trace = FALSE)),
                    control = control, 
                    show.info = TRUE)
  
  close(pb)
  run_number = NULL
  
  path = run$opt.path$env$path %>% 
    mutate(n = row_number(),.before=everything(),
           path = c(rep("random",bayes_setup$rand_runs),rep("bayes",bayes_setup$bayes_runs))) %>% 
    arrange(y) 
  
  # browser()
  
  toreturn = list(path = path,
                  best_preds = data.frame(actual = k$dt_test_target+bayes_setup$correction,
                                          pred = save_best_preds))
  
  save_best_preds = NULL
  
  # sink(NULL)
  
  return(toreturn)
  
}

glm_wrapper = function(k,correction = 10^-7){
  
  model = glm(formula = k$dt_train_target ~ .,
              family = poisson(),
              data = data.frame(k$dt_train))
  
  PL = poiss_loss(y_true = k$dt_test_target+correction,
                  y_pred = predict(model,data.frame(k$dt_test),type="response"))
  
  bp = data.frame(actual = k$dt_test_target+correction,
                  pred = PL)
  
  return(list(best_preds = bp,
              model = model))
  
}


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
    
    # pd = rep(NA,length(true))
    # pd[true==0] = pred[true==0]
    # pd[true!=0] = true[true!=0] * log(true[true!=0]/pred[true!=0]) + pred[true!=0] - true[true!=0]
    
    pd =  mean((y_pred - y_true - y_true * log((y_pred+correction)/(y_true+correction))))
    
  }
  
  return(2 * pd)
  
}

# single lift - vectors as input
single_lift = function(y_true,
                       y_pred,
                       tiles = 5,
                       display_names = c(y_true = "y_true",
                                         y_pred = "y_pred"),
                       type = "col") #or "line"
  {
  
  data.frame(true = y_true,
             pred = y_pred) %>% 
    mutate(tiles = ntile(pred,tiles)) %>% 
    group_by(tiles) %>% 
    summarise(true = mean(true),
              pred = mean(pred)) %>% 
    pivot_longer(cols = !tiles) %>% 
    mutate(name = case_when(name=="true" ~ display_names["y_true"],
                            name=="pred" ~ display_names["y_pred"])) %>% 
    ggplot(aes(x = tiles,y=value,fill=name,color=name,type=name))+
    {if(type=="col") geom_col(position = "dodge") else geom_point()}+
    {if(type=="line") geom_line(size=1) }+
    ggtitle(paste0(display_names["y_pred"]," single lift"))
  
}

double_lift=function(y_true,
                     y_pred_1,
                     y_pred_2,
                     display_names = c(y_true = "y_true",
                                   y_pred_1 = "y_pred_1",
                                   y_pred_2 = "y_pred_2"),
                     tiles = 5){
  
  data.frame(true = y_true,
             pred1 = y_pred_1,
             pred2 = y_pred_2) %>% 
    mutate(tiles = ntile(pred1/pred2,tiles)) %>% 
    group_by(tiles) %>% 
    summarise(true = mean(true),
              pred1 = mean(pred1),
              pred2 = mean(pred2)) %>% 
    pivot_longer(cols = !tiles) %>% 
    mutate(name = case_when(name=="true" ~ display_names["y_true"],
                            name=="pred1" ~ display_names["y_pred_1"],
                            name=="pred2" ~ display_names["y_pred_2"])) %>% 
    ggplot(aes(x = tiles,y=value,fill=name))+
    geom_col(position = "dodge")+
    ggtitle(paste0(display_names["y_pred_1"]," vs ",display_names["y_pred_2"]))
  
}


multiple_lift = function(y_true,
                         glm,
                         y_pred_df,
                         tiles = 10){
  
  cbind(y_true = y_true,pred_glm = glm, y_pred_df) %>%
    mutate(glm_tiles = ntile(pred_glm,tiles)) %>%
    # pivot_longer(cols = c(colnames(y_pred_df),pred_glm)) %>%
    group_by(glm_tiles) %>%
    # summarise_all(vars(c(colnames(y_pred_df),pred_glm)),.funs=mean) %>%
    summarise(avg = mean(y_true),
              glm = mean(pred_glm),
              nn = mean(ff_nn),
              localGLMnet = mean(localGLMnet),
              CANN = mean(CANN)) %>%
    ungroup() %>% 
    pivot_longer(cols = !glm_tiles) %>% 
    ggplot(aes(x = glm_tiles,y=value,group=name,color=name,linetype=name))+
    geom_point()+
    geom_line()
    
}


multiple_lift2 = function(y_true,
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

