source("init.R")
source("init_py.R")
source("models/feed_forward_neural_net.R")
source("models/localGLMnet.R")
source("models/CANN.R")

CV = 5 #K Folds

CV_vec = sample(1:CV,replace = T,size = nrow(dt_list$fre_mtpl2_freq))

#Lists
models = list()
results = list()
losses = data.frame(CV = paste0("CV_",1:CV),
                    glm = NA,
                    ff_nn = NA,
                    localGLMnet = NA,
                    CANN = NA,
                    XGB = NA)

# to retrive values from the article:
# fitted = readRDS("final_results")

# losses = fitted$losses
# results = fitted$results

#Cross Validation
for (i in 1:CV){
  
  train_rows = which(CV_vec != i)
  
  models[[paste0("CV_",i)]] = list()
  
  #DB creation
  results[[paste0("CV_",i)]] = data.frame(ID = dt_list$fre_mtpl2_freq$IDpol[-train_rows],
                                          actual = dt_list$fre_mtpl2_freq$ClaimNb[-train_rows],
                                          glm = NA,
                                          ff_nn = NA,
                                          localGLMnet = NA,
                                          CANN = NA) %>% 
    mutate(homog = mean(dt_list$fre_mtpl2_freq$ClaimNb[train_rows]))
  
  encoder = preproc(dt_frame = dt_list$fre_mtpl2_freq[train_rows,],
                    y = "ClaimNb",
                    num = "norm",
                    cat = "ohe",
                    bypass = NULL,
                    exclude = c("IDpol","Exposure"),
                    verbose = T)
  
  #Train - Test split
  train = encoder(dt_list$fre_mtpl2_freq[train_rows,])
  test = encoder(dt_list$fre_mtpl2_freq[-train_rows,])
  
  # homogenous model ------------------------------------------------- 
  
  losses$homog[i] =poisson_deviance(y_true = results[[paste0("CV_",i)]]$actual,
                                    y_pred = results[[paste0("CV_",i)]]$homog)
  
  # GLM ------------------------------------------------- 
  
  models[[paste0("CV_",i)]]$glm_model = glm(formula = ClaimNb~.,
                                            family = poisson,
                                            data = dt_list$fre_mtpl2_freq[train_rows,-c(1,3)])
  
  results[[paste0("CV_",i)]]$glm = as.vector(predict(models[[paste0("CV_",i)]]$glm_model,
                                                     dt_list$fre_mtpl2_freq[-train_rows,-c(1,3)],type="response"))
  
  losses$glm[i] =poisson_deviance(y_true = results[[paste0("CV_",i)]]$actual,
                                  y_pred = results[[paste0("CV_",i)]]$glm)
  
  # ff_nn - Feed Forward Neural Network ------------------------------------------------
  
  models[[paste0("CV_",i)]]$ff_nn_model = train_ff_nn(dt = train[,-1],
                                                      y = train[,1],
                                                      vdt = list(x_val = test[,-1],
                                                                 y_val = test[,1]),
                                                      n_dropout = c(F,F,F),
                                                      n_layers = 3,
                                                      n_units = c(20,15,15),
                                                      n_activ = c("sigmoid","sigmoid","sigmoid"),
                                                      # n_activ = c("tanh","tanh","tanh"),
                                                      lr = 0.005,
                                                      bs = 2^12,
                                                      ep = 1000)
  
  results[[paste0("CV_",i)]]$ff_nn = predict(models[[paste0("CV_",i)]]$ff_nn_model,test[,-1])[,1]
  
  losses$ff_nn[i] = poisson_deviance(y_true = results[[paste0("CV_",i)]]$actual,
                                     y_pred = results[[paste0("CV_",i)]]$ff_nn)
  
  # LocalGLMnet -------------------------------------------
  
  models[[paste0("CV_",i)]]$localGLMnet_model = train_LocalGLMnet(dt = train[,-1],
                                                                  y = train[,1],
                                                                  vdt = list(x_val = test[,-1],
                                                                             y_val = test[,1]),
                                                                  n_dropout = c(F,F,F),
                                                                  n_layers = 3,
                                                                  n_units = c(20,15,15),
                                                                  n_activ = c("tanh","tanh","tanh"),
                                                                  lr = 0.005,
                                                                  bs = 2^12,
                                                                  ep = 1000)
  
  results[[paste0("CV_",i)]]$localGLMnet = predict(models[[paste0("CV_",i)]]$localGLMnet_model,test[,-1])[,1]
  
  losses$localGLMnet[i] = poisson_deviance(y_true = results[[paste0("CV_",i)]]$actual,
                                           y_pred = results[[paste0("CV_",i)]]$localGLMnet)
  
  # CANN  - Combined Actuarial Neural Network ------------------------------------------
  
  #weights - used as initialization
  learn_GLM <- fitted(models[[paste0("CV_",i)]]$glm_model)
  test_GLM <- results[[paste0("CV_",i)]]$glm
  
  w_learn=as.matrix(log(learn_GLM))
  w_test=as.matrix(log(test_GLM))
  
  
  models[[paste0("CV_",i)]]$CANN_model = train_CANN(dt = train[,-1],
                                                    y = train[,1],
                                                    vdt = list(x_val =list( test[,-1], w_test),
                                                               y_val = test[,1]),
                                                    w=w_learn,
                                                    # n_dropout = c(F,F,F),
                                                    n_layers = 3,
                                                    n_units = c(20,15,15),
                                                    n_activ = c("tanh","tanh","tanh"),
                                                    lr = 0.005,
                                                    bs = 2^12,
                                                    ep = 1000)
  
  results[[paste0("CV_",i)]]$CANN = predict(models[[paste0("CV_",i)]]$CANN_model,list(as.matrix(test[,-1]),w_test))[,1]
  
  losses$CANN[i] = poisson_deviance(y_true = results[[paste0("CV_",i)]]$actual,
                                    y_pred = results[[paste0("CV_",i)]]$CANN)
  
}

# save files
# saveRDS(list(losses = losses,
#              results = results,
#              models = models),file = "results_v2.rds")
# 
# saveRDS(list(losses = losses,
#              results = results),file = "results_wo_models_v2.rds")

analysis = bind_rows(results,.id = "id")  %>% 
  select(id,actual,glm,ff_nn,localGLMnet,CANN,homog) %>% 
  pivot_longer(cols = glm:homog) %>% 
  mutate(actual = actual,
         value = value,
         poiss = Vectorize(poisson_deviance)(y_true = actual,
                                             y_pred = value)) 

# ovarall and per fold results
rbind(losses,
      losses %>%
        pivot_longer(cols = !CV) %>%
        group_by(name) %>%
        summarise(mean_poiss = mean(value)) %>%
        arrange(mean_poiss) %>%
        pivot_wider(values_from = mean_poiss,names_from = name) %>%
        mutate(CV = "mean_poiss"))

losses %>% 
  mutate_if(is.numeric,.funs = function(x){
    x*c(data.frame(k=CV_vec) %>% 
          count(k) %>% pull(n))}) %>% 
  janitor::adorn_totals()

analysis %>%
  filter(name!="homog") %>% 
  rename(model=name) %>% 
  ggplot(aes(x = poiss,fill=model,color=model,linetype=model))+
  geom_density(alpha=0.3,size=1)+
  ggplot2::scale_fill_manual(values = c("blue","yellow","green","grey"))+
  xlim(0,0.75)+
  # facet_wrap(~name)+
  ggdark::dark_theme_classic()+
  theme(panel.grid.minor = element_line(colour="darkgrey", size=0.01,linetype = 3))+
  ggtitle("Poisson deviance per observation, per model")+
  xlab("Poisson deviance")

# lift chart
multiple_lift(y_true = bind_rows(results,.id = "id") %>% pull(actual),
               y_pred_df = bind_rows(results,.id = "id") %>% select(glm,
                                                                    ff_nn,
                                                                    localGLMnet,
                                                                    CANN,
                                                                    # XGB,
                                                                    homog))+
  ggtitle("Combined lift chart")+
  xlab("Tiles")+
  ylab("Implied frequency")+
  ggdark::dark_theme_classic()

