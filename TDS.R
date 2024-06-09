source("init.R")
source("init_py.R")
source("models/feed_forward_neural_net.R")
source("models/localGLMnet.R")
source("models/CANN.R")
source("TDS_functions.R")

# fitted = readRDS("final_results") #identify the proper RDS

# OOF results per model point wide format
analysis_wide = bind_rows(results,.id = "id")  %>% 
  base::merge(dt_list$fre_mtpl2_freq,
        by.x = "ID",
        by.y = "IDpol")

# add lorenz 

# more tables on model performance

# where does the glm underperform the most

# classic lift?

one_way_chart(xvar = "VehBrand")

one_way_chart(dt = analysis_wide,
              models = c("glm","ff_nn","localGLMnet","CANN","homog"),
              buckets = 20)

one_way_chart(dt = analysis_wide,
              models = c("glm","localGLMnet","CANN"),
              xvar = "DrivAge", # string
              buckets = 10)





generate_lorenz_curve(predictions = analysis_wide$localGLMnet,
                      actuals = analysis_wide$actual)




# Final Charts:


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




# for DrivAge
one_way_chart(dt = analysis_wide %>% 
                # filter(DrivAge>24,DrivAge<68) %>% 
                filter(DrivAge<80) %>% 
                mutate(DrivAge = as.character(DrivAge)),
              models = c("glm","localGLMnet"),
              xvar = "DrivAge", # string
              buckets = 10)

one_way_chart(dt = analysis_wide %>% 
                # filter(DrivAge>24,DrivAge<68) %>% 
                filter(DrivAge<80) %>% 
                mutate(DrivAge = as.character(DrivAge)),
              models = c("glm","CANN"),
              xvar = "DrivAge", # string
              buckets = 10)

one_way_chart(dt = analysis_wide %>% 
                # filter(DrivAge>24,DrivAge<68) %>% 
                filter(DrivAge<80) %>% 
                mutate(DrivAge = as.character(DrivAge)),
              models = c("glm","ff_nn"),
              xvar = "DrivAge", # string
              buckets = 10)

# for 
one_way_chart(dt = analysis_wide,
              models = c("glm","CANN"),
              xvar = "VehBrand", # string
              buckets = 10)








