# fitted = readRDS("final_results") #identify the proper RDS

# OOF results per model point
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

one_way_chart(dt = analysis_wide %>% filter(DrivAge<25),
              models = c("glm","localGLMnet"),
              xvar = "VehPower", # string
              buckets = 10)








