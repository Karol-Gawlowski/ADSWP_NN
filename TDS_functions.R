one_way_chart = function(dt = analysis_wide, #wide format
                         models = c("glm","ff_nn"), # string vector
                         y = "actual", # actual string
                         xvar = "DrivAge", # string
                         expo = "Exposure",# string
                         buckets = 10 # int for numerical
){
  
  if(is.numeric(dt[[xvar]])){
    
    tot_ex = sum(dt[[expo]])
    
    dt = dt %>% 
      arrange(!!sym(xvar)) %>% 
      mutate(!!sym(xvar) := cut(cumsum(!!sym(expo)),breaks = seq(0,tot_ex,length.out = buckets)))
    
  }else if(!is.character(dt[[xvar]])){
    stop("x format wrong")
  }
  
  m = dt[c(xvar,y,models)] %>% 
    group_by(!!sym(xvar)) %>% 
    summarise_all(.funs = mean)
  
  e = dt[c(xvar,expo)] %>% 
    group_by(!!sym(xvar)) %>% 
    summarise(ex = sum(!!sym(expo)))
  
  prop = max(m[models])/max(e$ex)
  
  # IF IS STRING THEN ORDER BY EXPO SIZE
  
  # add secondary axis
  # add tilt if numeric
  
  ggplot()+
    geom_point(data = m %>% 
                 pivot_longer(cols = c(models,y),names_to = "model"),
               aes_string(x = xvar,y = "value",color = "model",shape = "model"),
               alpha = 0.8,
               size = 2)+
    geom_col(data = e %>% 
               mutate(ex = ex*prop*0.5),mapping = aes_string(x = xvar,y="ex"),
             alpha = 0.3) +
    ggdark::dark_theme_classic()+
    # theme(panel.grid.minor = element_line(colour="darkgrey", size=0.01,linetype = 3))+
    ggtitle("title")+
    xlab(xvar) %>% 
    return()
  
}