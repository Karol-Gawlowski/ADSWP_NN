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
    # ggdark::dark_theme_classic()+
    # theme(panel.grid.minor = element_line(colour="darkgrey", size=0.01,linetype = 3))+
    theme(axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=0.5))+ 
    scale_y_continuous(
      sec.axis = sec_axis(~ . / prop, name = "Exposure",labels = scales::comma)
    )+
    ggtitle(paste0("One way - ",xvar))+
    ylab("Freq")+
    xlab(xvar) %>% 
    return()
  
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


generate_lorenz_curve <- function(predictions, actuals) {
  # Check if the lengths of predictions and actuals are equal
  if (length(predictions) != length(actuals)) {
    stop("The lengths of predictions and actuals must be equal")
  }
  
  # Create a data frame with predictions and actuals
  data <- data.frame(predictions = predictions, actuals = actuals)
  
  # Sort the data frame by predictions
  data <- data[order(data$predictions), ]
  
  # Calculate cumulative sums for predictions and actuals
  data$cum_pred <- cumsum(data$predictions) / sum(data$predictions)
  data$cum_actual <- cumsum(data$actuals) / sum(data$actuals)
  
  # Add a point for (0,0) to the curve
  data <- rbind(c(0, 0, 0, 0), data)
  
  # Plot the Lorenz curve
  plot(data$cum_pred, data$cum_actual, type = "l", col = "blue",
       xlab = "Cumulative Proportion of Predictions",
       ylab = "Cumulative Proportion of Actuals",
       main = "Lorenz Curve for Model Predictions")
  
  # Add the line of equality (45-degree line)
  abline(0, 1, col = "red", lty = 2)
  
  return(data)
}
