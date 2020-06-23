library(shiny)
library(shinydashboard)
library(tidyverse)
library(plotly)
library(reshape2)
library(png)
library(grid)
library(magick)
library(extrafont)
library(jsonlite)
library("shiny")
library("shinythemes")
library(gridExtra)
library(grid)

##library(rsconnect)
##rsconnect::deployApp('path/to/your/app')

total_puzzles <- 30
datosFunnelUser <- read.csv("funnelOutput.csv", header = TRUE)
datosFunnelUser<-datosFunnelUser%>% filter(task_id != "SAND")

dfActivity <- read.csv("activityOutput.csv", header = TRUE)
dfActivity<-dfActivity%>% filter(task_id != "SAND")
datosDiff <- read.csv("difficultyOutput.csv", header = TRUE)
datosCompetencyELO<- read.csv("datosCompetencyELO_normalized.csv", header = TRUE)
datosDifficultyELO<- read.csv("datosDifficultyELO_normalized.csv", header = TRUE)
dfCompetencyELO<-NULL

#Controls student key to change plot
k <- NULL
eventCompetency<-0
eventL<- 0
dashboardSid<-dashboardSidebar(
  sidebarMenu(id = "menu",
    menuItem("Levels of Activity", tabName = "activity", icon = icon("chart-line")),
    menuItem("Levels of Difficulty", tabName = "difficulty", icon = icon("book")),
    menuItem("ELO-based learner", tabName = "dashboardELO", icon = icon("caret-square-down"),
             menuSubItem("User competency", tabName = "Subdashboardcompetency", icon = icon("user-graduate")),
             menuSubItem("Puzzle Difficulty", tabName = "SubdashboardELOdifficulty", icon = icon("pencil-alt")))
  )
) 


body <- dashboardBody(
  tabItems(
    tabItem(tabName = "Subdashboardcompetency",
            fluidRow(
              column (box(
                selectizeInput(inputId = "groupCompetency", label = "Choose a group",
                               choices = unique(datosCompetencyELO$group), options = list(
                                 placeholder = 'Please select a group',
                                 onInitialize = I('function() { this.setValue(""); }')
                               ))
              ), width = 12, offset = 3), column (box(plotlyOutput("competencyELO", height = 600, width = 1000), width = 12, height = 650), width = 12)
            )
    ),
    
    tabItem(tabName = "SubdashboardELOdifficulty",
            fluidRow(
              column (box(
                selectizeInput(inputId = "groupDifficulty", label = "Choose a group",
                               choices = unique(datosDifficultyELO$group), options = list(
                                 placeholder = 'Please select a group',
                                 onInitialize = I('function() { this.setValue(""); }')
                               ))
              ), width = 12, offset = 3), column (box(plotlyOutput("difficultyELO", height = 600, width = 1000), width = 12, height = 650), width = 12)
            )
    ),

    tabItem(tabName = "activity",
            fluidRow(column (box(selectizeInput(inputId = "groupLevels", label = "Choose a group",
                                                choices = unique(datosFunnelUser$group), options = list(
                                                  placeholder = 'Please select a group',
                                                  onInitialize = I('function() { this.setValue(""); }')
                                                ))), width = 12, box(
                                                  selectizeInput(inputId = "userLevel", label = "Choose a user",
                                                                 choices = unique(unique(dfActivity$user)), options = list(
                                                                   placeholder = 'Please select a user',
                                                                   onInitialize = I('function() { this.setValue(""); }')
                                                                 ))
                                                )),  column (box(plotlyOutput("levelsOfActivity", height = 600, width = 1000), width = 12, height = 650), width = 12)
            )
    ),

    tabItem(tabName = "difficulty",
            fluidRow(
              column (box(
                selectizeInput(inputId = "groupLevelsOfDifficulty", label = "Choose a group",
                               choices = unique(datosDiff$group), options = list(
                                 placeholder = 'Please select a group',
                                 onInitialize = I('function() { this.setValue(""); }')
                               ))
              ), width = 12, offset = 3), column (box(plotlyOutput("levelsOfDifficulty", height = 600, width = 1000), width = 12, height = 650), width = 12)
            )
    )
  )
)




ui <- dashboardPage(
  #dashboardHeader(title = "Shadowspect Dashboard", titleWidth = 500, tags$li(actionLink("info", label = "", icon = icon("info")),class = "dropdown")),
  dashboardHeader(title = "Shadowspect Dashboard", titleWidth = 500, dropdownMenu(
    type = "notifications", 
    icon = icon("info"),
    badgeStatus = NULL,
    headerText = "See also:",
    
    tags$li(actionLink("info", label = "About Us", icon = icon("lightbulb"))),
    tags$li(actionLink("help", label = "Help", icon = icon("question-circle")))
  )),
  
  skin = "black",
  dashboardSid,
  body
  
)



server <- function(session,input, output) {
  
  ############################
  
  
  observeEvent(input$info, {
    showModal(
      modalDialog(title = "About Us",
                  HTML("This work is part of the Bachelor thesis on learning analysis in educational games. This dashboard shows some of the metrics developed for the analysis of the data generated with the educational game Shadowspect.<br> 
Pedro Antonio Martínez Sánchez is a computer engineering student at the University of Murcia and presents this work as part of his Bachelor thesis.<br>",
"Jose A. Ruipérez is the tutor in this work "))
    )
  })
  
  observeEvent(input$help, {
    showModal(
      modalDialog(title = "Help",
                  p("For faster and more intuitive use, a system of links has been developed to access between metrics. By clicking on the user within the competition we can access the Levels of Activity metric. Clicking on the puzzle of Levels of Activity we access the difficulty of the puzzles"))
    )
  })
  
  observe({

    if (nchar(input$groupLevels) < 1) {k <<- NULL}
    else {
      eL <<- event_data(
        
        event = "plotly_click",
        source = "L",
        session = session
      )

      new_valueL <- ifelse(is.null(eL),"0",(eL$pointNumber+1)) # 0 if no selection
      if(eventL!=new_valueL) {
        eventL <<- new_valueL 
        if(eventL !=0) {
         updateTabItems(session, "menu",
                         selected = "SubdashboardELOdifficulty")
          groupL=input$groupLevels
          updateSelectizeInput(session, "groupDifficulty", selected= groupL)
          
          
        }
      }else{}
      
    }
  })

  observe({
    
    if (nchar(input$groupCompetency) < 1) {k <<- NULL}
    else {
      eC <<- event_data(
        
        event = "plotly_click",
        source = "C",
        session = session
      )
      sortUs <- sort(unique(dfCompetencyELO$user))[(eC$pointNumber + 1)]
      dfComp <- dfCompetencyELO[dfCompetencyELO$user==sortUs,]
      keyC <- unique(dfComp$userN)
      
      new_valueComp <- ifelse(is.null(eC),"0",(eC$pointNumber+1)) # 0 if no selection
      if(eventCompetency!=new_valueComp) {
        eventCompetency <<- new_valueComp 
        if(eventCompetency !=0) {
          updateTabItems(session, "menu",
                         selected = "activity")
          groupU = unique(datosCompetencyELO[datosCompetencyELO$user==keyC,]$group)
          updateSelectizeInput(session, "groupLevels", selected= groupU)
          selectComp=datosCompetencyELO %>% filter(group == groupU) %>% select(user)
          k <<- keyC
          updateSelectInput(session, "userLevel",choices = selectComp, selected = keyC)

        }
      }else{}
      
    }
  })

  observe({
   
  })
  
  
  output$competencyELO <- renderPlotly({
    
    if (nchar(input$groupCompetency) < 1) {}
    else {
      dfCompetencyELO <- datosCompetencyELO%>% filter(group == input$groupCompetency)%>% mutate(competency = round(competency,2)) 
      dfCompetencyELO <- dfCompetencyELO %>% mutate(userN = user)
      dfCompetencyELO <<- dfCompetencyELO %>% melt(id.vars = c("user","userN"))
      #dfCompetencyELO <- dfCompetencyELO %>% rename(kc = Knowledge_Components)
      names(dfCompetencyELO)[which(names(dfCompetencyELO) == "kc")] <- "KC"
      actiCompetency <-ggplot(dfCompetencyELO, aes(x=user, y = competency, fill = KC)) +  geom_bar(stat='identity', position='stack') +  theme_minimal() +  theme(axis.text.x = element_text(angle = 90, size = 7),plot.title = element_text(hjust = 0.5)) +
        theme(legend.position = 'bottom') + labs(title ="Competency ELO", x = "", y = "Competency")
      ggplotly(actiCompetency, tooltip = c("x", "y"), source="C")
    }
  })

  output$levelsOfDifficulty <- renderPlotly ({
    if (nchar(input$groupLevelsOfDifficulty) < 1) {}
    else {
      datosDiff <- datosDiff %>% filter(group==input$groupLevelsOfDifficulty)
      datos_melted <- datosDiff %>% melt(id.vars = c("task_id"), measure.vars = c("completed_time", "actions_completed", "p_incorrect", "p_abandoned", "norm_all_measures"), value.name = "value")
      #datos_melted$variable <- factor(datos_melted$variable, levels = c("completed_time", "actions_completed", "p_incorrect", "p_abandoned", "norm_all_measures"))
      subsetTime<-datos_melted%>% filter(variable == "completed_time")
      subsetActions<-datos_melted%>% filter(variable == "actions_completed")
      subsetIncorrect<-datos_melted%>% filter(variable == "p_incorrect")
      subsetAbandoned<-datos_melted%>% filter(variable == "p_abandoned")
      subsetGeneral<-datos_melted%>% filter(variable == "norm_all_measures")
  
      subsetTime$title = "Active Time"
      subsetActions$title2 = "Number of Actions"
      subsetIncorrect$title3 = "Percentage Incorrect"
      subsetAbandoned$title4= "Percentage Abandoned"
      subsetGeneral$title5="General Difficulty Measure"
      
      
      plotTime <-ggplot(subsetTime, aes(x=task_id, y = value)) +  geom_bar(stat='identity', position='stack') +  theme_minimal() +
        theme(axis.text.x = element_blank(),plot.title = element_text(hjust = 0.5)) + facet_wrap(~title)+
        theme(legend.position = 'bottom') + labs(title ="", x = "", y = "Time (s)")  
      

      plotAction <-ggplot(subsetActions, aes(x=task_id, y = value)) +  geom_bar(stat='identity', position='stack') +  theme_minimal() +
        theme(axis.text.x = element_blank(),plot.title = element_text(hjust = 0.5)) + facet_wrap(~title2)+
        theme(legend.position = 'bottom') + labs(title ="", x = "Action", y = "Action Measure")  
      
      plotIncorrect <-ggplot(subsetIncorrect, aes(x=task_id, y = value)) +  geom_bar(stat='identity', position='stack') +  theme_minimal() +
        theme(axis.text.x = element_blank(),plot.title = element_text(hjust = 0.5)) +facet_wrap(~title3)+
        theme(legend.position = 'bottom') + labs(title ="", x = "", y = "Incorrect Percentage")  
      
      plotAbandoned <-ggplot(subsetAbandoned, aes(x=task_id, y = value)) +  geom_bar(stat='identity', position='stack') +  theme_minimal() +
        theme(axis.text.x = element_blank(),plot.title = element_text(hjust = 0.5)) +facet_wrap(~title4)+
        theme(legend.position = 'bottom') + labs(title ="", x = "", y = "Abandoned Percentage") 
      
      plotGeneral <-ggplot(subsetGeneral, aes(x=task_id, y = value)) +  geom_bar(stat='identity', position='stack') +  theme_minimal() +
        theme(axis.text.x = element_text(angle = 90, size = 7),plot.title = element_text(hjust = 0.5)) +facet_wrap(~title5)+
        theme(legend.position = 'bottom') + labs(title ="", x = "", y = "Difficulty")  
      
      #fig <- plot_ly(subsetTime, x = ~task_id, y = ~value, type = 'bar', name = 'SF Zoo')
      #plotTimeLy<-ggplotly(plotTime)
      #plotActionLy<-ggplotly(plotAction)
      
      p1<-subplot(plotTime, plotAction,titleY = TRUE)
      p2<-subplot(plotIncorrect,plotAbandoned,titleY = TRUE)
      p3<-subplot(p1,p2,plotGeneral, nrows = 3,titleY = TRUE)
      
      
      #p_all <- subplot(plotTime, plotAction,plotIncorrect,plotAbandoned,plotGeneral, nrows=3, shareX = TRUE, shareY = TRUE, titleX = TRUE, titleY = TRUE)#%>%layout(title="Prueba")
      #p_all %>% layout(annotations = list(list(x = 1 , y = 1.05, text = "AA", showarrow = F, xref='paper', yref='paper')))
      ggplotly(p3) 
      
    }
  })
  
  output$difficultyELO <- renderPlotly({
    if (nchar(input$groupDifficulty) < 1) {}
    else {
      datosDifficultyELO <- datosDifficultyELO %>% filter(group == input$groupDifficulty)%>% mutate(difficulty = round(difficulty,2)) 
      actiDifficulty <-ggplot(datosDifficultyELO, aes(x=task_id, y = difficulty)) +  geom_bar(stat='identity', position='stack') +  theme_minimal() +  theme(axis.text.x = element_text(angle = 90, size = 7),plot.title = element_text(hjust = 0.5)) +
        theme(legend.position = 'bottom') + labs(title ="Difficulty ELO", x = "", y = "Difficulty")
      ggplotly(actiDifficulty, tooltip = c("x", "y"))
    }
  })  
  
  output$levelsOfActivity <- renderPlotly({
    if (nchar(input$userLevel) < 1) {}
    else {
      dfActivity <- dfActivity %>% mutate(value = round(value, 2)) %>% filter(user == input$userLevel) %>% select(-X, -group, -user) %>% arrange(task_id, metric) %>% filter(!(metric %in% c("move_shape", "undo_action", "redo_action", "paint", "scale_shape", "delete_shape", "rotate_view", "create_shape", "snapshot")))
      subsetEvent<-dfActivity%>% filter(metric == "event")
      subsetNevent<-dfActivity%>% filter(metric == "different_events")
      subsetActive<-dfActivity%>% filter(metric == "active_time")
      
      subsetEvent$title = "Number of Events"
      subsetNevent$title2 = "Number of Different Events"
      subsetActive$title3 = "Active Time (s)"
      
      plotEvent <-ggplot(subsetEvent, aes(x=task_id, y = value)) +  geom_bar(stat='identity', position='stack') +  theme_minimal() +  theme(axis.text.x = element_text(angle = 90, size = 7),plot.title = element_text(hjust = 0.5)) +
        theme(legend.position = 'bottom') + labs(title ="", x = "", y = "") + facet_wrap( ~ title) 
      plotDifEvent <-ggplot(subsetNevent, aes(x=task_id, y = value)) +  geom_bar(stat='identity', position='stack') +  theme_minimal() +  theme(axis.text.x = element_text(angle = 90, size = 7),plot.title = element_text(hjust = 0.5)) +
        theme(legend.position = 'bottom') + labs(title ="", x = "", y = "")+ facet_wrap( ~ title2)
      plotActive <-ggplot(subsetActive, aes(x=task_id, y = value)) +  geom_bar(stat='identity', position='stack') +  theme_minimal() +  theme(axis.text.x = element_text(angle = 90, size = 7),plot.title = element_text(hjust = 0.5)) +
        theme(legend.position = 'bottom') + labs(title ="", x = "", y = "")+ facet_wrap( ~ title3)
      
      p3<-subplot(plotActive,plotDifEvent,plotEvent, nrows = 1,shareX = TRUE,titleY = TRUE)
      p3$x$source<-"L"
      ggplotly(p3, tooltip = c("x", "y"))
    }
  })

}

shinyApp(ui, server)


