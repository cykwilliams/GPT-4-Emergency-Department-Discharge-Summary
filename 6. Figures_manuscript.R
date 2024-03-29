library(ggplot2)
df_fig2 = read.csv("Figure_2_manuscript.csv")
df_fig2$Domain = factor(df_fig2$Domain, levels = c('Accuracy', 'Hallucination', 'Clinical Omission'))

# Create a bar chart with ggplot2
fig2 = ggplot(df_fig2, aes(x = GPT.model, y = Total.cases, fill = Domain)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_manual(values = c("#009392", "#E9E29C", "#E88471")) +  # Set custom colors
  labs(title = "", x = "", y = "Proportion of cases (%)") +
  theme_minimal() +
  theme(axis.title.y = element_text(margin = margin(r = 20)))  # Adjust the margin as needed
ggsave("Figure_2.svg", fig2, width = 8, height = 6)

##############################

df_fig3 = read.csv("Figure_3_manuscript.csv")
df_fig3 = df_fig3[df_fig3$section_type != 'Total', ]
df_fig3 = df_fig3[df_fig3$section_type != '% Total',]
df_fig3_gpt35 = df_fig3[, c("section_type", "gpt35_crit1", "gpt35_crit2", "gpt35_crit3")]
df_fig3_gpt4 = df_fig3[, c("section_type", "gpt4_crit1", "gpt4_crit2", "gpt4_crit3")]

#Set same y-axis limits
y_axis_range = c(0, 35)

names(df_fig3_gpt35)[names(df_fig3_gpt35) == "gpt35_crit1"] <- "Accuracy"
names(df_fig3_gpt35)[names(df_fig3_gpt35) == "gpt35_crit2"] <- "Hallucination"
names(df_fig3_gpt35)[names(df_fig3_gpt35) == "gpt35_crit3"] <- "Clinical Omission"

names(df_fig3_gpt4)[names(df_fig3_gpt4) == "gpt4_crit1"] <- "Accuracy"
names(df_fig3_gpt4)[names(df_fig3_gpt4) == "gpt4_crit2"] <- "Hallucination"
names(df_fig3_gpt4)[names(df_fig3_gpt4) == "gpt4_crit3"] <- "Clinical Omission"

df_fig3_gpt35_reshaped = reshape2::melt(df_fig3_gpt35, id.vars = "section_type")
df_fig3_gpt35_reshaped$value <- as.numeric(as.character(df_fig3_gpt35_reshaped$value))
df_fig3_gpt35_reshaped$variable <- factor(df_fig3_gpt35_reshaped$variable, levels = c("Accuracy", "Hallucination", "Clinical Omission"))
df_fig3_gpt35_reshaped$section_type <- factor(df_fig3_gpt35_reshaped$section_type, levels = c('PC', 'HPC', 'PMH', 'Allergies', 'ROS', 'PE', 'Labs', 'Imaging', 'Plan', 'Other'))  # Replace with your actual category names

fig3A = # Create a faceted bar chart
  ggplot(df_fig3_gpt35_reshaped, aes(x = section_type, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.7) +
  facet_wrap(~variable, ncol = 1) +
  labs(title = "",
       x = "", #Remove x axis title ('Section of discharge summary') as will be putting side by side
       y = "Count") +
  scale_fill_manual(values = c("blue", "green", "red")) +
  coord_cartesian(ylim = y_axis_range)  + # Set the y-axis range
  theme(legend.position = "none") + # Remove the legend
  theme(axis.title.x = element_text(margin = margin(t = 10))) 
fig3A

ggsave("Figure_3A.svg", fig3A, width = 5, height = 8)

###

df_fig3_gpt4_reshaped = reshape2::melt(df_fig3_gpt4, id.vars = "section_type")
df_fig3_gpt4_reshaped$value <- as.numeric(as.character(df_fig3_gpt4_reshaped$value))
df_fig3_gpt4_reshaped$variable <- factor(df_fig3_gpt4_reshaped$variable, levels = c("Accuracy", "Hallucination", "Clinical Omission"))
df_fig3_gpt4_reshaped$section_type <- factor(df_fig3_gpt4_reshaped$section_type, levels = c('PC', 'HPC', 'PMH', 'Allergies', 'ROS', 'PE', 'Labs', 'Imaging', 'Plan', 'Other'))  # Replace with your actual category names

fig3B = # Create a faceted bar chart
  ggplot(df_fig3_gpt4_reshaped, aes(x = section_type, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.7) +
  facet_wrap(~variable, ncol = 1) +
  labs(title = "",
       x = "", #Remove x axis title ('Section of discharge summary') as will be putting side by side
       y = "") + #Remove y axis title as will be putting side by side
  scale_fill_manual(values = c("blue", "green", "red")) +
  coord_cartesian(ylim = y_axis_range)  + # Set the y-axis range
  theme(legend.position = "none") + # Remove the legend
  theme(axis.title.x = element_text(margin = margin(t = 10)))
fig3B

ggsave("Figure_3B.svg", fig3B, width = 5, height = 8)

