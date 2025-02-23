# OPAN 6603 - Project 3 ####

# SAXA 3 #
# Mike Johnson | Kesh Kamani | Ryan Mathis | Khushi Patel | Andrew Singh #

## Set Up ####

# Load Libraries
library(tidyverse) # For some data manipulation, ggplot, and more
library(GGally)
library(factoextra) # For PCA helper functions
library(corrplot) #For some correlation and PCA plots
library(plotly) # for 3d interactive visualization
library(caret) # for predictions
library(MLmetrics) # for multiclass summaries
library(cluster) # standardizing variables
library(readxl) # Source data is in .xlsx file

# Set random seed for reproducibility
set.seed(206)

# Set viz theme
theme_set(theme_classic())

# Read Data
reviews = read_excel('data-raw/Travel_Review.xlsx')

## Step 1: Create a train/test split ####
# Unsupervised Learning exercise. Not Needed.

## Step 2: Data Exploration ####

### Summary Statistics ####
# There's one NA in Gardens.
summary_stats = 
  reviews %>%
  select(-UserID) %>% 
  summarise_all(list(
    min = ~ round(min(., na.rm = TRUE),2),
    max = ~ round(max(., na.rm = TRUE),2),
    mean = ~ round(mean(., na.rm = TRUE),2),
    median = ~ round(median(., na.rm = TRUE),2),
    sd = ~ round(sd(., na.rm = TRUE),2),
    n = ~ n(),
    na = ~ sum(is.na(.))
  )) %>% 
  pivot_longer(
    cols = everything(),
    names_to = 'Variable'
  ) %>% 
  separate(
    Variable,
    into = c('Variable', 'Statistic'),
    sep = "_(?=[^_]+$)"
  ) %>% 
  pivot_wider(
    names_from = Statistic,
    values_from = value
  )

summary_stats

### Distribution Analysis ####
# Rating tend either be low or high. Not much in between.
reviews %>% 
  select(-UserID) %>% 
  pivot_longer(cols = everything(),
               names_to = "Variable",
               values_to = "Rating") %>% 
  ggplot(aes(x = Rating)) +
  geom_density(color = "steelblue", size = 1) +  # Use density plot
  labs(title = "Distribution of Ratings",
       x = "Rating",
       y = "Density")

# Visualize distribution for each variable to see observe any differences in ratings between variables
# Distributions suggest users are leaving a rating if they have a negative experience for most variables.
reviews %>% 
  select(-UserID) %>% 
  pivot_longer(cols = everything(),
               names_to = "Variable",
               values_to = "Rating") %>% 
  ggplot(aes(x = Rating)) +
  geom_density(color = "steelblue", size = 1) + # Use density plot
  facet_wrap(~ Variable, scales = "free") + # Get density plot for each variable
  labs(title = "Variable Distributions by Destination Type",
       x = "Rating",
       y = "Density")

### Correlation Analysis ####

# Measure correlation between variables
correlation = 
  reviews %>% 
  select(-UserID) %>%
  mutate(Gardens = replace_na(Gardens, 0)) %>% # Address the one NA so that correlation can be calculated
  cor()

heatmap(correlation, Rowv = NA, Colv = NA)

# Which variables have strong correlation? r > 0.5
strong_cor = correlation %>% 
  as.data.frame() %>% # Convert matrix to a data frame
  rownames_to_column(var = "Variable") %>% 
  pivot_longer(-Variable,
               names_to = "Variable 2",
               values_to = "r") %>% # Unpivot data
  filter(
    Variable != `Variable 2`
    & (r > 0.5 | r < -0.5)
  ) %>% 
  mutate(
    Var1 = pmin(Variable, `Variable 2`),
    Var2 = pmax(Variable, `Variable 2`)
  ) %>% 
  select(-Variable, -`Variable 2`) %>% 
  distinct(Var1, Var2, .keep_all = TRUE) %>% 
  rename(Variable = Var1, `Variable 2` = Var2) %>% 
  select(Variable, `Variable 2`, r) %>% 
  mutate(r = round(r, 2))

#### Parks and Theatres ####
## Cluster at the lower ratings that indicate a low rating for parks will generally mean a low rating for Theatres.
reviews %>% 
  ggplot(aes(x = Parks, y = Theatres)) +
  geom_point(color = "steelblue") +
  labs(title = "Parks vs Theatres")

#### Restaurants and Zoo ####
## Restaurant rating translates to a similar Zoo rating.
## Cluster between 2 and 4 where increasing Restaurant ratings translate to lower Zoo ratings.
reviews %>% 
  ggplot(aes(x = Restaurants, y = Zoo)) +
  geom_point(color = "steelblue") +
  labs(title = "Restaurants vs Zoo")

#### Pubs/Bars and Zoo ####
## Pub/Bars rating translates to a similar Zoo rating.
## Cluster between 2 and 4 where increasing Restaurant ratings translate to lower Zoo ratings.
reviews %>% 
  ggplot(aes(x = Pubs_Bars, y = Zoo)) +
  geom_point(color = "steelblue") +
  labs(title = "Pubs/Bars vs Zoo")

#### Pubs/Bars and Restaurants ####
## Pub/Bars rating translates to a similar Restaurant rating.
reviews %>% 
  ggplot(aes(x = Pubs_Bars, y = Restaurants)) +
  geom_point(color = "steelblue") +
  labs(title = "Pubs/Bars vs Restaurants")

#### Hotel / Other Lodgings and Juice Bars ####
## Several lanes that suggest hotel rating will translate to similar Juice Bar rating.
reviews %>% 
  ggplot(aes(x = Hotels_OtherLodgings, y = JuiceBars)) +
  geom_point(color = "steelblue") +
  labs(title = "Hotel / Other Lodgings vs Juice Bars")

#### Gyms and Swimming Pools ####
## One big cluster around 1. If users don't like gyms, they won't like pools.
reviews %>% 
  ggplot(aes(x = Gyms, y = `Swimming Pools`)) +
  geom_point(color = "steelblue") +
  labs(title = "Gyms vs Swimming Pools")

## Step 3: Data pre-processing ####

# Remove the User ID variable
df = 
  reviews %>% 
  select(-UserID)

# Replace the one NA in Gardens with 0
# Assuming that this user has no ratings for Gardens
df = 
  df %>% 
  mutate(Gardens = replace_na(Gardens, 0))

# Verify that the NA is changed
colSums(is.na(df))

## Step 4: Feature Engineering ####

# Scale Data
standardize = preProcess(df,
                         method = c("center", "scale"))

# TEAM - Use this data frame for your models!!!!!!!!!!!!!! :)
df_s = predict(standardize, df)

## Step 5: Feature & Model Selection ####

### K-Means Clustering ####

#### Determine Optimal Number of Clusters ----

# Elbow method
fviz_nbclust(df_s, kmeans, k.max = 10, method = "wss") +
  labs(subtitle = "Elbow Method for K-Means")

# Silhouette method
fviz_nbclust(df_s, kmeans, k.max = 10, method = "silhouette") +
  labs(subtitle = "Silhouette Method for K-Means")

#### K-Means Clustering ----

# Apply K-Means with the optimal k (assume k = 2 initially)
final_clus <- kmeans(df_s, centers = 2, nstart = 25)

# Visualizing K-Means Clusters
fviz_cluster(final_clus, data = df_s, palette = "jco") +
  labs(subtitle = "K-Means Clustering without PCA")

# Cluster sizes
table(final_clus$cluster)

# Silhouette Analysis for K-Means
silh <- silhouette(final_clus$cluster, dist(df_s))
fviz_silhouette(silh) + labs(subtitle = "Silhouette Analysis for K-Means")

# Attach k-means PCA clusters to reviews
reviews_clust = cbind(reviews, cluster = final_clus$cluster)

### Principal Component Analysis ####

#Run PCA without target variable
travel_pca <- prcomp(df_s, scale. = TRUE)

get_eig(travel_pca) #Obtain eigenvalues

travel_pca$rotation #variable loadings (i.e. eigenvectors)

head(travel_pca$x) #PC values/scores for all observations (i.e., projected values onto principal component axes)

cor(travel_pca$x) |> round(3) # projected data is orthogonal (zero covariance)

#### Variables analysis ####
get_eig(travel_pca) |>
  round(6) #Obtain eigenvalues and variance percent. The first 9 PCs explain 70.47% of the variation

get_eig(travel_pca) |>
  ggplot(aes(x = eigenvalue, y = variance.percent)) + 
  geom_point()

fviz_eig(travel_pca, addlabels=TRUE) #Visualize explained variances per component
#Plot shows that 43% (19.7+14.6+7.7) of the variation in the data is explained by the first three PCs

#Extract all the results (coordinates, squared cosine, contributions) for variables from PCA outputs
#The components of the get_pca_var() can be used in the plot of variables
pca_by_var <- get_pca_var(travel_pca)

pca_by_var$coord #coordinates of each variable

pca_by_var$cos2 #squared cosine of each variable, indicates importance of each dimension to each variable

pca_by_var$contrib #contributions of each variable to each component

#graph of variables mapped over the first two components
fviz_pca_var(
  travel_pca,
  repel = TRUE
) #User repel to avoid text overlapping (slow if many points)

#Quality of representation:
corrplot(
  pca_by_var$cos2, 
  is.corr=FALSE
) #Visualize quality of variables

#Contribution of variables to PCs:
corrplot(
  pca_by_var$contrib, 
  is.corr=FALSE
) #Visualize quality of variables

#Bar plots of variables contributions, sorted from highest to lowest
fviz_contrib(travel_pca, choice = "var", axes = 1) # Contributions of variables to PC1

fviz_contrib(travel_pca, choice = "var", axes = 2) # Contributions of variables to PC2

fviz_contrib(travel_pca, choice = "var", axes = 3) # Contributions of variables to PC3

#A circular plot for contribution values
# Color by contribution values
fviz_pca_var(
  travel_pca, col.var = "contrib",
  gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
  repel=TRUE
) #Use repel to avoid text overlapping (slow if many points)

#### Observation analysis ####

#Extract PCA results for individuals (i.e., observations)
#get_pca_ind () returns results for individuals, such as coord, cos2, and contrib
ind <- get_pca_ind(travel_pca)

ind$coord #coordinates of individuals (i.e., individual PCA values)

ind$cos2 #cos2 of each individual

ind$contrib #contributions of each individual to each component

fviz_pca_ind(travel_pca) #Plot on individual PC1 and PC2 coordinates/scores

#Plotting quality and contribution:

#coloring variables by cos2 values
fviz_pca_ind(travel_pca, col.ind = "cos2", 
             gradient.cols = c("#0073C2FF", "#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE)


# Contribution on PC1, shown for the top 50 individuals (change axes to 1:n for PC1 to PCn)
fviz_contrib(travel_pca, choice = "ind", axes=1, top=50)

#Brandscombdata
#Attach userIDs and the principal components values of each amenity
pcs <- as_tibble(travel_pca$x)

combdata <- tibble(
  userid = reviews$UserID, 
  pcs
)

ggplot(combdata, aes(x=PC1, y=PC2, color=userid)) +
  geom_point() + 
  labs(title="Destination Amenities by two PCs") +
  geom_text(aes(label=userid),nudge_x=0.25,nudge_y=0.25,check_overlap=T) +
  theme_bw()

### K-Means Clustering Using PCA ####

# Use the three principal components to build a another k-means model
df_pca = pcs %>% select(PC1, PC2, PC3)

#### Determine Optimal Number of Clusters ####
# Based on the two methods used. 4 appears to be the optimal number.

#Elbow method - with maximum 10 clusters
fviz_nbclust(df_pca, kmeans, k.max=10, nstart=25, method="wss")


#Silhouette method - maximum 10 clusters
fviz_nbclust(df_pca, kmeans, k.max=10, nstart=25, method="silhouette")

# Define k-means model
k_pca = kmeans(df_pca, centers = 4, nstart = 25)

# Visualize the clusters
fviz_cluster(k_pca, data = df_pca)

# Visualize silhouettes based on k = 4
silh_pca = silhouette(k_pca$cluster, dist(df_pca))
fviz_silhouette(silh_pca)

# Visualizing K-Means Clusters
fviz_cluster(k_pca, data = df_s, palette = "jco") +
  labs(subtitle = "K-Means Clustering without PCA")

# Cluster sizes
table(k_pca$cluster)

# Attach k-means PCA clusters to reviews
reviews_clust_pca = cbind(reviews, cluster_pca = k_pca$cluster)

## Step 6: Model Validation ####
# N/A

## Step 7: Predictions and Conclusions ####

# Aggregate clusters with means to interpret results without PCA
summary_clusters = 
  reviews_clust %>%
  group_by(cluster) %>%
  summarise(across(where(is.numeric), \(x) mean(x, na.rm = TRUE)))

# Aggregate clusters with means to interpret results with PCA
summary_clusters_pca = 
  reviews_clust_pca %>%
    group_by(cluster_pca) %>%
    summarise(across(where(is.numeric), \(x) mean(x, na.rm = TRUE)))

summary_clusters_pca

# Unpivot for analysis and visualization
summary_clusters_df = 
  summary_clusters_pca %>%
  pivot_longer(-cluster_pca, names_to = "Variable", values_to = "mean_rating")

# Visualize results
summary_clusters_df %>% 
  mutate(high_rating = mean_rating > 3.5) %>% 
  ggplot(aes(x = Variable, y = mean_rating, fill = high_rating)) +
  geom_bar(stat = "identity") + 
  coord_flip() +
  facet_wrap(~ cluster_pca) +
  scale_fill_manual(values = c('TRUE' = 'steelblue', 'FALSE' = 'lightsteelblue')) +
  labs(x = "Variable", y = "Mean Rating", title = "Mean Ratings by Cluster", fill = "Rating > 3.5")

### Distribution Analysis of Clusters ####

# Cluster 1
reviews_clust_pca %>% 
  filter(cluster_pca == 1) %>% 
  select(-UserID, -cluster_pca) %>% 
  pivot_longer(cols = everything(),
               names_to = "Variable",
               values_to = "Rating") %>% 
  ggplot(aes(x = Rating)) +
  geom_density(color = "steelblue", size = 1) +  # Use density plot
  facet_wrap(~ Variable, scales = "free") + # Get density plot for each variable
  xlim(0, 5) + # Set x scale from 0 to 5
  labs(title = "Distribution of Ratings",
       subtitle = "Cluster 1",
       x = "Rating",
       y = "Density")

# Cluster 2
reviews_clust_pca %>% 
  filter(cluster_pca == 2) %>% 
  select(-UserID, -cluster_pca) %>% 
  pivot_longer(cols = everything(),
               names_to = "Variable",
               values_to = "Rating") %>% 
  ggplot(aes(x = Rating)) +
  geom_density(color = "steelblue", size = 1) +  # Use density plot
  facet_wrap(~ Variable, scales = "free") + # Get density plot for each variable
  xlim(0, 5) + # Set x scale from 0 to 5
  labs(title = "Distribution of Ratings",
       subtitle = "Cluster 2",
       x = "Rating",
       y = "Density")

# Cluster 3
reviews_clust_pca %>% 
  filter(cluster_pca == 3) %>% 
  select(-UserID, -cluster_pca) %>% 
  pivot_longer(cols = everything(),
               names_to = "Variable",
               values_to = "Rating") %>% 
  ggplot(aes(x = Rating)) +
  geom_density(color = "steelblue", size = 1) +  # Use density plot
  facet_wrap(~ Variable, scales = "free") + # Get density plot for each variable
  xlim(0, 5) + # Set x scale from 0 to 5
  labs(title = "Distribution of Ratings",
       subtitle = "Cluster 3",
       x = "Rating",
       y = "Density")

# Cluster 4
reviews_clust_pca %>% 
  filter(cluster_pca == 4) %>% 
  select(-UserID, -cluster_pca) %>% 
  pivot_longer(cols = everything(),
               names_to = "Variable",
               values_to = "Rating") %>% 
  ggplot(aes(x = Rating)) +
  geom_density(color = "steelblue", size = 1) +  # Use density plot
  facet_wrap(~ Variable, scales = "free") + # Get density plot for each variable
  xlim(0, 5) + # Set x scale from 0 to 5
  labs(title = "Distribution of Ratings",
       subtitle = "Cluster 4",
       x = "Rating",
       y = "Density")



            