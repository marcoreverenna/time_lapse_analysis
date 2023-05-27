### Time lapse analysis

Time-lapse technology is an assisted reproduction technology (ART) performed by in vitro fertilization (IVF) centers to monitor the stages of embryo development. This study aimed to investigate segmental aneuploidies by collecting time-lapse data from three different IVF centers: IVIRMA, Care-Fertility and GeneraLife. The time-lapse data immediately revealed differences between embryos of different classes, including full aneuploids, segmental aneuploids and euploids. These differences were confirmed by statistical tests, including Kolmogorov-Smirnov, Wilcoxon signed-rank and Binomial tests, which demonstrated even marked differences (p-values < 0.01). Exploratory data analyses (EDA) indicated that segmental aneuploidies show a greater delay in development compared to euploid embryos and even compared to aneuploidies. The multicenter dataset preparation played a crucial role in the analysis. Therefore, strict quality filtering criteria were adopted, engineered variables were defined and a linear interpolation imputation method was established to enrich the dataset without losing variability. Only after the dataset was filtered for outliers and enriched with imputed times and engineered variables, it was possible to create machine learning models and test them among different centers. This study provides insights into the differences in embryo development between different classes and emphasizes the importance of proper data preparation and guidelines in conducting accurate and informative analyses. Future research could build on these findings and methods to explore other aspects of embryo development and their impact on reproductive outcomes.

#### Analsysis workflow

Starting inputs: Dataset IVIRMA Dataset Care-Fertility Dataset GeneraLife

We obtain the classification from IVIRMA using the chromosome variables
To obtain the classification in Care-Fertility and GeneraLife we did it manually for each embryo looking the descriptive variable

#### STEP 1 = 0_stardard_input

Info: starting from original dataset, create for each center a standard dataset to start the analysis
- data cleaning IVIRMA
- data cleaning Care-Fertility
- data cleaning GeneraLife
outputs: va_standard.xlsx uk_standard.xlsx ro_standard.xlsx

#### STEP 2 = 1_imput_features_eng

Info: preparing your master dataset to the next univariate and multivariate analysis
- summary numbers
- missing values
- boxplots and histogram (you can see where are the missing values and the delay of segmentals)
- linear imputation
- features engineering
outputs: final_dataset.xlsx

#### STEP 3 = 2_univariate_analysis

Info:
- checking if your data is parametric (shapiro and bartlet)
- if your data is parametric use T-test two sample and T-test paired
- if your data is not parametric use Kolmogorov Smirnov
- binomial test
- correlation heatmap
- tails analysis

#### STEP 4 = 3_multivariate_analysis

Info:
- check which model is more appropiate (random forest or logistic regression)
- create logistic regression model with imputed times + delays , indipendent times, standardised times
- make ROC curves for each type of logistic regression model
- make a gaussianization for each center from indipendent times and create ROC curves

#### STEP 5 = 4_translatability

Info:
- check which model perform well in other dataset
