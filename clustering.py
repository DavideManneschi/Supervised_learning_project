import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler # standard scaler
from sklearn.decomposition import PCA  # pca
from scipy.cluster.hierarchy import dendrogram, linkage # dendrograms
from sklearn.cluster         import KMeans # k-means clustering



def import_data():
    path = "C:/Users/Davide Manneschi/Documents/python_datasets_ml/Mobile_App_Survey_Data.xlsx"
    read = pd.read_excel(path, na_values="NaN")

    return read


def checking_missing_values(file):
    ## checking values

    missing_values = file[file.iloc[:, 0:] == 0].sum()


def describe_dataset(file):
    descriptive_statistics = file.describe()

    return descriptive_statistics





def dividing_dataset(file):

    total_keys = [i for i in file.keys()]

    total_keys_2 = [i for i in file.keys()]
    demographic_list = []

    non_dem_keys = [z for z in total_keys if "q24" in z or "q25" in z or "q26" in z]

    for i in non_dem_keys:
        if i in total_keys_2:
            total_keys_2.remove(i)

    # print(total_keys_2)

    non_demographic_final_dataset = file[[i for i in non_dem_keys]]
    demographic_final_dataset = file[[i for i in total_keys_2]]

    chomprensive_final_complete_dataset = pd.concat([demographic_final_dataset, non_demographic_final_dataset], axis=1)

    print(chomprensive_final_complete_dataset)
    return chomprensive_final_complete_dataset








def log_dataframe(file):
    data_dem, data_non_dem = file.iloc[:, :48], file.iloc[:, -40:]


    log_data_non_dem = pd.DataFrame(np.log(data_non_dem))
    log_gistograms = log_data_non_dem.hist()

    complete_dataframe_with_log = pd.concat([data_dem, log_data_non_dem], axis=1)

    print(complete_dataframe_with_log)

    return complete_dataframe_with_log



#################################################################################################

def scaling_dataset(file):
    data_dem, data_non_dem = file.iloc[:, :48], file.iloc[:, -40:]
    print(data_non_dem.keys())



    #à faccio trasposta
    trasposed_data_dem, trasposed_data_non_dem = file.iloc[:, :48].T, file.iloc[:, -40:].T



    ## chiamo lo scaler
    scaler_function = StandardScaler()


    ## traspondo la seconda
    scaler_function.fit(trasposed_data_non_dem)


    ## scalo le righe
    scaled_data = pd.DataFrame(scaler_function.transform(trasposed_data_non_dem))

    re_transposed_dem,re_transposed_non_dem=trasposed_data_dem.T,scaled_data.T



    final_dataframe_scaled = pd.concat([re_transposed_dem, re_transposed_non_dem], axis=1)

    print(final_dataframe_scaled)


    ## i still need to scale the columns!

    ## scale the columns
    scaler_function.fit(re_transposed_non_dem)
    scaled_final_columns=pd.DataFrame(scaler_function.transform((re_transposed_non_dem)))
    print("the scaled",scaled_final_columns)

    the_final_dataframe_scaled = pd.concat([re_transposed_dem, scaled_final_columns], axis=1)

    print(the_final_dataframe_scaled)












    return  the_final_dataframe_scaled





##################################################################################################



#### scree plot##################################################


def scree_plot(pca_object):
    fig, ax = plt.subplots(figsize=(10, 8))
    features = range(pca_object.n_components_)

    # developing a scree plot
    plt.plot(features,
             pca_object.explained_variance_ratio_,
             linewidth=2,
             marker='o',
             markersize=10,
             markeredgecolor='black',
             markerfacecolor='grey')

    # setting more plot options
    plt.title('Scree Plot')
    plt.xlabel('PCA feature')
    plt.ylabel('Explained Variance')
    plt.xticks(features)



    # displaying the plot
    plt.show()


###############################################################################################




def pca_analysis(file):



    count=0

    data_dem, data_non_dem = file.iloc[:, :48], file.iloc[:, -40:]

    pca = PCA(n_components=6, random_state=219)

    fitting_data = pca.fit_transform(data_non_dem)



    for i in pca.explained_variance_ratio_:

        count+=1
    print(pca.explained_variance_ratio_.sum())






    return pca











### final dataframe with pca










######## factor loading#####################################


def factor_loading(pca):





    index=["keep up tech dev","people ask my advice","enjoy purchasing gadgets","there is too much tech",
           "i enjoy tech for control in my life","i look forr aps for saving time","music is an important part in my life",

           "i like learning about tv shows when i am not watching","there is too much info on social media",
           "im checking friends and family in social media",
           "the internet make easy to stay in touch with friends and family ",
           "the internet make it easy to avoid friend and family",
           "i am opinion leader","i like to stand out from others","i like to give advice to others","i like to take the lead in decision making",

           "im the first to try new things","i dont like responsability","i like beeing in control","i am a risk taker","i think i am creative",

           "i am optimistic ","i am active and on the go","i feel streached for the time","i look for bargain, deals","i like any kind of shopping",

           "i like package deals so i don't have to plan",

           "i shop online",

           "i like luxury brands","i buy desiner brands"," i can not get enought apps","it not the number but how cool they are","i like to show off",

           "my children have an impact","its worth to spend extra dollar for  extra app features","there is no point in earning money not to spend",
           "i am influenced by what is hot","i buy brands that reflect my style","i make impulse purchases","my mobile is a source of entertainament"

            ]

    factors=pd.DataFrame((pca.components_).round(2))
    indexed=factors.set_axis(index,axis=1)
    transposed_indexed=np.transpose(indexed)


    print(transposed_indexed)

    transposed_indexed.to_excel("C:/Users/Davide Manneschi/Desktop/cartella attuale/transposed_1.xlsx",
                                                    )


    transposed_indexed.columns=["Busy tech savy","Time available","Trend follower",
                      "Earning boomer","practocal millenial","Kardashian"]


    print(transposed_indexed)

    return transposed_indexed





def analyse_factor_loading(pca,file,factor_loading_):##### qua per loading
    data_dem, data_non_dem = file.iloc[:, :48], file.iloc[:, -40:]

    loading_factors=pca.transform(data_non_dem)
    dataframe_factor_loading=pd.DataFrame(loading_factors)
    dataframe_factor_loading.columns=factor_loading_.columns

    print(dataframe_factor_loading)

    print(np.var(dataframe_factor_loading))


    return dataframe_factor_loading



def scaling_factor_loading(file):
    ## chiamo lo scaler
    scaler_function = StandardScaler()

    scaler_function.fit(file)
    scaled_transform_dataframe=pd.DataFrame(scaler_function.transform(file))
    scaled_transform_dataframe.columns=file.columns

    print(np.var(scaled_transform_dataframe))
    return scaled_transform_dataframe


def clustering(pca_scaled):
    # grouping data based on Ward distance
    standard_mergings_ward = linkage(y=pca_scaled,
                                     method='ward',
                                     optimal_ordering=True)

    # setting plot size
    fig, ax = plt.subplots(figsize=(12, 12))

    # developing a dendrogram
    dendrogram(Z=standard_mergings_ward,
               leaf_rotation=90,
               leaf_font_size=6)

    # rendering the plot
    plt.show()


def k_means(pca_scaled):
    customers_k_pca = KMeans(n_clusters=4,
                             random_state=219)
    customers_k_pca.fit(pca_scaled)
    customers_kmeans_pca = pd.DataFrame({'Cluster': customers_k_pca.labels_})
    print(customers_kmeans_pca.iloc[:, 0].value_counts())


    centroids_pca = customers_k_pca.cluster_centers_


    centroids_pca_df = pd.DataFrame(centroids_pca)


    centroids_pca_df.columns = ["Busy tech savy","Time available","Trend follower",
                      "Earning boomer","practocal millenial","Kardashian"]


    print(centroids_pca_df.round(2))

    return customers_kmeans_pca


def concat_dataframe_cluster_principal(customers_k_means,factor_loading,file):





    data_dem, data_non_dem = file.iloc[:, :48], file.iloc[:, -40:]



    cluster_k_dataframe = pd.concat([customers_k_means, factor_loading], axis=1)


    dataframe_with_dem=pd.concat([data_dem,cluster_k_dataframe],axis=1)

    brand_=dataframe_with_dem.copy()



    return dataframe_with_dem.copy()




def rename_channels(dataframe_dem):

    pd.set_option("display.max_rows",None)








    age_bins = {1: "under_18",
                     2: "18-24",
              3: "25-29",
              4:"30-34",
              5:"35-39",
              6:"40-44",
              7:"45-49",
              8:"50-54",
              9:"55-59",
              10:"60-64",
              11:"65-over"}

    dataframe_dem["q1"].replace(age_bins, inplace=True)



    number_apps={1:"1-5",
                 2:"6-10",
                 3:"11-30",
                 4:"31+",
                 5:"Don't Know",
                 6:"None"}

    dataframe_dem["q11"].replace(number_apps,inplace=True)


    free_download={1:"None",
                   2:"1%-25%",
                   3:"26%-50%",
                   4:"51%-75%",
                   5:"76%-99%",
                   6:"ALL"}

    dataframe_dem["q12"].replace(free_download,inplace=True)

    visit_websites={1:"Very often",
                    2:"Sometimes",
                    3:"Rarely",
                    4:"Almost never"}

    columns=["q13r1","q13r2","q13r3","q13r4","q13r5","q13r6","q13r7",
                  "q13r8","q13r9","q13r10","q13r11","q13r12"]

    dataframe_dem[columns]=dataframe_dem[columns].replace(visit_websites)


    education_level={1:"High_school",
                     2:"High_school_graduate",
                     3:"Some college",
                     4:"College_graduate",
                     5:"Some post grad studies",
                     6:"Post graduate degree"}

    dataframe_dem["q48"].replace(education_level, inplace=True)


    marital_status={1:"Married",
                    2:"Single",
                    3:"Single with partner",
                    4:"Separated,widowed,divorced"}

    dataframe_dem["q49"].replace(marital_status, inplace=True)





    race={1:"White",
          2:"Black",
          3:"Asian",
          4:"Native hawaian",
          5:"American indian",
          6:"Other race"}

    dataframe_dem["q54"].replace(race, inplace=True)


    etnicity={1:"Yes",
              2:"No"}

    dataframe_dem["q55"].replace(etnicity, inplace=True)


    salary={1:"Under $10000",
            2:"$10000-$14999",
            3:"$15000-$19999",
            4:"$20000-$29999",
            5:"$30000-$39999",
            6:"$40000-$49999",
            7:"$50000-$59999",
            8:"$60000-$69999",
            9:"$70000-$79999",
            10:"$80000-$89999",
            11:"$90000-$99999",
            12:"$100000-$124999",
            13:"$125000-$149999",
            14:"$150000+"}

    dataframe_dem["q56"].replace(salary, inplace=True)


    gender={1:"Male",
            2:"Female"}

    dataframe_dem["q57"].replace(gender, inplace=True)

    dataframe_dem.columns = ["CaseID", "Age", "Iphone*", "Ipod_touch*", "Android*", "Black_berry*", "Nokia*",
                      "Windows_phone*"
                      "Hp*", "Tablet*", "Other_smartphone*", "None*", "Music_identification_apps*",
                      "Tv_check_in_apps*", "Entertainment_apps*",
                      "Tv_show_apps*",
                      "Gaming_apps*", "Social_network*", "General_news_apps*", "Shopping_apps*",
                      "Publications_news_apps*",
                      "Other*", "None*", "No_apps", "Total_number_of_apps", "Free_apps", "Facebook",
                      "Twitter",
                      "Myspace",
                      "Pandora_radio", "Vevo", "Youtube", "AOL_radio", "Last.fm", "Yahoo_entertainment",
                      "IMDB",
                      "Linkedin", "Netflix", "Level_of_education*", "Martial_status*", "No_children*",
                      "Children_under_6*",
                      "Children 6-12*", "Children 13-17*", "Children 18+*", "Race*", "Hispanic/Latino*",
                      "Annual_income*", "Gender*", "Cluster", "Busy tech savy", "Time available", "Trend follower",
                      "Earning parents", "practical millenial", "Kardashian"]

    print(dataframe_dem)

    count=0
    for i in dataframe_dem[""]:
        if i =="very often":
            count+=1

    print("count",count/len(dataframe_dem["Black_berry*"]))






















    return dataframe_dem




def demgraphic_analysis(dataframe_dem):



    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(x="Age",
                y="Busy tech savy",
                hue='Cluster',
                data=dataframe_dem)

    # formatting and displaying the plot
    plt.tight_layout()
    plt.show()






















def correlation_matrix(file):

    data_dem, data_non_dem = file.iloc[:, :48], file.iloc[:, -40:]

    the_corr=data_non_dem.corr()
    heat=sns.heatmap(the_corr,annot=True)
    plt.show()



def correlation_principal_components(pca,data_dem):
    fig, ax = plt.subplots(figsize=(12, 12))

    # developing a PC to feature heatmap
    sns.heatmap(pca.components_,
                cmap='coolwarm',
                square=True,
                annot=True,
                linewidths=0.1,
                linecolor='black')

    # setting more plot options
    plt.yticks([0, 1, 2, 3, 4, 5],
               ["PC 1", "PC 2", "PC 3", "PC 4", "PC 5", "PC 6"])

    plt.xticks(range(0, 88),
               data_dem.columns,
               rotation=60,
               ha='left')

    plt.xlabel(xlabel="Feature")
    plt.ylabel(ylabel="Principal Component")

    # displaying the plot
    plt.show()






#function_to_get_pca=pca_analysis(scaling_dataset(dividing_dataset(import_data())))

#function_to_get_scree_plot=scree_plot(pca_analysis(scaling_dataset(dividing_dataset(import_data()))))

#function_to_display_centriod=(k_means((scaling_factor_loading(analyse_factor_loading(pca_analysis(scaling_dataset(dividing_dataset(import_data()))),scaling_dataset(dividing_dataset(import_data())),
                     #  factor_loading(pca_analysis(scaling_dataset(dividing_dataset(import_data())))))))))





function_to_get_k_means=k_means((scaling_factor_loading(analyse_factor_loading(pca_analysis(scaling_dataset(dividing_dataset(import_data()))),scaling_dataset(dividing_dataset(import_data())),
                       factor_loading(pca_analysis(scaling_dataset(dividing_dataset(import_data()))))))))

function_to_get_factor_loading=(analyse_factor_loading(pca_analysis(scaling_dataset(dividing_dataset(import_data()))),scaling_dataset(dividing_dataset(import_data())),
                      factor_loading(pca_analysis(scaling_dataset(dividing_dataset(import_data()))))))



#function_to_get_dentogram=clustering(function_to_get_k_means)




demgraphic_analysis(rename_channels(concat_dataframe_cluster_principal(function_to_get_k_means,function_to_get_factor_loading,dividing_dataset(import_data())
                                                                           )))
















