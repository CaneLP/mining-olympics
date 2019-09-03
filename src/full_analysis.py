from data_processing import *
from clustering import *


pd.set_option('display.max_rows', None)
pd.set_option('expand_frame_repr', False)

# use this when full.csv is available
data = pd.read_csv('../data/full.csv', error_bad_lines=False)

countries_data = create_countries_data(data)
# print(countries_data.head(20))

# TODO: add options for user to choose columns for clustering and clustering type and all

# klasterovanje K-sredina

k_means_elbow_method(countries_data, attributes=['ParticipantsCount', 'MedalsCount'],
                     xlabel='Broj klastera', ylabel='Rezultat (WCSS)')
n_clusters = int(input("Enter number of clusters to use for K-means clustering: "))
k_means_clustering(countries_data, attributes=['ParticipantsCount', 'MedalsCount'], n_clusters=n_clusters,
                   xlabel='Broj ucesnika', ylabel='Broj medalja')

k_means_elbow_method(countries_data, attributes=['PartPerGamesScaled', 'SuccessRate'],
                     xlabel='Broj klastera', ylabel='Rezultat (WCSS)')
n_clusters = int(input("Enter number of clusters to use for K-means clustering: "))
k_means_clustering(countries_data, attributes=['PartPerGamesScaled', 'SuccessRate'], n_clusters=n_clusters,
                   xlabel='Broj ucesnika po olimpijadi', ylabel='Uspesnost takmicara')


# hijearhijsko klasterovanje

Z = hierarchy_dendrogram(countries_data, attributes=['ParticipantsCount', 'MedalsCount'], max_d=3000,
                         xlabel='Redni broj klastera (Broj klastera u klasteru)', ylabel='Rastojanje')
n_clusters = int(input("Enter number of clusters to use for hierarchical clustering: "))
hierarchy_clustering(countries_data, Z, n_clusters=n_clusters, attributes=['ParticipantsCount', 'MedalsCount'],
                     max_d=3000, xlabel='Broj ucesnika', ylabel='Broj medalja')

Z = hierarchy_dendrogram(countries_data, attributes=['PartPerGamesScaled', 'SuccessRate'], max_d=0.25,
                         xlabel='Redni broj klastera (Broj klastera u klasteru)', ylabel='Rastojanje')
n_clusters = int(input("Enter number of clusters to use for hierarchical clustering: "))
hierarchy_clustering(countries_data, Z, n_clusters=n_clusters, attributes=['PartPerGamesScaled', 'SuccessRate'],
                     max_d=0.25, xlabel='Broj ucesnika po olimpijadi', ylabel='Uspesnost takmicara')

# normalizacija potrebnih podataka (visina, tezina, godine) i klasterovanje ucesnika po tim parametrima

size_age_medal = get_sizeage(data, medal_won=1)
size_age_no_medal = get_sizeage(data, medal_won=1)

k_means_elbow_method(size_age_medal, ['SizeNorm', 'AgeNorm'])
n_clusters = int(input("Enter number of clusters to use for K-means clustering: "))
k_means_clustering(size_age_medal, n_clusters, ['SizeNorm', 'AgeNorm'],
                   xlabel='Normalizovan zbir visine i tezine', ylabel='Normalizovan broj godina',
                   save_plot=True, save_plot_path='clusters-km-normalized-data-medal.png')

k_means_elbow_method(size_age_no_medal, ['SizeNorm', 'AgeNorm'])
n_clusters = int(input("Enter number of clusters to use for K-means clustering: "))
k_means_clustering(size_age_no_medal, n_clusters, ['SizeNorm', 'AgeNorm'],
                   xlabel='Normalizovan zbir visine i tezine', ylabel='Normalizovan broj godina',
                   save_plot=True, save_plot_path='clusters-km-normalized-data-no-medal.png')
