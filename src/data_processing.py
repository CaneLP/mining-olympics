import pandas as pd
import numpy as np
from sklearn import preprocessing


def get_data(
        fields=None,
        noc=None,
        merge_noc=True,
        printable=False
):
    if fields is None:
        athlete_events = pd.read_csv('../data/athlete_events.csv', error_bad_lines=False)
    else:
        if noc is not None:
            fields.append('NOC')

        athlete_events = pd.read_csv('../data/athlete_events.csv', error_bad_lines=False, usecols=fields)

    # fields = ['Name', 'Sex', 'Age', 'Height', 'Weight', 'Team', 'NOC', 'Games', 'Year', 'Season', 'City', 'Sport',
    #           'Event', 'Medal']

    df = pd.DataFrame(athlete_events)

    if noc is not None:
        nationality_filter = athlete_events['NOC'] == noc
        df = df[nationality_filter]

    if merge_noc is True:
        noc_regions = pd.read_csv('../data/noc_regions.csv')
        df = pd.merge(df, noc_regions, on='NOC', how='left')

        df['region'].fillna(noc_regions['notes'], inplace=True)
        df['notes'].fillna('None', inplace=True)
        df['region'].fillna('None', inplace=True)

    if printable is True:
        pd.set_option('display.max_rows', None)
        pd.set_option('expand_frame_repr', False)
        print(df)

    # posto su u pitanju medjuigre i IOC ne priznaje, brisemo
    df = df[df['Year'] != 1906]

    return df


def fill_na_attr_rows(data):
    na_attributes = data.columns[data.isna().any()].tolist()
    # print(na_attributes)

    data['Medal'].fillna('None', inplace=True)

    age_mean = int(data['Age'].mean())
    data['Age'] = data['Age'].fillna(age_mean)
    data['Age'].isna().sum()

    height_mode = data['Height'].mode()[0]
    data['Height'] = data['Height'].fillna(height_mode)
    data['Height'].isna().sum()

    weight_mode = data['Weight'].mode()[0]
    data['Weight'] = data['Weight'].fillna(weight_mode)
    data['Weight'].isna().sum()

    return data


def get_na_rows_count(data):
    null_rows = data[data.isna().any(axis=1)]
    return null_rows.shape[0]


def create_countries_data(data):
    # ucitavanje podataka i dodavanje indikatorske kolone -
    # da li je medalja osvojena ili ne
    data['MedalWon'] = np.where(data.loc[:, 'Medal'] == 'None', 0, 1)

    # izracunavanje broja osvojenih medalja za svaku drzavu
    # ucesnicu grupisanjem po atributu 'region' i
    # sumiranjem prethodno napravljene kolone
    country_medals_won_athletes = data.groupby(['region'])['MedalWon'].agg('sum').reset_index()
    country_medals_won_athletes = country_medals_won_athletes.sort_values('MedalWon', ascending=False)
    country_medals_won_athletes.rename(columns={'MedalWon': 'MedalsCount'}, inplace=True)

    # izracunavanje ukupnog broja predstavnika na svim
    # olimpijadama za svaku drzavu
    athletes_count = data.drop_duplicates(['ID', 'Year'])
    athletes_count = athletes_count.groupby(['region'])['ID'].count().reset_index()
    athletes_count.rename(columns={'ID': 'ParticipantsCount'}, inplace=True)

    # izracunavanje ukupnog broja olimpijada na kojima
    # je svaka drzava ucestvovala
    games_count = data.drop_duplicates(['region', 'Games'])
    games_count = games_count.groupby(['region'])['Year'].count().reset_index()
    games_count.rename(columns={'Year': 'GamesCount'}, inplace=True)

    # spajanje prethodno izracunatih vrednosti u DataFrame,
    # po atributu 'region'
    country_data = pd.DataFrame(country_medals_won_athletes)
    country_data = pd.merge(country_data, athletes_count, on='region')
    country_data = pd.merge(country_data, games_count, on='region')

    # racunanje broja ucesnika po igrama i efikasnosti
    # drzave, i njihovo skaliranje na 1
    country_data['PartPerGamesScaled'] = country_data['ParticipantsCount'] / country_data['GamesCount']
    country_data['SuccessRate'] = country_data['MedalsCount'] / country_data['ParticipantsCount']

    x = country_data[['PartPerGamesScaled', 'SuccessRate']].values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    country_data[['PartPerGamesScaled', 'SuccessRate']] = x_scaled

    country_data.to_csv(r'../data/country_stats.csv')

    return country_data


def get_sizeage(data, medal_won):
    data['HeightNorm'] = (data['Height'] - data['Height'].mean()) / np.std(data['Height'], axis=0)
    data['WeightNorm'] = (data['Weight'] - data['Weight'].mean()) / np.std(data['Weight'], axis=0)
    data['SizeNorm'] = data['HeightNorm'] + data['WeightNorm']
    data['AgeNorm'] = (data['Age'] - data['Age'].mean()) / np.std(data['Age'], axis=0)

    data = data[data['MedalWon'] == medal_won]

    return data[['SizeNorm', 'AgeNorm']]
