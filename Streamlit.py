import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])

import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.subplots as sp

#subprocess.run(["streamlit", "run", "Streamlit.py"])

import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
import sys

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)
np.set_printoptions(threshold=sys.maxsize)

# limited to 2024 to prioritize most relevant year
year_list = [2024]


def extract_players(player_data):
    # convert clutch ratio, which already accounts for total rounds
    player_data[['won', 'played']] = player_data['Clutches (won/played)'].str.split('/', expand=True)
    # the ratio can be NaN if it never happened
    player_data['won'] = pd.to_numeric(player_data['won'], errors='coerce')
    player_data['played'] = pd.to_numeric(player_data['played'], errors='coerce')
    # get total for player
    players_grouped = player_data.groupby('Player').agg({'won': 'sum', 'played': 'sum'})
    # fix for when a player never entered a clutch scenario
    players_grouped['Clutch_Ratio'] = players_grouped.apply(
        lambda row: row['won'] / row['played'] if row['played'] != 0 else 0, axis=1)

    # these stats are stored as strings due to the % sign, they need to be converted
    stats_to_convert_to_num = ['Kill, Assist, Trade, Survive %', 'Headshot %']
    for col in stats_to_convert_to_num:
        player_data[col] = player_data[col].str.rstrip('%').astype(float) / 100

    stats_to_average = ['Average Combat Score', 'Headshot %', 'Assists Per Round', 'First Deaths Per Round']
    weighted_avg_dict = {}
    for stat in stats_to_average:
        player_data[f'weighted_{stat}'] = player_data[stat] * player_data['Rounds Played']
        total_rounds = player_data.groupby('Player')['Rounds Played'].sum()
        weighted_stat_sum = player_data.groupby('Player')[f'weighted_{stat}'].sum()
        weighted_avg_dict[stat] = weighted_stat_sum / total_rounds
    weighted_avg_df = pd.DataFrame(weighted_avg_dict)
    full_stats_df = pd.concat([weighted_avg_df, players_grouped['Clutch_Ratio']], axis=1)

    # normalize 0-1, this seemed to work better than using std
    normalized_stats = (full_stats_df - full_stats_df.min()) / full_stats_df.max()
    return normalized_stats


# pull all the player data to make sure the clusters are the same between years
player_data_list = []
for year in year_list:
    partial_player_data = pd.read_csv(f'vct_{year}/players_stats/players_stats.csv')
    player_data_list.append(partial_player_data)
full_player_df = pd.concat(player_data_list, ignore_index=True)
normal_player_data = extract_players(full_player_df)
#normal_player_data.to_excel('test.xlsx')

best_cluster_num = -1
best_silhouette_score = -1
for i in range(3, 10, 1):
    labels = AgglomerativeClustering(n_clusters=i).fit(normal_player_data).labels_
    silhouette_score_result = silhouette_score(normal_player_data, labels)
    if silhouette_score_result > best_silhouette_score:
        best_silhouette_score = silhouette_score_result
        best_cluster_num = i

labels = AgglomerativeClustering(n_clusters=best_cluster_num).fit(normal_player_data).labels_
normal_player_data['cluster'] = labels

# get summary data about the clusters
cluster_data = normal_player_data.groupby('cluster').mean().reset_index()
print(cluster_data.head())
print(cluster_data.describe())
#cluster_data.to_excel('cluster_test.xlsx')

# combines the datasets to find the teams that played in a round
match_player_list = []
for year in year_list:
    print('starting', year)
    # load players and matches
    match_results = pd.read_csv(f'vct_{year}/matches/win_loss_methods_count.csv')
    win_columns = ['Elimination', 'Detonated', 'Defused', 'Time Expiry (No Plant)']
    match_results['Win'] = match_results[['Elimination', 'Detonated', 'Defused', 'Time Expiry (No Plant)']].sum(
        axis=1) == 13
    match_results['MatchKey'] = match_results[['Tournament', 'Stage', 'Match Type', 'Team']].agg(' - '.join, axis=1)
    player_stats = pd.read_csv(f'vct_{year}/players_stats/players_stats.csv')
    player_stats['MatchKey'] = player_stats[['Tournament', 'Stage', 'Match Type', 'Teams']].agg(' - '.join, axis=1)

    # merge to get the list of players for a particular match
    player_stats = player_stats.drop(['Tournament', 'Stage', 'Match Type', 'Teams'], axis=1)
    merged = match_results.merge(player_stats, how='left', on='MatchKey')
    players_per_match = merged.groupby('MatchKey')['Player'].agg(set)
    match_results['Players'] = match_results['MatchKey'].map(players_per_match)
    match_player_list.append(match_results)

match_player_df = pd.concat(match_player_list)

match_player_df['Players'] = match_player_df['Players'].apply(list)
match_player_df['team_comp'] = match_player_df['Players'].apply(
    lambda players: [normal_player_data['cluster'][player] for player in players]
)

# Transform `team_comp` to count occurrences of each cluster
def count_clusters(comp):
    return tuple(comp.count(i) for i in range(3))  # Example: (count_0, count_1, count_2)


# Apply the transformation to create a new column for cluster counts
match_player_df['cluster_counts'] = match_player_df['team_comp'].apply(count_clusters)


# Add Opponent's cluster counts using Match Name
def assign_opponent_cluster(df):
    # Create a mapping for team to cluster_counts
    team_to_cluster = dict(zip(df['Team'], df['cluster_counts']))

    # Extract opponent name from 'Match Name'
    def get_opponent_team(row):
        teams = row['Match Name'].split(' vs ')
        return teams[0] if teams[1] == row['Team'] else teams[1]

    # Assign opponent cluster_counts
    df['Opponent_team'] = df.apply(get_opponent_team, axis=1)
    df['Opponent_cluster'] = df['Opponent_team'].map(team_to_cluster)
    return df


match_player_df = assign_opponent_cluster(match_player_df)


# Define a function to calculate same and different type win rates
def calculate_win_rates(group):
    cluster_type = group['cluster_counts'].iloc[0]

    # Same type matches
    same_type_matches = group[group['Opponent_cluster'] == cluster_type]
    total_same = len(same_type_matches)
    same_wins = same_type_matches['Win'].sum()

    # Different type matches
    different_type_matches = group[group['Opponent_cluster'] != cluster_type]
    total_different = len(different_type_matches)
    different_wins = different_type_matches['Win'].sum()

    return pd.Series({
        'same_type_matches': total_same,
        'same_type_winrate': same_wins / total_same if total_same > 0 else 0,
        'different_type_matches': total_different,
        'different_type_winrate': different_wins / total_different if total_different > 0 else 0
    })


# Group by cluster_counts and calculate win rates
team_stats = match_player_df.groupby('cluster_counts', group_keys=False).apply(calculate_win_rates).reset_index()

# Add total matches and wins from earlier
team_stats = team_stats.merge(
    match_player_df.groupby('cluster_counts').agg(total_matches=('Win', 'count'), wins=('Win', 'sum')).reset_index(),
    on='cluster_counts'
)

# Calculate overall win rate
team_stats['overall_winrate'] = team_stats['wins'] / team_stats['total_matches']

# Add total team counts
team_counts = match_player_df.groupby('cluster_counts')['Team'].nunique().reset_index()
team_counts.rename(columns={'Team': 'total_teams'}, inplace=True)
team_stats = team_stats.merge(team_counts, on='cluster_counts')

team_stats = team_stats.sort_values('overall_winrate')
# Final sorted result
print(team_stats)
#team_stats.to_excel('overall_winrate.xlsx')

filtered_teams = match_player_df[match_player_df['cluster_counts'] == (0, 3, 2)]

# Print out the teams
# print(filtered_teams[['Team', 'cluster_counts']])
print(filtered_teams['Team'].unique())

# the following is made by chatGPT
#https://chatgpt.com/share/67501031-86a8-8011-88bf-4a6592680286

numeric_columns = [col for col in cluster_data.columns if col != "cluster"]
clusters = cluster_data.index
fig = go.Figure()
for cluster in clusters:
    fig.add_trace(go.Bar(
        x=numeric_columns,
        y=cluster_data.loc[cluster, numeric_columns].values,
        name=f"Cluster {cluster}",
    ))
fig.update_layout(
    barmode='group',
    title="Statistics Per Cluster",
    xaxis_title="Statistics",
    yaxis_title="Average, all 0-1",
    template="plotly_dark"
)
st.plotly_chart(fig)

player_counts = normal_player_data['cluster'].value_counts()
pie_fig = go.Figure(
    data=[go.Pie(
        labels=player_counts.index,
        values=player_counts.values,
        hoverinfo='label+percent',
        textinfo='value',
        name="Clusters",
    )]
)
pie_fig.update_layout(
    title="Player Distribution Across Clusters",
    template="plotly_dark",
    legend_title="Cluster Labels"
)
st.plotly_chart(pie_fig)


fig = go.Figure()
fig.add_trace(go.Bar(
    x=team_stats['cluster_counts'].astype(str),
    y=team_stats['different_type_winrate'],
    name="Different Type Win Rate",
    marker_color='orange'
))
fig.add_trace(go.Bar(
    x=team_stats['cluster_counts'].astype(str),
    y=team_stats['overall_winrate'],
    name="Overall Win Rate",
    marker_color='blue'
))
fig.update_layout(
    barmode='group',
    title="Win Rates by Team Composition",
    xaxis_title="Team Composition (0s, 1s, 2s)",
    yaxis_title="Win Rate",
    legend_title="Win Rate Type",
    template="plotly_dark"
)
st.plotly_chart(fig)
