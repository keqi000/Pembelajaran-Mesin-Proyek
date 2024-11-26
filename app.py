import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score

# Membaca dataset dari file CSV
matches = pd.read_csv("matches.csv", index_col=0)

# Menghapus kolom yang tidak relevan
del matches["comp"]  # Menghapus kolom 'comp' karena hanya satu kompetisi
del matches["notes"]  # Menghapus kolom 'notes' karena tidak relevan

# Mengubah kolom 'date' menjadi tipe datetime
matches["date"] = pd.to_datetime(matches["date"])

# Membuat kolom target untuk hasil pertandingan (1 untuk menang, 0 untuk tidak)
matches["target"] = (matches["result"] == "W").astype("int")

# Menambahkan kode untuk venue dan lawan
matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes

# Mengambil jam dari kolom 'time'
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")

# Mengambil kode hari dari tanggal
matches["day_code"] = matches["date"].dt.dayofweek

matches = matches.reset_index(drop=True)

# Add team performance metrics
matches['goals_scored'] = matches.groupby('team')['gf'].transform(lambda x: x.rolling(window=5).mean()).fillna(0)
matches['goals_conceded'] = matches.groupby('team')['ga'].transform(lambda x: x.rolling(window=5).mean()).fillna(0)
matches['shots_ratio'] = matches['sh'] / matches['sot'].replace(0, 1)

# Menyiapkan model Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=5, min_samples_leaf=5, max_features='sqrt', random_state=1)

# Memisahkan data menjadi data pelatihan dan pengujian
train = matches[matches["date"] < '2022-01-01']
test = matches[matches["date"] > '2022-01-01']

# Mendefinisikan fitur yang digunakan untuk prediksi
predictors = ["venue_code", "opp_code", "hour", "day_code", "goals_scored", "goals_conceded", "shots_ratio"]

# Melatih model dengan data pelatihan
rf.fit(train[predictors], train["target"])

# Menghitung akurasi pada data pengujian
accuracy = rf.score(test[predictors], test["target"])

# Melakukan prediksi pada data pengujian
preds = rf.predict(test[predictors])

# Menghitung akurasi model
error = accuracy_score(test["target"], preds)

# Menggabungkan hasil aktual dan prediksi
combined = pd.DataFrame(dict(actual=test["target"], predicted=preds))

# Menghitung precision score
precision = precision_score(test["target"], preds)

# Mengelompokkan pertandingan berdasarkan tim
grouped_matches = matches.groupby("team")
group = grouped_matches.get_group("Manchester City").sort_values("date")

# Fungsi untuk menghitung rolling averages
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

# Kolom yang digunakan untuk rolling averages
cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]

# Menghitung rolling averages untuk setiap tim
matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel('team')  # Menghapus level multi-index
matches_rolling.index = range(matches_rolling.shape[0])  # Mengatur ulang index

# Fungsi untuk membuat prediksi
def make_predictions(data, predictors):
    train = data[data["date"] < '2022-01-01']
    test = data[data["date"] > '2022-01-01']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
    error = precision_score(test["target"], preds)
    return combined, error

# Membuat prediksi dan menghitung error
combined, error = make_predictions(matches_rolling, predictors + new_cols)

# Menggabungkan hasil prediksi dengan informasi tambahan dari matches_rolling
combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True)

# Mendefinisikan kelas untuk menangani nilai yang hilang
class MissingDict(dict):
    __missing__ = lambda self, key: key

# Pemetaan nama tim untuk konsistensi
map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves"
}
mapping = MissingDict(**map_values)

# Menambahkan kolom baru untuk nama tim yang dipetakan
combined["new_team"] = combined["team"].map(mapping)

# Menggabungkan hasil prediksi dengan informasi tim baru
merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"])




# Streamlit Interface
st.title("Football premier league Prediction")
st.markdown("### Predict the outcome of a football match")

# Team selection
team_options = sorted(matches['team'].unique())
team1 = st.sidebar.selectbox('Select Home Team', team_options)
team2_options = [team for team in team_options if team != team1]
team2 = st.sidebar.selectbox('Select Away Team', team2_options)

if st.sidebar.button('Predict Match Result'):
    # First, prepare the data for both teams with rolling averages
    def prepare_team_data(team_name, opponent_name, venue_code, hour=16, day_code=6):
        base_data = {
            'venue_code': [venue_code],
            'opp_code': [matches[matches['team'] == opponent_name]['opp_code'].iloc[0]],
            'hour': [hour],
            'day_code': [day_code],
            'goals_scored': [matches[matches['team'] == team_name]['goals_scored'].mean()],
            'goals_conceded': [matches[matches['team'] == team_name]['goals_conceded'].mean()],
            'shots_ratio': [matches[matches['team'] == team_name]['shots_ratio'].mean()]
        }
        
        # Add rolling averages
        for col in cols:
            base_data[f'{col}_rolling'] = [matches[matches['team'] == team_name][col].mean()]
        
        return pd.DataFrame(base_data)

    # Update predictors to include rolling averages
    predictors = ["venue_code", "opp_code", "hour", "day_code", "goals_scored", "goals_conceded", "shots_ratio"] + new_cols

    # Prepare data for both teams
    team1_data = prepare_team_data(team1, team2, 0)  # Home team
    team2_data = prepare_team_data(team2, team1, 1)  # Away team

    # Make predictions
    team1_pred = rf.predict(team1_data[predictors])
    team1_prob = rf.predict_proba(team1_data[predictors])[0][1]
    team2_pred = rf.predict(team2_data[predictors])
    team2_prob = rf.predict_proba(team2_data[predictors])[0][1]

    # Normalize probabilities
    total_prob = team1_prob + team2_prob
    remaining_prob = 1 - total_prob
    team1_normalized = team1_prob + (remaining_prob * (team1_prob/total_prob))
    team2_normalized = team2_prob + (remaining_prob * (team2_prob/total_prob))

    # Display results
    st.write("### Match Prediction")
    st.write(f"Probability of {team1} winning: {team1_normalized*100:.2f}%")
    st.write(f"Probability of {team2} winning: {team2_normalized*100:.2f}%")

    if team1_normalized > team2_normalized:
        st.success(f"🏆 {team1} diprediksi akan memenangkan pertandingan!")
    else:
        st.success(f"🏆 {team2} diprediksi akan memenangkan pertandingan!")

    
    # Display team statistics
    # Display team statistics
    st.write("### Overall Team Performance")
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**{team1} Stats (Overall Average)**")
        st.write(f"Goals Scored: {matches[matches['team'] == team1]['gf'].mean():.2f}")
        st.write(f"Goals Conceded: {matches[matches['team'] == team1]['ga'].mean():.2f}")
        st.write(f"Shots Ratio: {matches[matches['team'] == team1]['shots_ratio'].mean():.2f}")
        st.write(f"Venue: {matches['venue'][matches['team'] == team1].iloc[-1]}")
        st.write(f"Most Common Match Hour: {matches[matches['team'] == team1]['hour'].mode().iloc[0]}:00")
        st.write(f"Most Common Day: {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][matches[matches['team'] == team1]['day_code'].mode().iloc[0]]}")

    with col2:
        st.write(f"**{team2} Stats (Overall Average)**")
        st.write(f"Goals Scored: {matches[matches['team'] == team2]['gf'].mean():.2f}")
        st.write(f"Goals Conceded: {matches[matches['team'] == team2]['ga'].mean():.2f}")
        st.write(f"Shots Ratio: {matches[matches['team'] == team2]['shots_ratio'].mean():.2f}")
        st.write(f"Venue: {matches['venue'][matches['team'] == team2].iloc[-1]}")
        st.write(f"Most Common Match Hour: {matches[matches['team'] == team2]['hour'].mode().iloc[0]}:00")
        st.write(f"Most Common Day: {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][matches[matches['team'] == team2]['day_code'].mode().iloc[0]]}")

        
    st.write("*Opponent Code tinggi menunjukkan tim ini sering berhadapan dengan tim lawan yang memiliki performa historis dan kemenangan yang lebih baik")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))

    # Define key performance metrics
# Define key performance metrics
    stats = ['Goals Scored', 'Goals Conceded', 'Shots Ratio', 'Match Hour', 'Match Day', 'Opponent Code']
    team1_stats = [
        float(matches[matches['team'] == team1]['gf'].mean()),
        float(matches[matches['team'] == team1]['ga'].mean()),
        float(matches[matches['team'] == team1]['shots_ratio'].mean()),
        float(matches[matches['team'] == team1]['hour'].mode().iloc[0]),
        float(matches[matches['team'] == team1]['day_code'].mode().iloc[0]),
        float(team1_data['opp_code'].iloc[0])
    ]

    team2_stats = [
        float(matches[matches['team'] == team2]['gf'].mean()),
        float(matches[matches['team'] == team2]['ga'].mean()),
        float(matches[matches['team'] == team2]['shots_ratio'].mean()),
        float(matches[matches['team'] == team2]['hour'].mode().iloc[0]),
        float(matches[matches['team'] == team2]['day_code'].mode().iloc[0]),
        float(team2_data['opp_code'].iloc[0])
    ]

    x = np.arange(len(stats))
    width = 0.35

    # Create styled bars
    bars1 = ax.bar(x - width/2, team1_stats, width, label=team1, color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, team2_stats, width, label=team2, color='#3498db', alpha=0.8)

    # Style the plot
    ax.set_ylabel('Performance Values', fontsize=12)
    ax.set_title('Team Performance Comparison', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(stats, rotation=30, ha='right')
    ax.legend(loc='upper right')

    # Add value labels with improved positioning
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom',
                    fontsize=10)

    # Enhance grid and layout
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Display in Streamlit
    st.pyplot(fig)


    # Calculate team form (last 5 matches)
    def get_team_form(team_name):
        recent_matches = matches[matches['team'] == team_name].sort_values('date').tail(5)
        wins = sum(recent_matches['result'] == 'W')
        draws = sum(recent_matches['result'] == 'D')
        losses = sum(recent_matches['result'] == 'L')
        return wins, draws, losses

    # Get form for both teams
    team1_wins, team1_draws, team1_losses = get_team_form(team1)
    team2_wins, team2_draws, team2_losses = get_team_form(team2)

    # Display team form
    st.write("### Recent Team Form (Last 5 Matches)")
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**{team1}**")
        st.write(f"Wins: {team1_wins} | Draws: {team1_draws} | Losses: {team1_losses}")
        form_score1 = (team1_wins * 3 + team1_draws) / 15 * 100
        st.progress(form_score1/100)
        st.write(f"Form: {form_score1:.1f}%")

    with col2:
        st.write(f"**{team2}**")
        st.write(f"Wins: {team2_wins} | Draws: {team2_draws} | Losses: {team2_losses}")
        form_score2 = (team2_wins * 3 + team2_draws) / 15 * 100
        st.progress(form_score2/100)
        st.write(f"Form: {form_score2:.1f}%")

    # Add confidence message based on model accuracy
    st.write("### Prediction Confidence")
    result_counts = merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 0)]["actual_x"].value_counts()

    total_actual = result_counts.sum()

    # Menghitung jumlah hasil yang diprediksi 1 dan sebenarnya 1
    predicted_correct = result_counts.get(1, 0)  # Menggunakan get untuk menghindari KeyError jika 1 tidak ada

    # Menghitung proporsi
    proportion = predicted_correct / total_actual * 100 if total_actual > 0 else 0
   
    if proportion >= 75:
        st.success(f"High confidence prediction ({proportion:.1f}% accuracy)")
    elif proportion >= 60:
        st.info(f"Moderate confidence prediction ({proportion:.1f}% accuracy)")
    else:
        st.warning(f"Low confidence prediction ({proportion:.1f}% accuracy)")

    # Add head-to-head analysis
    st.write("### Prediction Analysis")
    if abs(team1_normalized - team2_normalized) > 0.2:
        st.write("Strong prediction: Clear favorite identified")
    elif abs(team1_normalized - team2_normalized) > 0.1:
        st.write("Moderate prediction: Slight advantage to favorite")
    else:
        st.write("Close prediction: Match could go either way")
