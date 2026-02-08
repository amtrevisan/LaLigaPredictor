import logging
import os
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import soccerdata as sd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
load_dotenv()

# Suppress verbose logging from dependencies
logging.getLogger("soccerdata").setLevel(logging.ERROR)
logging.getLogger("tls_requests").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# Ill add brazil league later, want to do champions and WC but they are diffent, ill see...
LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
]


class MultiLeaguePredictor:
    def __init__(self):
        self.all_matches_df = None
        self.h2h_stats = defaultdict(
            lambda: defaultdict(
                lambda: {
                    "matches": 0,
                    "home_wins": 0,
                    "away_wins": 0,
                    "draws": 0,
                    "home_gf": 0,
                    "home_ga": 0,
                }
            )
        )
        self.ref_stats = defaultdict(
            lambda: defaultdict(
                lambda: {
                    "matches": 0,
                    "home_wins": 0,
                    "draws": 0,
                    "away_wins": 0,
                }
            )
        )

    # Later add dynamically adding seasons so i dont need to manually adjust when a season ends and starts
    def fetch_data(self, seasons=None):
        if seasons is None:
            seasons = [
                "1718",
                "1819",
                "1920",
                "2021",
                "2122",
                "2223",
                "2324",
                "2425",
            ]

        all_matches = []

        # Try different data sources in case FBref is blocked
        data_sources = [
            ("FBref", lambda: sd.FBref(leagues=LEAGUES, seasons=seasons)),
            ("ClubElo", lambda: sd.ClubElo(leagues=LEAGUES, seasons=seasons)),
        ]

        for source_name, source_factory in data_sources:
            try:
                print(f"Trying {source_name}...")
                data_source = source_factory()
                schedule = data_source.read_schedule()
                
                if schedule is not None and not schedule.empty:
                    schedule = schedule.reset_index()

                    # league name mapping
                    for idx, row in schedule.iterrows():
                        if pd.isna(row["Home"]) or pd.isna(row["Away"]):
                            continue

                        match_data = {
                            "league": row["league"],
                            "season": row["season"],
                            "date": pd.to_datetime(row["Date"]) if "Date" in row else None,
                            "matchday": row.get("Matchday", np.nan),
                            "home_team": row["Home"],
                            "away_team": row["Away"],
                            "home_score": row.get("HomeGoals", np.nan),
                            "away_score": row.get("AwayGoals", np.nan),
                        }

                        if pd.notna(row.get("HomeGoals")) and pd.notna(row.get("AwayGoals")):
                            if row["HomeGoals"] > row["AwayGoals"]:
                                match_data["result"] = "HOME_TEAM"
                            elif row["HomeGoals"] < row["AwayGoals"]:
                                match_data["result"] = "AWAY_TEAM"
                            else:
                                match_data["result"] = "DRAW"
                        else:
                            match_data["result"] = None

                        all_matches.append(match_data)

                    if all_matches:
                        self.all_matches_df = pd.DataFrame(all_matches).sort_values(
                            "date", na_position="last"
                        )
                        self.all_matches_df = self.all_matches_df.reset_index(drop=True)
                        print(f"Successfully loaded data from {source_name}")
                        return self

            except Exception as e:
                print(f"Failed to load from {source_name}: {e}")
                continue

        # If all sources fail, create empty DataFrame
        print("No data sources available, creating empty DataFrame")
        self.all_matches_df = pd.DataFrame()
        return self

    def calculate_stats(self):
        if self.all_matches_df is None or len(self.all_matches_df) == 0:
            print("No data to calculate stats")
            return self

        stats = defaultdict(
            lambda: {
                "matches": 0,
                "points": 0,
                "gf": 0,
                "ga": 0,
                "home_matches": 0,
                "home_points": 0,
                "away_matches": 0,
                "away_points": 0,
                "form": [],
                "elo": 1500,
                "goals_last_3": [],
                "conceded_last_3": [],
                "goals_last_5": [],
                "conceded_last_5": [],
                "goals_last_10": [],
                "conceded_last_10": [],
            }
        )

        cols = [
            "home_ppg",
            "away_ppg",
            "home_gd",
            "away_gd",
            "home_home_ppg",
            "away_away_ppg",
            "home_form",
            "away_form",
            "home_elo",
            "away_elo",
            "home_attack_strength",
            "away_attack_strength",
            "home_defense_strength",
            "away_defense_strength",
            "home_form_3",
            "away_form_3",
            "home_form_5",
            "away_form_5",
            "home_goals_3",
            "away_goals_3",
            "home_conceded_3",
            "away_conceded_3",
            "home_goals_5",
            "away_goals_5",
            "home_conceded_5",
            "away_conceded_5",
            "home_goals_10",
            "away_goals_10",
            "home_conceded_10",
            "away_conceded_10",
            "home_league_position",
            "away_league_position",
            "h2h_home_win_rate",
            "h2h_home_goals_diff",
        ]
        for col in cols:
            self.all_matches_df[col] = np.nan

        league_avg_goals = 1.4

        for idx, row in self.all_matches_df.iterrows():
            home_team = row["home_team"]
            away_team = row["away_team"]
            h, a = stats[home_team], stats[away_team]

            self.all_matches_df.at[idx, "home_ppg"] = h["points"] / max(1, h["matches"])
            self.all_matches_df.at[idx, "away_ppg"] = a["points"] / max(1, a["matches"])
            self.all_matches_df.at[idx, "home_gd"] = (h["gf"] - h["ga"]) / max(
                1, h["matches"]
            )
            self.all_matches_df.at[idx, "away_gd"] = (a["gf"] - a["ga"]) / max(
                1, a["matches"]
            )
            self.all_matches_df.at[idx, "home_home_ppg"] = h["home_points"] / max(
                1, h["home_matches"]
            )
            self.all_matches_df.at[idx, "away_away_ppg"] = a["away_points"] / max(
                1, a["away_matches"]
            )

            self.all_matches_df.at[idx, "home_form"] = (
                np.mean(h["form"][-5:]) if h["form"] else 1.0
            )
            self.all_matches_df.at[idx, "away_form"] = (
                np.mean(a["form"][-5:]) if a["form"] else 1.0
            )

            self.all_matches_df.at[idx, "home_elo"] = h["elo"]
            self.all_matches_df.at[idx, "away_elo"] = a["elo"]

            home_attack = (h["gf"] / max(1, h["matches"])) / league_avg_goals
            away_attack = (a["gf"] / max(1, a["matches"])) / league_avg_goals
            home_defense = (h["ga"] / max(1, h["matches"])) / league_avg_goals
            away_defense = (a["ga"] / max(1, a["matches"])) / league_avg_goals

            self.all_matches_df.at[idx, "home_attack_strength"] = (
                home_attack if h["matches"] > 0 else 1.0
            )
            self.all_matches_df.at[idx, "away_attack_strength"] = (
                away_attack if a["matches"] > 0 else 1.0
            )
            self.all_matches_df.at[idx, "home_defense_strength"] = (
                home_defense if h["matches"] > 0 else 1.0
            )
            self.all_matches_df.at[idx, "away_defense_strength"] = (
                away_defense if a["matches"] > 0 else 1.0
            )

            self.all_matches_df.at[idx, "home_form_3"] = (
                np.mean(h["form"][-3:]) if len(h["form"]) >= 3 else 1.0
            )
            self.all_matches_df.at[idx, "away_form_3"] = (
                np.mean(a["form"][-3:]) if len(a["form"]) >= 3 else 1.0
            )
            self.all_matches_df.at[idx, "home_form_5"] = (
                np.mean(h["form"][-5:]) if len(h["form"]) >= 5 else 1.0
            )
            self.all_matches_df.at[idx, "away_form_5"] = (
                np.mean(a["form"][-5:]) if len(a["form"]) >= 5 else 1.0
            )

            self.all_matches_df.at[idx, "home_goals_3"] = (
                np.mean(h["goals_last_3"]) if h["goals_last_3"] else 1.2
            )
            self.all_matches_df.at[idx, "away_goals_3"] = (
                np.mean(a["goals_last_3"]) if a["goals_last_3"] else 1.2
            )
            self.all_matches_df.at[idx, "home_conceded_3"] = (
                np.mean(h["conceded_last_3"]) if h["conceded_last_3"] else 1.2
            )
            self.all_matches_df.at[idx, "away_conceded_3"] = (
                np.mean(a["conceded_last_3"]) if a["conceded_last_3"] else 1.2
            )

            self.all_matches_df.at[idx, "home_goals_5"] = (
                np.mean(h["goals_last_5"]) if h["goals_last_5"] else 1.2
            )
            self.all_matches_df.at[idx, "away_goals_5"] = (
                np.mean(a["goals_last_5"]) if a["goals_last_5"] else 1.2
            )
            self.all_matches_df.at[idx, "home_conceded_5"] = (
                np.mean(h["conceded_last_5"]) if h["conceded_last_5"] else 1.2
            )
            self.all_matches_df.at[idx, "away_conceded_5"] = (
                np.mean(a["conceded_last_5"]) if a["conceded_last_5"] else 1.2
            )

            self.all_matches_df.at[idx, "home_goals_10"] = (
                np.mean(h["goals_last_10"]) if h["goals_last_10"] else 1.2
            )
            self.all_matches_df.at[idx, "away_goals_10"] = (
                np.mean(a["goals_last_10"]) if a["goals_last_10"] else 1.2
            )
            self.all_matches_df.at[idx, "home_conceded_10"] = (
                np.mean(h["conceded_last_10"]) if h["conceded_last_10"] else 1.2
            )
            self.all_matches_df.at[idx, "away_conceded_10"] = (
                np.mean(a["conceded_last_10"]) if a["conceded_last_10"] else 1.2
            )

            matchday_stats = []
            for team_name, team_stat in stats.items():
                if team_stat["matches"] > 0:
                    matchday_stats.append(
                        (
                            team_name,
                            team_stat["points"],
                            team_stat["gf"] - team_stat["ga"],
                        )
                    )

            matchday_stats.sort(key=lambda x: (-x[1], -x[2]))
            position_map = {
                team_name: idx + 1
                for idx, (team_name, _, _) in enumerate(matchday_stats)
            }

            self.all_matches_df.at[idx, "home_league_position"] = position_map.get(
                home_team, 20
            )
            self.all_matches_df.at[idx, "away_league_position"] = position_map.get(
                away_team, 20
            )

            h2h = self.h2h_stats[home_team][away_team]
            if h2h["matches"] > 0:
                total_points = h2h["home_wins"] * 3 + h2h["draws"]
                self.all_matches_df.at[idx, "h2h_home_win_rate"] = total_points / (
                    h2h["matches"] * 3
                )
                self.all_matches_df.at[idx, "h2h_home_goals_diff"] = (
                    h2h["home_gf"] - h2h["home_ga"]
                ) / h2h["matches"]
            else:
                self.all_matches_df.at[idx, "h2h_home_win_rate"] = 0.5
                self.all_matches_df.at[idx, "h2h_home_goals_diff"] = 0.0

            if pd.notna(row["result"]):
                h["matches"] += 1
                a["matches"] += 1
                h["home_matches"] += 1
                a["away_matches"] += 1

                home_score = row["home_score"]
                away_score = row["away_score"]

                if pd.notna(home_score):
                    h["gf"] += home_score
                    h["ga"] += away_score
                    a["gf"] += away_score
                    a["ga"] += home_score

                    h["goals_last_3"].append(home_score)
                    h["conceded_last_3"].append(away_score)
                    a["goals_last_3"].append(away_score)
                    a["conceded_last_3"].append(home_score)
                    if len(h["goals_last_3"]) > 3:
                        h["goals_last_3"] = h["goals_last_3"][-3:]
                        h["conceded_last_3"] = h["conceded_last_3"][-3:]
                    if len(a["goals_last_3"]) > 3:
                        a["goals_last_3"] = a["goals_last_3"][-3:]
                        a["conceded_last_3"] = a["conceded_last_3"][-3:]

                    h["goals_last_5"].append(home_score)
                    h["conceded_last_5"].append(away_score)
                    a["goals_last_5"].append(away_score)
                    a["conceded_last_5"].append(home_score)
                    if len(h["goals_last_5"]) > 5:
                        h["goals_last_5"] = h["goals_last_5"][-5:]
                        h["conceded_last_5"] = h["conceded_last_5"][-5:]
                    if len(a["goals_last_5"]) > 5:
                        a["goals_last_5"] = a["goals_last_5"][-5:]
                        a["conceded_last_5"] = a["conceded_last_5"][-5:]

                    h["goals_last_10"].append(home_score)
                    h["conceded_last_10"].append(away_score)
                    a["goals_last_10"].append(away_score)
                    a["conceded_last_10"].append(home_score)
                    if len(h["goals_last_10"]) > 10:
                        h["goals_last_10"] = h["goals_last_10"][-10:]
                        h["conceded_last_10"] = h["conceded_last_10"][-10:]
                    if len(a["goals_last_10"]) > 10:
                        a["goals_last_10"] = a["goals_last_10"][-10:]
                        a["conceded_last_10"] = a["conceded_last_10"][-10:]

                h2h["matches"] += 1
                h2h["home_gf"] += home_score
                h2h["home_ga"] += away_score
                if row["result"] == "HOME_TEAM":
                    h2h["home_wins"] += 1
                elif row["result"] == "AWAY_TEAM":
                    h2h["away_wins"] += 1
                else:
                    h2h["draws"] += 1

                expected_home = 1 / (1 + 10 ** ((a["elo"] - h["elo"] - 100) / 400))

                if row["result"] == "HOME_TEAM":
                    actual_home = 1.0
                    h["points"] += 3
                    h["home_points"] += 3
                    h["form"].append(3)
                    a["form"].append(0)
                elif row["result"] == "AWAY_TEAM":
                    actual_home = 0.0
                    a["points"] += 3
                    a["away_points"] += 3
                    h["form"].append(0)
                    a["form"].append(3)
                else:
                    actual_home = 0.5
                    h["points"] += 1
                    a["points"] += 1
                    h["home_points"] += 1
                    a["away_points"] += 1
                    h["form"].append(1)
                    a["form"].append(1)

                K = 32
                h["elo"] = h["elo"] + K * (actual_home - expected_home)
                a["elo"] = a["elo"] + K * ((1 - actual_home) - (1 - expected_home))

                if len(h["form"]) > 10:
                    h["form"] = h["form"][-10:]
                if len(a["form"]) > 10:
                    a["form"] = a["form"][-10:]

        return self

    def create_features(self, df):
        X = pd.DataFrame()

        X["ppg_diff"] = df["home_ppg"].fillna(1.5) - df["away_ppg"].fillna(1.5)
        X["gd_diff"] = df["home_gd"].fillna(0) - df["away_gd"].fillna(0)
        X["home_advantage"] = df["home_home_ppg"].fillna(1.5) - df[
            "away_away_ppg"
        ].fillna(1.5)
        X["elo_diff"] = df["home_elo"].fillna(1500) - df["away_elo"].fillna(1500)

        X["xg_home"] = df["home_attack_strength"].fillna(1.0) * df[
            "away_defense_strength"
        ].fillna(1.0)
        X["xg_away"] = df["away_attack_strength"].fillna(1.0) * df[
            "home_defense_strength"
        ].fillna(1.0)
        X["xg_diff"] = X["xg_home"] - X["xg_away"]

        X["form_3_diff"] = df["home_form_3"].fillna(1.0) - df["away_form_3"].fillna(1.0)
        X["form_5_diff"] = df["home_form_5"].fillna(1.0) - df["away_form_5"].fillna(1.0)

        X["goals_3_diff"] = df["home_goals_3"].fillna(1.2) - df["away_goals_3"].fillna(
            1.2
        )
        X["goals_5_diff"] = df["home_goals_5"].fillna(1.2) - df["away_goals_5"].fillna(
            1.2
        )
        X["goals_10_diff"] = df["home_goals_10"].fillna(1.2) - df[
            "away_goals_10"
        ].fillna(1.2)

        X["conceded_3_diff"] = df["away_conceded_3"].fillna(1.2) - df[
            "home_conceded_3"
        ].fillna(1.2)
        X["conceded_5_diff"] = df["away_conceded_5"].fillna(1.2) - df[
            "home_conceded_5"
        ].fillna(1.2)
        X["conceded_10_diff"] = df["away_conceded_10"].fillna(1.2) - df[
            "home_conceded_10"
        ].fillna(1.2)

        X["position_diff"] = df["away_league_position"].fillna(20) - df[
            "home_league_position"
        ].fillna(20)

        X["home_attack_vs_away_defense"] = df["home_goals_5"].fillna(1.2) - df[
            "away_conceded_5"
        ].fillna(1.2)
        X["away_attack_vs_home_defense"] = df["away_goals_5"].fillna(1.2) - df[
            "home_conceded_5"
        ].fillna(1.2)

        X["h2h_home_win_rate"] = df["h2h_home_win_rate"].fillna(0.5)
        X["h2h_home_goals_diff"] = df["h2h_home_goals_diff"].fillna(0.0)

        X["strength"] = (
            X["ppg_diff"] * 0.20
            + X["home_advantage"] * 0.16
            + X["elo_diff"] * 0.01
            + X["xg_diff"] * 0.15
            + X["form_3_diff"] * 0.08
            + X["form_5_diff"] * 0.07
            + X["goals_5_diff"] * 0.06
            + X["conceded_5_diff"] * 0.06
            + X["position_diff"] * 0.08
            + X["home_attack_vs_away_defense"] * 0.05
            + X["away_attack_vs_home_defense"] * 0.04
            + X["h2h_home_win_rate"] * 0.03
        )

        return X

    def test(self, train_seasons=None, test_season="2425", test_matchdays=None):
        if train_seasons is None:
            train_seasons = ["1718", "1819", "1920", "2021", "2122", "2223", "2324"]

        # Check if data was loaded successfully
        if self.all_matches_df is None:
            print("Error: No data available. Check fetch_data() for errors.")
            return {}

        target_map = {"HOME_TEAM": 1, "AWAY_TEAM": 0, "DRAW": 2}
        overall_accuracy = {}

        for league_name in LEAGUES:
            print(f"\n{league_name}")
            print("-" * 40)

            league_train_df = self.all_matches_df[
                (self.all_matches_df["league"] == league_name)
                & (self.all_matches_df["season"].isin(train_seasons))
                & (self.all_matches_df["result"].notna())
            ]

            league_test_df = self.all_matches_df[
                (self.all_matches_df["league"] == league_name)
                & (self.all_matches_df["season"] == test_season)
                & (self.all_matches_df["result"].notna())
            ]

            if test_matchdays:
                league_test_df = league_test_df[
                    league_test_df["matchday"].isin(test_matchdays)
                ]

            if len(league_train_df) == 0 or len(league_test_df) == 0:
                print(
                    f"Insufficient data (train: {len(league_train_df)}, test: {len(league_test_df)})"
                )
                continue

            X_train = self.create_features(league_train_df)
            y_train = league_train_df["result"].map(target_map)
            X_test = self.create_features(league_test_df)
            y_test = league_test_df["result"].map(target_map)

            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                random_state=42,
                class_weight="balanced",
            )
            model.fit(X_train, y_train)

            train_acc = model.score(X_train, y_train) * 100
            test_acc = model.score(X_test, y_test) * 100

            overall_accuracy[league_name] = test_acc

            print(f"  Training Accuracy: {train_acc:.1f}%")
            print(f"  Test Accuracy:     {test_acc:.1f}%")
            print(f"  Test Samples:      {len(league_test_df)}")

            # Display predictions
            print(f"\n  {league_name} Predictions (10 games max):")
            predictions = model.predict(X_test)
            reverse_map = {0: "AWAY_WIN", 1: "HOME_WIN", 2: "DRAW"}

            for idx, (i, row) in enumerate(league_test_df.iterrows()):
                if idx >= 10:
                    break
                pred = reverse_map[predictions[idx]]
                actual = reverse_map[y_test.iloc[idx]]
                correct = "✓" if predictions[idx] == y_test.iloc[idx] else "✗"
                score_str = f"{int(row['home_score'])}-{int(row['away_score'])}"
                print(
                    f"    {correct} {row['home_team']:20s} vs {row['away_team']:20s} "
                    f"| Pred: {pred:10s} | Actual: {actual:10s} | Score: {score_str}"
                )

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY - TEST ACCURACY BY LEAGUE")
        print("=" * 80)
        for league_name, acc in sorted(
            overall_accuracy.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {league_name:30s}: {acc:.1f}%")

        if overall_accuracy:
            avg_acc = np.mean(list(overall_accuracy.values()))
            print(f"\n  Average Accuracy:              {avg_acc:.1f}%")

        print("=" * 80 + "\n")

        return overall_accuracy


if __name__ == "__main__":
    predictor = MultiLeaguePredictor()
    predictor.fetch_data()
    predictor.calculate_stats()
    predictor.test(test_matchdays=list(range(1, 16)))

