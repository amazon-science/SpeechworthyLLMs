# from copilot

from collections import defaultdict
import math

def expected_score(rating1, rating2):
    return 1 / (1 + 10 ** ((rating2 - rating1) / 400))

def update_elo(rating, expected, actual, k=32):
    return rating + k * (actual - expected)

def calculate_elo(match_up_list):
    # Initialize Elo ratings for each model
    elo_ratings = defaultdict(lambda: 1200)

    for match_up in match_up_list:
        model0 = match_up[0]
        model1 = match_up[1]

        # Calculate expected scores
        expected0 = expected_score(elo_ratings[model0], elo_ratings[model1])
        expected1 = expected_score(elo_ratings[model1], elo_ratings[model0])

        # Assume model0 won the match
        actual0 = 1
        actual1 = 0

        # Update Elo ratings
        elo_ratings[model0] = update_elo(elo_ratings[model0], expected0, actual0)
        elo_ratings[model1] = update_elo(elo_ratings[model1], expected1, actual1)

    return elo_ratings

# Example usage
match_up_list = [
    ("ppo_response", "base_response"),
    ("ppo_response", "icl_prompted_response"),
    ("ppo_response", "dpo_response"),
    ("icl_prompted_response", "base_response"),
    ("icl_prompted_response", "base_prompted_response"),
    ("icl_prompted_response", "dpo_response"),
    ("base_prompted_response", "base_response"),
    ("dpo_response", "base_response"),
    ("ppo_response", "ref_response"),
    ("icl_prompted_response", "ref_response"),
]

elo_ratings = calculate_elo(match_up_list)
print(elo_ratings)