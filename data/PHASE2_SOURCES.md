# Phase-two data and scope

TweetEval’s SemEval-2017 sentiment task supplies 59,899 human-labelled posts. The original test split retains all 12,284 posts. Text-only overlap removal protects it from training contamination; ambiguous training labels and repeated training text are excluded. The upstream dataset licence is unspecified, so raw posts and error text stay local.

Use `python predict_tweets.py --data examples/black_friday_posts.csv`. These examples are authored illustrations. Reconstruct local error text by source ID after fetching the pinned TweetEval files. This project measures general historical social sentiment, rather than Black Friday public opinion.

Source URLs and immutable hashes are in the accompanying provenance JSON. Original data notices continue to apply.
