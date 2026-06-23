from vaderSentiment.vaderSentiment import (
    SentimentIntensityAnalyzer,
)

_analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(
    text: str,
) -> tuple[str, float]:
    if not text or len(text.strip()) < 5:
        return "neutral", 0.0

    scores = _analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return label, round(compound, 4)
