# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
import string
from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

        # Add emoji keywords to the positive and negative sets
        self.positive_words.update([
            "happy_emoji", "laughing_emoji", "grinning_face_emoji", "heart_eyes_emoji"
        ])
        self.negative_words.update([
            "sad_emoji", "smiling_face_with_tear_emoji", "crying_face_emoji", "angry_face_emoji"
        ])

        # Add booster words
        self.booster_words = set(["so", "very", "really", "wicked"])

        # Sarcasm markers: presence of any of these flips the final score
        self.sarcasm_markers = set(["sarcastically", "sarcasm"])

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        This method now:
          - Normalizes repeated characters ("soooo" -> "soo")
          - Replaces common emojis with descriptive text
          - Removes punctuation (while preserving apostrophes in contractions)
          - Strips leading and trailing whitespace
          - Converts everything to lowercase
          - Splits on spaces
        """
        # Normalize repeated characters
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)

        # Split hyphenated words so "bitter-sweet" becomes "bitter sweet"
        text = text.replace('-', ' ')

        # Handle emojis by replacing them with descriptive text
        emoji_map = {
            ":)": " happy_emoji ",
            ":-)": " happy_emoji ",
            "🙂": " happy_emoji ",
            "😂": " laughing_emoji ",
            "😀": " grinning_face_emoji ",
            "😍": " heart_eyes_emoji ",
            ":(": " sad_emoji ",
            ":-(": " sad_emoji ",
            "🥲": " smiling_face_with_tear_emoji ",
            "😭": " crying_face_emoji ",
            "😠": " angry_face_emoji "
        }
        for emoji, word in emoji_map.items():
            text = text.replace(emoji, word)

        # Remove punctuation, but keep apostrophes and underscores
        punctuation_to_remove = string.punctuation.replace("'", "").replace("_", "")
        translator = str.maketrans('', '', punctuation_to_remove)
        text = text.translate(translator)

        cleaned = text.strip().lower()
        tokens = cleaned.split()

        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score.
        Negative words decrease the score.
        Handles negation and booster words.
        """
        tokens = self.preprocess(text)
        score = 0
        i = 0
        while i < len(tokens):
            token = tokens[i]
            word_score = 0
            if token in self.positive_words:
                word_score = 1
            elif token in self.negative_words:
                word_score = -1

            # Check for booster words
            if i > 0 and tokens[i-1] in self.booster_words:
                word_score *= 2

            # Check for negation within a 2-token window.
            # "not happy" (+1) → negative (-1); "not mad" (-1) → neutral (0)
            # Window of 2 handles "not very happy" and "not really tired".
            negated = any(
                tokens[i - k] == 'not'
                for k in range(1, 3)
                if i - k >= 0
            )
            if negated:
                if word_score < 0:
                    word_score = 0
                else:
                    word_score *= -1
            
            score += word_score
            i += 1

        # Flip score if sarcasm is explicitly signaled
        if any(t in self.sarcasm_markers for t in tokens):
            score *= -1

        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The default mapping is:
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"

        TODO: You can adjust this mapping if it makes sense for your model.
        For example:
          - Use different thresholds (for example score >= 2 to be "positive")
          - Add a "mixed" label for scores close to zero
        Just remember that whatever labels you return should match the labels
        you use in TRUE_LABELS in dataset.py if you care about accuracy.
        """
        tokens = self.preprocess(text)
        score = self.score_text(text)
        if score > 0:
            return "positive"
        elif score < 0:
            return "negative"
        else:
            # A score of 0 with both positive and negative signals = mixed
            has_positive = any(t in self.positive_words for t in tokens)
            has_negative = any(t in self.negative_words for t in tokens)
            if has_positive and has_negative:
                return "mixed"
            return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        tokens = self.preprocess(text)

        positive_hits: List[str] = []
        negative_hits: List[str] = []
        score = 0

        for token in tokens:
            if token in self.positive_words:
                positive_hits.append(token)
                score += 1
            if token in self.negative_words:
                negative_hits.append(token)
                score -= 1

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
