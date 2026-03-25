import unittest
from mood_analyzer import MoodAnalyzer

class TestMoodAnalyzer(unittest.TestCase):

    def setUp(self):
        """Set up a new MoodAnalyzer instance for each test."""
        self.analyzer = MoodAnalyzer()

    def test_preprocess_empty_string(self):
        """Test preprocessing an empty string."""
        self.assertEqual(self.analyzer.preprocess(""), [])

    def test_preprocess_whitespace(self):
        """Test preprocessing a string with only whitespace."""
        self.assertEqual(self.analyzer.preprocess("   "), [])

    def test_preprocess_punctuation(self):
        """Test preprocessing a string with punctuation."""
        self.assertEqual(self.analyzer.preprocess("Hello, world!"), ["hello", "world"])

    def test_preprocess_contractions(self):
        """Test preprocessing a string with contractions."""
        self.assertEqual(self.analyzer.preprocess("I'm happy"), ["i'm", "happy"])

    def test_preprocess_emojis(self):
        """Test preprocessing a string with emojis."""
        self.assertEqual(self.analyzer.preprocess("I am happy :)"), ["i", "am", "happy", "happy_emoji"])

    def test_preprocess_repeated_characters(self):
        """Test preprocessing a string with repeated characters."""
        self.assertEqual(self.analyzer.preprocess("sooo good"), ["soo", "good"])

    def test_score_text_positive(self):
        """Test scoring a positive text."""
        self.assertEqual(self.analyzer.score_text("happy wonderful day"), 2)

    def test_score_text_negative(self):
        """Test scoring a negative text."""
        self.assertEqual(self.analyzer.score_text("sad terrible day"), -2)

    def test_score_text_neutral(self):
        """Test scoring a neutral text."""
        self.assertEqual(self.analyzer.score_text("this is a sentence"), 0)

    def test_score_text_mixed(self):
        """Test scoring a text with mixed signals."""
        self.assertEqual(self.analyzer.score_text("happy but sad"), 0)

    def test_score_text_negation(self):
        """Test scoring a text with negation."""
        self.assertEqual(self.analyzer.score_text("not happy"), -1)
        self.assertEqual(self.analyzer.score_text("not sad"), 1)

    def test_score_text_booster(self):
        """Test scoring a text with booster words."""
        self.assertEqual(self.analyzer.score_text("so happy"), 2)
        self.assertEqual(self.analyzer.score_text("very sad"), -2)

    def test_predict_label_positive(self):
        """Test predicting a positive label."""
        self.assertEqual(self.analyzer.predict_label("awesome day"), "positive")

    def test_predict_label_negative(self):
        """Test predicting a negative label."""
        self.assertEqual(self.analyzer.predict_label("awful day"), "negative")

    def test_predict_label_neutral(self):
        """Test predicting a neutral label."""
        self.assertEqual(self.analyzer.predict_label("this is a pen"), "neutral")

if __name__ == '__main__':
    unittest.main()
