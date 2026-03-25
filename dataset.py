"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
    "wonderful",
    "proud",    # "exhausted but proud" should register
    "joy",      # "tears of joy"
    "sweet",    # half of "bitter-sweet" after hyphen split
    "sick",     # slang: "this party is sick" = amazing
    "fire",     # slang: "that song is fire" = great
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
    "exhausted",     # stronger than "tired"
    "mad",           # enables negation: "not mad" flips correctly
    "disappointed",  # "I'm just disappointed"
    "meh",           # "feeling pretty meh"
    "bitter",        # half of "bitter-sweet" after hyphen split
    "crying",        # usually signals distress
]

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
    "This is the best day ever! :D",
    "I'm so done with all this work. 😩",
    "I guess it's fine. Whatever.",
    "I'm excited for the party but also nervous.",
    "This is lowkey the best pizza I've ever had.",
    "I absolutely love being ignored. #sarcasm",
    "No cap, this new album is fire.",
    "Feeling kinda down today 🥲",
    "Just finished a marathon, I'm tired but proud.",
    "This movie is so boring I'm gonna die 💀",
]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
    "positive",  # "This is the best day ever! :D"
    "negative",  # "I'm so done with all this work. 😩"
    "neutral",   # "I guess it's fine. Whatever."
    "mixed",     # "I'm excited for the party but also nervous."
    "positive",  # "This is lowkey the best pizza I've ever had."
    "negative",  # "I absolutely love being ignored. #sarcasm"
    "positive",  # "No cap, this new album is fire."
    "negative",  # "Feeling kinda down today 🥲"
    "mixed",     # "Just finished a marathon, I'm tired but proud."
    "negative",  # "This movie is so boring I'm gonna die 💀"
]

# TODO: Add 5-10 more posts and labels.
#
# Requirements:
#   - For every new post you add to SAMPLE_POSTS, you must add one
#     matching label to TRUE_LABELS.
#   - SAMPLE_POSTS and TRUE_LABELS must always have the same length.
#   - Include a variety of language styles, such as:
#       * Slang ("lowkey", "highkey", "no cap")
#       * Emojis (":)", ":(", "🥲", "😂", "💀")
#       * Sarcasm ("I absolutely love getting stuck in traffic")
#       * Ambiguous or mixed feelings
#
# Tips:
#   - Try to create some examples that are hard to label even for you.
#   - Make a note of any examples that you and a friend might disagree on.
#     Those "edge cases" are interesting to inspect for both the rule based
#     and ML models.
#
# Example of how you might extend the lists:
#
# SAMPLE_POSTS.append("Lowkey stressed but kind of proud of myself")
# TRUE_LABELS.append("mixed")
#
# Remember to keep them aligned:
#   len(SAMPLE_POSTS) == len(TRUE_LABELS)
