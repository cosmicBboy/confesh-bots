# Dreambot API

This is API assumes that the developer has a text data source on which
you want to "interpret" dreams. The way `dreambot` does this is by

1. Grabbing dream interpretations from dreammoods.com, where a document
   is an interpretation entry for a particular dream symbol, e.g.
   **Abyss**: _To dream of an abyss signifies an obstacle that is creating
   much anxiety for you._
2. Training a Word2Vec model on the dream corpus
3. Generating a dream interpretation through a query-matching heuristic:
   - Tokenize and preprocess query `"Yesterday I dreamt that I met a ghost in the abyss" ->
     {"yesterday", "i", "met", "a", "ghost", "in", "the", "abyss"}`, which
     we call a query token list.
   - Subset the dream corpus including only matches in the dream symbol
     and the query token list.
   - Suppose {"ghost", "abyss"} are the set of dream symbols. Rank the
     sentences by `distance(query_token_list, candidate_vector)`, where
     `distance` is the [cosine similarity function](https://en.wikipedia.org/wiki/Cosine_similarity),
     and `query_token_vector` is the dream you want to interpret
     and `candidate_vector` is one of the sentences in in the dream corpus
     subset.
   - Pick the highest `n` ranking sentences and concatenate them
     together to create a `dream_interpretation`.
