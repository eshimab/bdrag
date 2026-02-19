def model_spell(query):
    model_query = f"""Fix any spelling errors in this movie search query.
Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Corrected:"""
    return model_query


def model_rewrite(query):
    model_query = f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""
    return model_query


def model_expand(query):
    model_query = f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}"
"""
    return model_query


def model_rerank_indv(query, doc):
    return f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""


def model_rerank_batch(query, doc_list):

    doc_list_str = ""
    for doc in doc_list:
        title_string = f"--> Movie Title: {doc["title"]}"
        id_string = f"  --> Movie ID: {doc["id"]}"
        if len(doc["description"]) < 101:
            print(f"{title_string} DESCRIPTION TOO SHORT")
            raise ValueError("DESCRIPTION TOO SHORT")
        desc_string = f"  --> Movie Description:\n{doc["description"]}"
        total_string = "\n".join([title_string, id_string, desc_string])
        doc_list_str = doc_list_str + f"\n\n{total_string}"

    return_string = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:

{doc_list_str}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
"""
    return return_string
