"""
Multi-Query Retriever with Project-Name Boosting
=================================================
1. Extracts project name from user query (regex, zero-cost)
2. Generates 3 diverse search queries via a fast Mistral call
3. Searches all variants against pgvector, deduplicates
4. Boosts chunks whose source/content match the detected project
5. Falls back to keyword expansion if LLM call fails
"""

import re
from typing import Tuple, List
from kb.vector_store import get_vector_store
from config.settings import settings


# ── Known project names for regex detection ──────────────────────────────────
_PROJECT_NAMES = [
    "life in blue",
    "codename dear life", "dear life",
    "reneev page 22", "page 22", "page22",
    "eden", "godrej eden",
    "levvel 7", "levvel7", "level 7",
    "forever young",
    "codename cornerstone", "cornerstone",
    "dobariya green parmeshwar", "green parmeshwar", "dobariya",
    "aaryan gloria", "gloria",
    "godrej green glades", "green glades",
]

# Source file patterns that map to project names (for metadata boosting)
_PROJECT_SOURCE_MAP = {
    "life in blue":          ["life_in_blue", "Life_In_Blue"],
    "dear life":             ["dear_life", "Dear_Life", "codename_dear_life"],
    "codename dear life":    ["dear_life", "Dear_Life", "codename_dear_life"],
    "page 22":               ["page_22", "Page_22", "reneev_page_22", "Reneev_Page_22"],
    "reneev page 22":        ["page_22", "Page_22", "reneev_page_22", "Reneev_Page_22"],
    "eden":                  ["eden", "Eden", "godrej_eden"],
    "godrej eden":           ["eden", "Eden", "godrej_eden"],
    "levvel 7":              ["levvel", "Levvel", "levvel7"],
    "forever young":         ["forever_young", "Forever_Young"],
    "cornerstone":           ["cornerstone", "Cornerstone", "codename_cornerstone"],
    "codename cornerstone":  ["cornerstone", "Cornerstone", "codename_cornerstone"],
    "dobariya":              ["dobariya", "Dobariya", "green_parmeshwar"],
    "green parmeshwar":      ["dobariya", "Dobariya", "green_parmeshwar"],
    "aaryan gloria":         ["aaryan_gloria", "gloria"],
    "green glades":          ["green_glades", "godrej_green_glades"],
}


def _detect_project(query: str) -> str | None:
    """Detect a known project name in the query. Returns the canonical name or None."""
    q_lower = query.lower()
    # Check longest names first to avoid partial matches
    sorted_names = sorted(_PROJECT_NAMES, key=len, reverse=True)
    for name in sorted_names:
        if name in q_lower:
            return name
    return None


def _generate_multi_queries(query: str, project_name: str | None) -> List[str]:
    """
    Use a fast Mistral call to generate 3 diverse search queries.
    Falls back to simple keyword expansion if the LLM call fails.
    """
    try:
        from langchain_mistralai import ChatMistralAI

        llm = ChatMistralAI(
            model="mistral-small-latest",
            api_key=settings.mistral_api_key,
            temperature=0.3,
            max_tokens=250
        )

        project_context = ""
        if project_name:
            project_context = f"\nThe user is asking about the project: '{project_name}'. Include the project name in each variant."

        prompt = f"""You are a real estate search query optimizer. Given a user's question about a property/project, generate exactly 3 diverse search queries that would help retrieve relevant information from a knowledge base.

Each query should target different aspects: overview/location, pricing/configurations, and amenities/specifications.
{project_context}

User question: "{query}"

Output exactly 3 queries, one per line. No numbering, no bullets, no explanations. Just the queries."""

        response = llm.invoke(prompt)
        lines = [line.strip() for line in response.content.strip().split("\n") if line.strip()]
        # Take up to 3 valid lines
        variants = lines[:3]
        if variants:
            print(f"[Retriever] Multi-query variants: {variants}")
            return variants
    except Exception as e:
        print(f"[Retriever] Multi-query LLM failed, using fallback: {e}")

    # Fallback: simple keyword expansion
    return _keyword_expand_fallback(query, project_name)


def _keyword_expand_fallback(query: str, project_name: str | None) -> List[str]:
    """Zero-cost fallback: generate 2 query variants using keyword templates."""
    variants = []
    if project_name:
        variants.append(f"{project_name} project location configurations pricing amenities")
        variants.append(f"{project_name} residential apartments specifications possession details")
    else:
        variants.append(f"{query} project details pricing location")
    return variants


def _boost_project_chunks(docs: list, project_name: str | None) -> list:
    """
    Re-order documents to prioritize chunks that match the detected project.
    Chunks matching the project's source files or content get sorted first.
    """
    if not project_name:
        return docs

    source_patterns = _PROJECT_SOURCE_MAP.get(project_name, [project_name])
    project_lower = project_name.lower()

    def score(doc):
        s = 0
        source = doc.metadata.get("source", "").lower()
        content_lower = doc.page_content.lower()

        # Boost by source file match
        for pattern in source_patterns:
            if pattern.lower() in source:
                s += 10
                break

        # Boost by content mention
        if project_lower in content_lower:
            s += 5

        return s

    return sorted(docs, key=score, reverse=True)


def retrieve_context(query: str, kb_id: str) -> Tuple[str, List[str]]:
    """
    Multi-query retrieval with project-name boosting.
    1. Detect project name from query
    2. Generate 3 LLM-powered search variants (or fallback)
    3. Run all variants + original query through MMR retrieval
    4. Deduplicate, boost project-relevant chunks, return top 8
    """
    try:
        kb_id_clean = kb_id or "main-kb"
        if kb_id_clean in ("null", "None"):
            kb_id_clean = "main-kb"

        vectorstore = get_vector_store(kb_id_clean)

        # Step 1: Detect project name
        project_name = _detect_project(query)
        if project_name:
            print(f"[Retriever] Detected project: '{project_name}'")

        # Step 2: Generate diverse query variants
        variants = _generate_multi_queries(query, project_name)

        # Always include the original query
        queries = [query] + variants
        # Deduplicate queries
        seen_q = set()
        unique_queries = []
        for q in queries:
            q_lower = q.lower().strip()
            if q_lower not in seen_q:
                seen_q.add(q_lower)
                unique_queries.append(q)
        queries = unique_queries

        print(f"[Retriever] Final queries ({len(queries)}): {queries}")

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.6}
        )

        all_docs = []
        for q in queries:
            try:
                docs = retriever.invoke(q)
                all_docs.extend(docs)
            except Exception as e:
                print(f"[Retriever] Query failed '{q[:50]}...': {e}")

        # Step 3: Deduplicate by content
        seen = set()
        unique_docs = []
        for doc in all_docs:
            h = doc.page_content.strip()
            if h not in seen:
                seen.add(h)
                unique_docs.append(doc)

        # Step 4: Boost project-relevant chunks
        unique_docs = _boost_project_chunks(unique_docs, project_name)

        # Top 8 unique results
        unique_docs = unique_docs[:8]

        context_str = "\n\n---\n\n".join([doc.page_content for doc in unique_docs])
        sources = list(set([doc.metadata.get("source", "Unknown") for doc in unique_docs]))
        return context_str, sources
    except Exception as e:
        print(f"[Retriever] Error: {e}")
        return "", []
