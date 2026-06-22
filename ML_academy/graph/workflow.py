def route_query(query):

    query = query.lower()

    if "quiz" in query:
        return "quiz"

    elif "progress" in query:
        return "progress"

    elif "code" in query:
        return "coding"

    elif "python" in query:
        return "coding"

    elif "what is" in query:
        return "rag"

    elif "explain" in query:
        return "rag"

    return "teacher"