from memory.memory import get_topics


def progress_agent():

    topics = get_topics()

    if not topics:
        return "No topics completed yet."

    result = "Completed Topics:\n\n"

    for topic in topics:
        result += f"✓ {topic}\n"

    return result