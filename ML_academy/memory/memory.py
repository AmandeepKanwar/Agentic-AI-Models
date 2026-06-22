import json


MEMORY_FILE = "memory/student.json"


def load_memory():

    with open(MEMORY_FILE, "r") as file:
        return json.load(file)


def save_topic(topic):

    data = load_memory()

    if topic not in data["completed_topics"]:
        data["completed_topics"].append(topic)

    with open(MEMORY_FILE, "w") as file:
        json.dump(data, file, indent=4)

def get_topics():

    data = load_memory()

    return data["completed_topics"]