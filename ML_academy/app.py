from agents.coding import coding_agent
from agents.teacher import teacher_agent
from agents.quiz import quiz_agent
from agents.progress import progress_agent
from agents.rag_agent import rag_agent
from graph.workflow import route_query


print("🤖 ML Academy Agent")
print("Type exit to quit")


while True:

    query = input("\nAsk: ")

    if query.lower() == "exit":
        break

    route = route_query(query)

    if route == "quiz":
        response = quiz_agent(query)

    elif route == "coding":
        response = coding_agent(query)

    elif route == "progress":
        response = progress_agent()

    elif route == "rag":
        response = rag_agent(query)

    else:
        response = teacher_agent(query)

    print("\n")
    print(response)