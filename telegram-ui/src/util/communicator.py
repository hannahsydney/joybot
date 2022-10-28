import random

counter = 0

def generate_response(reply: str) -> str:
    global counter

    if reply == "Start":
        return "hello"

    counter += 1

    if random.randint(0, 3) < 1:
        print()
        return "end"

    return "hello" + str(counter)
