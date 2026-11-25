from collections import deque


class ConversationMemory:
    def __init__(self, max_turns=6):
        self.turns = deque(maxlen=max_turns)

    def add(self, question, answer):
        self.turns.append({"question": question, "answer": answer})

    def resolve_follow_up(self, question):
        if not self.turns:
            return question
        if any(word in question.lower() for word in ("it", "that", "this", "they", "its")):
            previous = self.turns[-1]["question"]
            return f"{previous} Follow-up: {question}"
        return question

    def transcript(self):
        return list(self.turns)
