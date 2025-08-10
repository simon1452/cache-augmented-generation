from typing import List, Dict

class DialogManager:
    def __init__(self):
        self.history: List[Dict[str, str]] = []

    def add_turn(self, user: str, assistant: str):
        self.history.append({"user": user, "assistant": assistant})

    def get_history(self) -> List[Dict[str, str]]:
        return self.history