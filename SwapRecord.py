class SwapRecord:
    def __init__(self) -> None:
        self.pair = [-1, -1]
        self.points = []
        self.energy = 0
        self.edge = -1

class Stack:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return len(self.items) == 0

    def push(self, item):
        self.items.append(item)
        self.items.sort(key=lambda x: x.energy, reverse=False)  # 根据energy值进行排序

    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        else:
            print("Stack is empty")

    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        else:
            print("Stack is empty")