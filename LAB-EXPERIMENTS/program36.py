class MAXQCoordinator:
    def __init__(self):
        self.subtasks = {
            'Harvest': ['Agent1_Gather', 'Agent2_Gather'],
            'Deposit': ['Agent1_Store', 'Agent2_Store']
        }

    def evaluate_task(self, inventory_level):
        if inventory_level < 5:
            return "Task: Harvest -> Actions: " + ", ".join(self.subtasks['Harvest'])
        return "Task: Deposit -> Actions: " + ", ".join(self.subtasks['Deposit'])

maxq = MAXQCoordinator()
print(maxq.evaluate_task(2))
print(maxq.evaluate_task(8))