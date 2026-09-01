class HouseholdHRL:
    def __init__(self):
        self.high_level_tasks = ['CleanKitchen', 'VacuumLivingRoom']

    def subtask_clean_kitchen(self):
        actions = ['navigate_sink', 'wash_dishes', 'wipe_counters']
        return [f"Execute: {a}" for a in actions]

    def subtask_vacuum(self):
        actions = ['navigate_room', 'power_on_vacuum', 'dock_station']
        return [f"Execute: {a}" for a in actions]

    def execute_hierarchy(self, task_idx):
        if task_idx == 0:
            return self.subtask_clean_kitchen()
        return self.subtask_vacuum()

hrl = HouseholdHRL()
print("MAXQ Hierarchy Task Plan:", hrl.execute_hierarchy(0))