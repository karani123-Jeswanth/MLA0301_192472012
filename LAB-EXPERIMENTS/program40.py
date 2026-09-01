import numpy as np

class AdaptiveTutorEnv:
    def __init__(self):
        # Student Mastery Levels: 0: Low, 1: Medium, 2: High
        self.student_level = 0
        self.actions = ['EasyQuiz', 'StandardLesson', 'ChallengingProject']

    def step(self, action_idx):
        # Rewards student progress while avoiding frustration
        if self.student_level == 0 and action_idx == 0:
            self.student_level = 1
            return 10, "Learning Progressed!"
        elif self.student_level == 1 and action_idx == 1:
            self.student_level = 2
            return 15, "Concept Mastered!"
        elif self.student_level == 0 and action_idx == 2:
            return -10, "Frustration (Too hard)!"
        return 2, "Maintained Practice."

tutor = AdaptiveTutorEnv()
rew, msg = tutor.step(0)
print(f"Tutor Intervention -> Reward: {rew}, Outcome: {msg}")