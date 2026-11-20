"""Shared code for the scale-hint study. See README.md."""

# Answer units VSI-Bench uses for each numeric task.
TASKS = {
    "object_abs_distance": "m",
    "object_size_estimation": "cm",
    "room_size_estimation": "m2",
}

ROOT_DEFAULTS = {
    "questions": "data/questions.jsonl",
    "split": "splits/scenes.json",
    "frames": "cache/frames",
    "teacher": "cache/teacher_scale.json",
    "learned": "cache/learned_scale.json",
    "results": "results",
}
