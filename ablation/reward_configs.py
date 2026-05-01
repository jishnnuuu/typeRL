REWARD_CONFIGS = {
    "delta_only": {
        "delta_skill": 1.0,
        "accuracy": 0.0,
        "weak_avg": 0.0,
        "timer_penalty": 0.0,
        "std_penalty": 0.0,
    },
    "accuracy_only": {
        "delta_skill": 0.0,
        "accuracy": 1.0,
        "weak_avg": 0.0,
        "timer_penalty": 0.0,
        "std_penalty": 0.0,
    },
    "weak_avg_only": {
        "delta_skill": 0.0,
        "accuracy": 0.0,
        "weak_avg": 1.0,
        "timer_penalty": 0.0,
        "std_penalty": 0.0,
    },
    "timer_only": {
        "delta_skill": 0.0,
        "accuracy": 0.0,
        "weak_avg": 0.0,
        "timer_penalty": 1.0,
        "std_penalty": 0.0,   
    },
    "std_only": {
        "delta_skill": 0.0,
        "accuracy": 0.0,
        "weak_avg": 0.0,
        "timer_penalty": 0.0,
        "std_penalty": 1.0,   
    },
    "full": {
        "delta_skill": 4.33,
        "accuracy": 0.04,
        "weak_avg": 1.25,     
        "timer_penalty": 0.6,
        "std_penalty": 0.4,
    },
}