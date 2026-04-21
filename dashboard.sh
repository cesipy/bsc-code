#!/usr/bin/env bash
source venv310/bin/activate
optuna-dashboard sqlite:///res/experiments/multi_task_optim.db --port 11111