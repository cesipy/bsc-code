import pytest

import scripts.autoresearch_full_training_devicefix_repair1_controller as controller


def _session(trial_limit, **extra):
    return {
        "pilot_max_analysis_trials": trial_limit,
        "pilot_execution_wall_seconds": 604800,
        "weekly_usage_stop_used_percent": 50,
        "weekly_snapshot_max_age_seconds": 120,
        **extra,
    }


def _used_ids():
    return {f"prior-{index}" for index in range(48)}


def _jobs():
    return [{"trial_id": trial_id} for trial_id in controller.EXPECTED_JOBS]


def test_explicit_zero_remaining_reserve_accepts_exact_two_trial_cap50():
    controller._check_trial_budget(
        _session(50, repair_reserve_remaining=0),
        _jobs(),
        _used_ids(),
    )


def test_default_reserve_stays_one_trial_and_only_explicit_zero_consumes_it():
    with pytest.raises(controller.ControllerError, match="queued jobs plus remaining repair reserve"):
        controller._check_trial_budget(_session(50), _jobs(), _used_ids())
    controller._check_trial_budget(_session(51), _jobs(), _used_ids())
    with pytest.raises(controller.ControllerError, match="queued jobs plus remaining repair reserve"):
        controller._check_trial_budget(_session(49, repair_reserve_remaining=0), _jobs(), _used_ids())
    with pytest.raises(controller.ControllerError, match="must be 0 or 1"):
        controller._check_trial_budget(_session(51, repair_reserve_remaining=2), _jobs(), _used_ids())
