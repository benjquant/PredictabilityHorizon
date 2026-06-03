from click.testing import CliRunner

from predictability_horizon.runners.cli import main


def test_cli_lists_systems():
    res = CliRunner().invoke(main, ["systems"])
    assert res.exit_code == 0
    assert "pendulum" in res.output and "acrobot" in res.output and "cartpole" in res.output
