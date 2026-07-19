from ssvep_toolkit.cli import main


def test_registry_cli_initializes_and_summarizes(tmp_path, capsys):
    database = tmp_path / "runs.sqlite3"
    assert main(["registry", "--database", str(database), "init"]) == 0
    assert database.exists()
    assert main(["registry", "--database", str(database), "summary"]) == 0
    output = capsys.readouterr().out
    assert '"studies": 0' in output
    assert '"metrics": 0' in output
