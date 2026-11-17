"""Exercise the actual CLI entry point with streams and files."""
import io
import sys
from numlang.__main__ import main


def test_script_runner(tmp_path, capsys):
    p = tmp_path / 'sample.nl'
    p.write_text('compare root x^3-2x-5 in 2 3\ncompare solve [2] = [6]\n', encoding='utf-8-sig')
    assert main([str(p)]) == 0
    out = capsys.readouterr().out
    assert '2.09455148154' in out and '[3]' in out


def test_script_runner_error(tmp_path, capsys):
    p = tmp_path / 'bad.nl'
    p.write_text('root x in -1 1\nwrong\n', encoding='utf-8')
    assert main([str(p)]) == 1
    captured = capsys.readouterr()
    assert 'ROOT:' in captured.out
    assert 'line 2:' in captured.err


def test_repl_recovers_after_error(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'stdin', io.StringIO('wrong\nsolve [2] = [6]\nquit\n'))
    assert main([]) == 0
    captured = capsys.readouterr()
    assert 'line 1:' in captured.err and '[3]' in captured.out


def test_repl_multiline_scene(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'stdin', io.StringIO('scene curtain\nh=4 ms\nend\nquit\n'))
    assert main([]) == 0
    assert 'SCENE curtain' in capsys.readouterr().out


def test_missing_script(tmp_path, capsys):
    assert main([str(tmp_path / 'missing.nl')]) == 1
    assert 'Error:' in capsys.readouterr().err
