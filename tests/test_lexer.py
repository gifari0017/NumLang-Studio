import pytest
from numlang.lang import LanguageError, tokenize


def test_token_sequence_and_columns():
    tokens = tokenize(' 2x + SIN(.5E-2) ')
    assert [(t.kind, t.text, t.column) for t in tokens] == [
        ('number', '2', 2), ('name', 'x', 3), ('op', '+', 5),
        ('name', 'sin', 7), ('op', '(', 10), ('number', '.5e-2', 11),
        ('op', ')', 16), ('end', '', 18)]


@pytest.mark.parametrize('source', ['x@2', 'x**2$', 'x[0]', 'x=2'])
def test_illegal_character(source):
    with pytest.raises(LanguageError, match='column.*unexpected character'):
        tokenize(source)
