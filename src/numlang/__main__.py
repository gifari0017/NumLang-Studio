"""Run NumLang interactively or execute a .nl script."""
import argparse
from pathlib import Path
import sys

from numlang.lang.interpreter import Interpreter
from numlang.lang.lexer import LanguageError


def main(argv=None):
    parser = argparse.ArgumentParser(description='NumLang Studio command interpreter')
    parser.add_argument('script', nargs='?', type=Path, help='NumLang script (.nl)')
    args = parser.parse_args(argv)
    interpreter = Interpreter()
    if args.script is not None:
        try:
            source = args.script.read_text(encoding='utf-8-sig')
            # Print completed commands even if a later source line fails.
            for i, line in enumerate(source.splitlines(), 1):
                result = interpreter.execute_line(line, i)
                if result is not None:
                    print(result.text)
            interpreter.finish()
        except (OSError, UnicodeError, LanguageError) as exc:
            print(f'Error: {exc}', file=sys.stderr)
            return 1
        return 0
    print('NumLang Studio | type help for commands, quit to exit')
    line_number = 0
    while True:
        try:
            line = input('... ' if interpreter.in_scene else 'nl> ')
        except EOFError:
            try:
                interpreter.finish()
            except LanguageError as exc:
                print(f'Error: {exc}', file=sys.stderr)
                return 1
            return 0
        except KeyboardInterrupt:
            interpreter.abort_scene()
            print('\nCancelled. Type quit to exit.')
            continue
        line_number += 1
        if line.strip().lower() in ('quit', 'exit'):
            interpreter.abort_scene()
            return 0
        try:
            result = interpreter.execute_line(line, line_number)
            if result is not None:
                print(result.text)
        except LanguageError as exc:
            print(f'Error: {exc}', file=sys.stderr)


if __name__ == '__main__':
    raise SystemExit(main())
