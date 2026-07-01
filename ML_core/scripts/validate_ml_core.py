import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE_DIRS = ["core", "agents", "pipelines", "scripts"]


def main():
    checked = []
    for code_dir in CODE_DIRS:
        for path in (ROOT / code_dir).rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"))
            checked.append(path.relative_to(ROOT).as_posix())
    print(f"syntax ok: {len(checked)} files")


if __name__ == "__main__":
    main()
