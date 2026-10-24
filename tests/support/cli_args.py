"""Execute only the real CLI's contiguous argparse block and mixing fallback.

This is parameter-block testing, not module startup or training integration.
Unknown structure fails immediately; no training imports or train() are run.
"""

import argparse
import ast
from datetime import datetime


def _is_call(node, owner, method):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == owner
        and node.func.attr == method
    )


def parse_cli_block(path):
    """Read constructor, all argument definitions, parse and the next fallback.

    The caller controls real sys.argv (pytest's monkeypatch restores it).
    Only argparse, datetime and the builtin argument types are supplied.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    main_test = ast.parse('__name__ == "__main__"', mode="eval").body
    mains = [
        node
        for node in tree.body
        if isinstance(node, ast.If) and ast.dump(node.test) == ast.dump(main_test)
    ]
    assert len(mains) == 1, f"{path}: expected one main guard"
    body = mains[0].body
    constructor = body[0]
    assert (
        isinstance(constructor, ast.Assign)
        and len(constructor.targets) == 1
        and isinstance(constructor.targets[0], ast.Name)
        and constructor.targets[0].id == "parser"
        and _is_call(constructor.value, "argparse", "ArgumentParser")
    ), f"{path}: expected parser constructor first"
    parse_indices = [
        i
        for i, node in enumerate(body)
        if isinstance(node, ast.Assign) and _is_call(node.value, "parser", "parse_args")
    ]
    assert len(parse_indices) == 1, f"{path}: expected one parse_args assignment"
    parse_index = parse_indices[0]
    parse_node = body[parse_index]
    assert (
        len(parse_node.targets) == 1
        and isinstance(parse_node.targets[0], ast.Name)
        and parse_node.targets[0].id == "args"
        and not parse_node.value.args
        and not parse_node.value.keywords
    ), f"{path}: expected args = parser.parse_args()"
    definitions = body[1:parse_index]
    assert definitions and all(
        isinstance(node, ast.Expr) and _is_call(node.value, "parser", "add_argument")
        for node in definitions
    ), f"{path}: unexpected statement in argument definitions"
    assert parse_index + 1 < len(body), f"{path}: missing mixing fallback"
    fallback = body[parse_index + 1]
    target = ast.parse("args.cg_mixing_weight = None").body[0].targets[0]
    assert (
        isinstance(fallback, ast.Assign)
        and len(fallback.targets) == 1
        and ast.dump(fallback.targets[0]) == ast.dump(target)
    ), f"{path}: expected mixing fallback immediately after parse_args"
    block = ast.Module(body=body[: parse_index + 2], type_ignores=[])
    namespace = {
        # datetime.strftime uses the real builtin importer for stdlib time.
        "__builtins__": {
            "float": float,
            "int": int,
            "str": str,
            "__import__": __import__,
        },
        "argparse": argparse,
        "datetime": datetime,
    }
    exec(compile(block, str(path), "exec"), namespace)
    return namespace["parser"], namespace["args"]
