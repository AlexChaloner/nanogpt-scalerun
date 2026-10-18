"""nanoGPT-style scalar config files and --key=value overrides.

Configuration files contain assignments and simple arithmetic. They are parsed
as data, so imports, function calls, and arbitrary Python are rejected.
"""

import argparse
import ast
import math
import operator
from pathlib import Path


DEFAULTS = {
    "out_dir": "runs/out", "dataset": "shakespeare_char", "data_dir": "",
    "init_from": "scratch", "eval_only": False, "eval_interval": 0,
    "n_layer": 2, "n_head": 2, "n_embd": 64, "block_size": 64,
    "vocab_size": 0, "bias": True, "dropout": 0.0, "batch_size": 8,
    "gradient_accumulation_steps": 1, "max_iters": 200,
    "learning_rate": 0.001, "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95,
    "grad_clip": 1.0, "decay_lr": True, "warmup_iters": 20,
    "lr_decay_iters": 0, "min_lr": 0.0001,
    "device": "auto", "dtype": "float32", "compile": False,
    "log_interval": 10, "eval_batch_size": 32, "checkpoint_interval": 50,
    "seed": 1337, "num_threads": 4, "contest": "", "scale": "s0",
    "recipe_id": "baseline",
    "stop_after_steps": 0,
}


def _expression(node, values, depth=0):
    if depth > 24:
        raise ValueError("configuration expression is too deeply nested")
    if isinstance(node, ast.Constant) and type(node.value) in (str, int, float, bool, type(None)):
        value = node.value
    elif isinstance(node, ast.Name) and node.id in values:
        value = values[node.id]
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _expression(node.operand, values, depth + 1)
        if type(value) not in (int, float):
            raise ValueError("arithmetic requires numeric configuration values")
        value = value if isinstance(node.op, ast.UAdd) else -value
    elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv)):
        left = _expression(node.left, values, depth + 1)
        right = _expression(node.right, values, depth + 1)
        if type(left) not in (int, float) or type(right) not in (int, float):
            raise ValueError("arithmetic requires numeric configuration values")
        functions = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                     ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv}
        value = functions[type(node.op)](left, right)
    else:
        raise ValueError("configs support scalar assignments and arithmetic; imports/calls are not allowed")
    if type(value) in (int, float) and (abs(value) > 10**20 or not math.isfinite(value)):
        raise ValueError("configuration number is outside the supported finite range")
    return value


def _value(text):
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def _set(values, key, value):
    if key not in DEFAULTS:
        raise ValueError(f"Unknown configuration key: {key}")
    expected = type(DEFAULTS[key])
    if expected is float and type(value) is int:
        value = float(value)
    if type(value) is not expected:
        raise ValueError(f"{key} needs a {expected.__name__}, received {value!r}")
    values[key] = value


def load_config(path=None, overrides=None):
    values = dict(DEFAULTS)
    if path:
        source = Path(path).read_text(encoding="utf-8")
        if len(source) > 100_000:
            raise ValueError("configuration file is too large")
        for statement in ast.parse(source, filename=str(path)).body:
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
                continue
            if (not isinstance(statement, ast.Assign) or len(statement.targets) != 1
                    or not isinstance(statement.targets[0], ast.Name)):
                raise ValueError("configuration files must contain scalar assignments")
            _set(values, statement.targets[0].id, _expression(statement.value, values))
    for key, value in (overrides or {}).items():
        _set(values, key, value)
    return values


def parse_config(argv=None):
    parser = argparse.ArgumentParser(description="Train a GPT with nanoGPT-style configs and ScaleRun compute caps")
    parser.add_argument("config", nargs="?", help="config/train_shakespeare_char.py or another scalar config")
    for key, default in DEFAULTS.items():
        parser.add_argument(f"--{key}", type=_value, default=argparse.SUPPRESS,
                            help=f"default: {default!r}")
    arguments = vars(parser.parse_args(argv))
    path = arguments.pop("config", None)
    return load_config(path, arguments), Path(path).resolve() if path else None
