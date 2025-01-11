from types import NoneType, UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin

import msgspec
from traiNNer.utils.redux_options import (
    DatasetOptions,
    LogOptions,
    PathOptions,
    ReduxOptions,
    SchedulerOptions,
    TrainOptions,
    ValOptions,
)


def get_desc(field_type: type) -> str:
    origin_type = get_origin(field_type)
    if origin_type is Annotated:
        args = get_args(field_type)
        return args[1].description
    return ""


def type_to_str(field_name: str, field_type: type) -> str:
    origin_type = get_origin(field_type)
    if origin_type is Annotated:
        args = get_args(field_type)
        if get_origin(args[0]) in {Union, UnionType}:
            types = [
                t.__name__ for t in get_args(args[0]) if t is not NoneType
            ]  # all types in the union, filter out None
            return ", ".join(types)
        elif get_origin(args[0]) is Literal:
            # literals = get_args(args[0])
            # print("ohno", field_type, literals)
            return "Literal"
        else:
            return args[0].__name__
    print("skip", field_name)
    # return field_type.__name__ if hasattr(field_type, "__name__") else str(field_type)
    return ""


def get_literal_options(field_type: type) -> str:
    origin_type = get_origin(field_type)
    if origin_type is Annotated:
        args = get_args(field_type)
        if get_origin(args[0]) is Literal:
            literals = get_args(args[0])
            # print("ohno", field_type, literals)
            return ", ".join(literals)
    return ""


def parse_field(field_name: str, field_type: type, field_default: Any) -> str:
    """Parses a single field and returns its markdown documentation."""
    if field_name == "clip_size":
        print("no")
    type_name = type_to_str(field_name, field_type)
    if not type_name:
        return ""

    # default_value = field_default if field_default != msgspec.UNSET else "None"
    parts = [f"### {field_name}"]
    desc = get_desc(field_type)
    if desc:
        parts.append(desc)
    if type_name == "Literal":
        parts.append(f"Options: {get_literal_options(field_type)}")
    else:
        parts.append(f"Type: {type_name}")
    # parts.append(f"Default: {default_value}")  #TODO

    return "\n\n  ".join(parts)


def generate_markdown_doc(
    schema: type,
    header: str,
) -> str:
    print("generate markdown doc", schema)
    """
    Generates markdown documentation for a given Msgspec schema, including nested types.

    Args:
        schema: The Msgspec schema class.
        processed_schemas: A set of already processed schemas to avoid duplication.

    Returns:
        A string containing the markdown documentation.
    """

    docs = [f"## {header}\n"]

    for field in schema.__annotations__:
        field_type = schema.__annotations__[field]
        field_default = schema.__dict__.get(field, msgspec.UNSET)
        docs.append(parse_field(field, field_type, field_default))

    return "\n".join(docs)


if __name__ == "__main__":
    # Example usage:
    output_path = "docs/source/config_reference.md"

    with open(output_path, "w") as fout:
        fout.write("# Config file reference\n")
        fout.write(f"{generate_markdown_doc(ReduxOptions, 'Top level options')}\n")
        fout.write(
            f"{generate_markdown_doc(DatasetOptions, 'Dataset options (`datasets.train` and `datasets.val`)')}\n"
        )
        fout.write(f"{generate_markdown_doc(PathOptions, 'Path options (`path`)')}\n")
        fout.write(
            f"{generate_markdown_doc(TrainOptions, 'Train options (`train`)')}\n"
        )
        fout.write(
            f"{generate_markdown_doc(SchedulerOptions, 'Scheduler options (`train.scheduler`)')}\n"
        )
        fout.write(
            f"{generate_markdown_doc(ValOptions, 'Validation options (`val`)')}\n"
        )
        fout.write(
            f"{generate_markdown_doc(LogOptions, 'Logging options (`logger`)')}\n"
        )

    print(f"Documentation written to {output_path}")
