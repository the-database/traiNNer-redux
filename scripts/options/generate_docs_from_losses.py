import inspect
from inspect import signature

from traiNNer.losses import LOSS_REGISTRY
from traiNNer.utils.misc import is_json_compatible


def function_to_markdown(func, header):
    # Start building the Markdown output
    md = [f"## {header}", ""]

    try:
        sig = signature(func)
        md.append("")
        md.append("```yaml")
        md.append(f"type: {header}")
        for param_name, param in sig.parameters.items():
            pd = param.default
            # print(header, param_name, pd)
            if param_name == "scale":
                continue
            if isinstance(pd, tuple):
                pd = list(pd)
            elif isinstance(pd, bool):
                pd = str(pd).lower()
            elif pd is None:
                pd = "~"
            if param_name == "self" or not is_json_compatible(pd):
                continue
            param_doc = (
                f"{param_name}: {pd}"
                if pd is not param.empty
                else f"{param_name}: {param.annotation}"
            )

            md.append(f"{param_doc}")
        md.append("```")
    except Exception as e:
        print(e)
        md.append("No parameter information available.")

    # Join the Markdown lines into a single string
    return "\n".join(md)


def class_to_markdown(cls, header):
    """
    Converts a Python class's docstring and its parameters' docstrings into Markdown format.

    Args:
        cls (type): The class to document.

    Returns:
        str: A string containing the Markdown documentation for the class.
    """

    # Document the `__init__` method if available
    init_method = getattr(cls, "__init__", None)

    return function_to_markdown(init_method, header)


def callable_to_markdown(callable, header) -> str:
    if inspect.isfunction(callable):
        return function_to_markdown(callable, header)
    return class_to_markdown(callable, header)


documented_archs = set()
# cls_to_names: dict[type, list[str]] = {}
# for _, arch in FILTERED_REGISTRY:
#     if inspect.isfunction(arch):
#         net = arch(scale=4)
#         cls: type = net.__class__
#     else:
#         cls = arch  # type: ignore
#     if cls not in cls_to_names:
#         cls_to_names[cls] = []
#     cls_to_names[cls].append(arch.__name__.lower())

docs = {}

if __name__ == "__main__":
    output_path = "docs/source/loss_reference.md"

    for _, arch in LOSS_REGISTRY:
        # print("arch", arch)
        if inspect.isfunction(arch):
            net = arch(scale=4)
            cls: type = net.__class__
        else:
            cls = arch  # type: ignore

        cls_key = cls.__name__

        if cls_key not in docs:
            docs[cls_key] = {}

        # if cls not in documented_archs:
        #     fout.write(f"## {cls_key}\n")
        arch_key = arch.__name__.lower()
        print(arch_key)
        markdown = callable_to_markdown(arch, arch_key)
        docs[cls_key][arch_key] = markdown
        # fout.write(f"{markdown}\n")
        # documented_archs.add(cls)

    with open(output_path, "w") as fout:
        fout.write("# Loss reference\n")
        fout.write(
            "This page lists all available parameters for each loss function in traiNNer-redux. While the default configs use recommended default values and shouldn't need to be modified by most users, advanced users may wish to inspect or modify loss function params to suit their specific use case.\n"
        )
        for arch_group, doc_dict in sorted(docs.items()):
            # fout.write(f"### {arch_group}\n")
            for _, arch_md in sorted(doc_dict.items()):
                fout.write(f"{arch_md}\n")
