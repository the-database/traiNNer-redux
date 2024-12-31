import inspect
from inspect import signature

from traiNNer.archs import ARCH_REGISTRY, SPANDREL_REGISTRY
from traiNNer.utils.misc import is_json_compatible

EXCLUDE_BENCHMARK_ARCHS = {
    "artcnn",
    "dct",
    "dunet",
    "eimn",
    "hat",
    "swinir",
    "swin2sr",
    "lmlt",
    "vggstylediscriminator",
    "unetdiscriminatorsn_traiNNer",
    "vggfeatureextractor",
}
FILTERED_REGISTRY = [
    (name, arch)
    for name, arch in list(SPANDREL_REGISTRY) + list(ARCH_REGISTRY)
    if name not in EXCLUDE_BENCHMARK_ARCHS
]


def function_to_markdown(func, header):
    # Start building the Markdown output
    md = [f"## {header}", ""]

    try:
        sig = signature(func)
        md.append("")
        md.append("```")
        for param_name, param in sig.parameters.items():
            pd = param.default
            print(header, param_name, pd)
            if isinstance(pd, tuple):
                pd = list(pd)
            elif isinstance(pd, bool):
                pd = str(pd).lower()
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


if __name__ == "__main__":
    output_path = "docs/source/arch_reference.md"

    with open(output_path, "w") as fout:
        for _, arch in FILTERED_REGISTRY:
            if inspect.isfunction(arch):
                net = arch(scale=4)
                cls: type = net.__class__
            else:
                cls = arch  # type: ignore
            if cls not in documented_archs:
                fout.write(f"# {cls.__name__}\n")
            markdown = callable_to_markdown(arch, arch.__name__)
            fout.write(f"{markdown}\n")
            documented_archs.add(cls)
