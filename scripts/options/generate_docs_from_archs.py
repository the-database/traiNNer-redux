import inspect

from traiNNer.archs import ARCH_REGISTRY, SPANDREL_REGISTRY

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


def class_to_markdown(cls, header):
    """
    Converts a Python class's docstring and its parameters' docstrings into Markdown format.

    Args:
        cls (type): The class to document.

    Returns:
        str: A string containing the Markdown documentation for the class.
    """
    # Get the class docstring
    class_doc = cls.__doc__.strip() if cls.__doc__ else ""

    # Start building the Markdown output
    md = [f"# {header}", ""]

    # Document the `__init__` method if available
    init_method = getattr(cls, "__init__", None)

    # Extract parameters from `__init__` method
    from inspect import signature

    try:
        sig = signature(init_method)
        md.append("")
        md.append("```")
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            param_doc = (
                f"{param_name}: {param.default}"
                if param.default is not param.empty
                else f"{param_name}: {param.annotation}"
            )

            md.append(f"{param_doc}")
        md.append("```")
    except Exception:
        md.append("No parameter information available.")

    # Join the Markdown lines into a single string
    return "\n".join(md)


documented_archs = set()
cls_to_names: dict[type, list[str]] = {}
for _, arch in FILTERED_REGISTRY:
    if inspect.isfunction(arch):
        net = arch(scale=4)
        cls: type = net.__class__
    else:
        cls = arch  # type: ignore
    if cls not in cls_to_names:
        cls_to_names[cls] = []
    cls_to_names[cls].append(arch.__name__.lower())
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
                markdown = class_to_markdown(cls, ", ".join(cls_to_names[cls]))
                fout.write(f"{markdown}\n")
                documented_archs.add(cls)
for k, v in cls_to_names.items():
    print(k, v)
