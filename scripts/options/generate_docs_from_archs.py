import inspect
from collections.abc import Callable
from inspect import signature

from traiNNer.archs import ARCH_REGISTRY, SPANDREL_REGISTRY
from traiNNer.utils.misc import is_json_compatible

EXCLUDE_BENCHMARK_ARCHS = {
    "artcnn",
    "dct",
    "eimn",
    "hat",
    "swinir",
    "swin2sr",
    "lmlt",
    "vggfeatureextractor",
}
FILTERED_REGISTRY = [
    (name, arch)
    for name, arch in list(SPANDREL_REGISTRY) + list(ARCH_REGISTRY)
    if name not in EXCLUDE_BENCHMARK_ARCHS
]


def function_to_markdown(func: Callable, header: str) -> str:
    # Start building the Markdown output
    md = [f"#### {header}", ""]

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
                print("skip", param_name, pd)
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


def class_to_markdown(cls: type, header: str) -> str:
    """
    Converts a Python class's docstring and its parameters' docstrings into Markdown format.

    Args:
        cls (type): The class to document.

    Returns:
        str: A string containing the Markdown documentation for the class.
    """

    # Document the `__init__` method if available
    init_method = getattr(cls, "__init__", None)
    assert init_method is not None
    return function_to_markdown(init_method, header)


def callable_to_markdown(callable: Callable | type, header: str) -> str:
    if inspect.isfunction(callable):
        return function_to_markdown(callable, header)
    assert isinstance(callable, type)
    return class_to_markdown(callable, header)


documented_archs = set()

discriminators = {
    "dunet",
    "metagan2",
    "multiscalepatchgandiscriminatorsn",
    "patchgandiscriminatorsn",
    "unetdiscriminatorsn",
    "vggstylediscriminator",
}

g_docs = {}
d_docs = {}

if __name__ == "__main__":
    output_path = "docs/source/arch_reference.md"

    for _, arch in FILTERED_REGISTRY:
        # print("arch", arch)
        if inspect.isfunction(arch):
            net = arch(scale=4)
            cls: type = net.__class__
        else:
            cls = arch  # type: ignore
        if cls.__name__.lower() not in discriminators:
            doc = g_docs
        else:
            doc = d_docs

        cls_key = cls.__name__

        if cls_key not in doc:
            doc[cls_key] = {}

        arch_key = arch.__name__.lower()
        print(arch_key)
        markdown = callable_to_markdown(arch, arch_key)
        doc[cls_key][arch_key] = markdown

    with open(output_path, "w") as fout:
        fout.write("# Architecture reference\n")
        fout.write(
            "This page lists all available parameters for each architecture in traiNNer-redux. While the default configs use the official defaults and shouldn't need to be modified by most users, advanced users may wish to inspect or modify architectures to train to their liking. Please keep in mind that changing parameters for generator architectures can affect compatibility with using pretrain models.\n"
        )
        fout.write("## Generator architectures (`network_g`)\n")
        for arch_group, doc_dict in sorted(g_docs.items()):
            fout.write(f"### {arch_group}\n")
            for _, arch_md in sorted(doc_dict.items()):
                fout.write(f"{arch_md}\n")

        fout.write("## Discriminator architectures (`network_d`)\n")
        for arch_group, doc_dict in sorted(d_docs.items()):
            fout.write(f"### {arch_group}\n")
            for _, arch_md in sorted(doc_dict.items()):
                fout.write(f"{arch_md}\n")
