import importlib.metadata
import importlib.util
import sys
import tomllib

from packaging.requirements import InvalidRequirement, Requirement


def get_requirements_from_pyproject() -> list[Requirement]:
    with open("pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)

    try:
        dependencies = pyproject["project"]["dependencies"]
    except KeyError as err:
        raise RuntimeError("No dependencies found in pyproject.toml") from err

    try:
        return [Requirement(dep) for dep in dependencies]
    except InvalidRequirement as err:
        raise RuntimeError(f"Invalid dependency in pyproject.toml: {err}") from None


def check_dependencies() -> None:
    if sys.platform == "win32":
        cmd = "./install.bat"
    else:
        cmd = "./install.sh"

    for req in get_requirements_from_pyproject():
        # skip dependencies for other platforms, e.g. "triton-windows ...; sys_platform == 'win32'"
        if req.marker is not None and not req.marker.evaluate():
            continue

        try:
            installed_version = importlib.metadata.version(req.name)
        except importlib.metadata.PackageNotFoundError:
            raise RuntimeError(
                f"{req.name} is not installed. Please run this command to install: {cmd}"
            ) from None

        if req.specifier and not req.specifier.contains(
            installed_version, prereleases=True
        ):
            raise RuntimeError(
                f"{req.name} version {installed_version} does not satisfy {req.specifier}. Please run this command to update dependencies: {cmd}"
            )

    # Linux torch brings triton (ROCm torch its ROCm build); Windows gets triton-windows above
    if importlib.util.find_spec("triton") is None:
        raise RuntimeError(
            f"triton is not installed. Please run this command to install: {cmd}"
        )
