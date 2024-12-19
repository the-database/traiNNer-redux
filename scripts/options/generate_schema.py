import inspect
import os
from collections.abc import Callable
from typing import Annotated, Union, get_type_hints

from msgspec import NODEFAULT, Meta, Struct, defstruct, field, json
from msgspec.structs import fields
from traiNNer.archs import ARCH_REGISTRY, SPANDREL_REGISTRY
from traiNNer.losses import LOSS_REGISTRY
from traiNNer.utils.redux_options import ReduxOptions, TrainOptions
from traiNNer.utils.registry import Registry


def generate_registry_structs(registry: Registry | list) -> dict[str, type[Struct]]:
    structs = {}

    for name, cls in registry:
        try:
            sig = inspect.signature(cls.__init__)
            hints = get_type_hints(cls.__init__)

            exclude_types = {type, Callable}

            fields = []
            for k, param in sig.parameters.items():
                if k in {"self", "args", "kwargs"}:
                    continue

                hint = hints.get(k)
                if hint is None:
                    continue

                if (
                    hasattr(hint, "__origin__") and hint.__origin__ in exclude_types
                ) or hint in exclude_types:
                    continue

                default = (
                    param.default
                    if param.default is not inspect.Parameter.empty
                    else NODEFAULT
                )
                fields.append((k, hint, default))

            struct = defstruct(name=name, fields=fields, tag=True)
            structs[name] = struct
        except Exception as e:
            print("SKIP", name, e)

    return structs


def create_dynamic_train_options(
    cls: type[Struct], name: str, new_val: tuple
) -> type[Struct]:
    new_fields = [
        (
            f.name,
            f.type,
            field(default_factory=f.default_factory)
            if f.default_factory != NODEFAULT
            else field(default=f.default),
        )
        for f in fields(cls)
    ]
    new_fields.append(new_val)
    new_fields = tuple(new_fields)
    return defstruct(name=name, fields=new_fields, kw_only=True)


LOSS_STRUCTS = generate_registry_structs(LOSS_REGISTRY)
ARCH_STRUCTS = generate_registry_structs(list(ARCH_REGISTRY) + list(SPANDREL_REGISTRY))
LossType = Union[tuple(LOSS_STRUCTS.values())]  # noqa: UP007
ArchType = Union[tuple(ARCH_STRUCTS.values())]  # noqa: UP007

new_train_opts = create_dynamic_train_options(
    TrainOptions,
    "TrainOptions",
    (
        "losses",
        Annotated[
            list[LossType],
            Meta(description="The list of loss functions to optimize."),
        ],
    ),
)
new_redux_opts = create_dynamic_train_options(
    ReduxOptions, "ReduxOptions", ("train", new_train_opts)
)


new_redux_opts = create_dynamic_train_options(
    new_redux_opts, "ReduxOptions", ("network_g", ArchType)
)

schema = json.schema(new_redux_opts)
p = os.path.abspath("schemas/redux-config.schema.json")
print(p)
with open(p, "w") as schema_file:
    schema_file.write(json.encode(schema).decode())
