import inspect
import os
from collections.abc import Callable
from typing import Annotated, Literal, Union, get_type_hints

from msgspec import Meta, Struct, defstruct, json
from traiNNer.archs import ARCH_REGISTRY
from traiNNer.losses import LOSS_REGISTRY
from traiNNer.utils.redux_options import ReduxOptions, TrainOptions
from traiNNer.utils.registry import Registry


def generate_registry_structs(registry: Registry) -> dict[str, type[Struct]]:
    loss_structs = {}

    for name, cls in registry:
        try:
            # Inspect the constructor's signature
            sig = inspect.signature(cls.__init__)
            hints = get_type_hints(cls.__init__)

            exclude_types = {type, Callable}

            # Generate fields for the msgspec Struct
            fields = [
                (k, hints[k])
                for k in sig.parameters
                if k not in {"self", "args", "kwargs"}
                and not (
                    (
                        hasattr(hints[k], "__origin__")
                        and hints[k].__origin__ in exclude_types
                    )
                    or hints[k] in exclude_types
                )
            ]
            fields.append(("type", Literal[name]))

            # Dynamically create a Struct subclass
            print(name, fields)
            struct = defstruct(name=name, fields=fields, tag=True, tag_field="kind")
            loss_structs[name] = struct
            # break  # TODO remove
        except Exception as e:
            print("SKIP", name, e)

    return loss_structs


def create_dynamic_train_options(
    cls: type[Struct], name: str, new_val: tuple
) -> type[Struct]:
    fields = list(cls.__annotations__.items())
    fields.append(new_val)
    fields = tuple(fields)

    # Dynamically define a new TrainOptions struct
    return defstruct(name=name, fields=fields, tag=True)


# Generate the loss-specific structs
LOSS_STRUCTS = generate_registry_structs(LOSS_REGISTRY)
ARCH_STRUCTS = generate_registry_structs(ARCH_REGISTRY)
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

# new_arch_opts = defstruct(name="NetworkOptions", fields=[("", ArchType)])

new_redux_opts = create_dynamic_train_options(
    new_redux_opts, "ReduxOptions", ("network_g", ArchType)
)

schema = json.schema(new_redux_opts)
p = os.path.abspath("schemas/redux-config.schema.json")
print(p)
with open(p, "w") as schema_file:
    schema_file.write(json.encode(schema).decode())
