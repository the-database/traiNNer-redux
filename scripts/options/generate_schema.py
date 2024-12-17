# Generate a JSON schema for yaml config file
import os

import msgspec
from traiNNer.utils.redux_options import ReduxOptions

schema = msgspec.json.schema(ReduxOptions)
p = os.path.abspath("schemas/redux-config.schema.json")
print(p)
with open(p, "w") as schema_file:
    schema_file.write(msgspec.json.encode(schema).decode())
