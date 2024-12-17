# Generate a JSON schema for yaml config file
import msgspec
from traiNNer.utils.redux_options import ReduxOptions

schema = msgspec.json.schema(ReduxOptions)

with open("schemas/redux-config.schema.json", "w") as schema_file:
    schema_file.write(msgspec.json.encode(schema).decode())
