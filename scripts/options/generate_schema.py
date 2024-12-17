# Generate a JSON schema for yaml config file
import msgspec
from traiNNer.utils.redux_options import ReduxOptions

schema = msgspec.json.schema(ReduxOptions)

# Print out that schema as JSON
print(msgspec.json.encode(schema))
