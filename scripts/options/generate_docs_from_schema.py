import json
import os
from typing import Any


def format_type(prop_schema: dict[str, Any]) -> str:
    """Format the type information from the property schema."""
    t = prop_schema.get("type")
    if isinstance(t, list):
        return ", ".join(t)
    elif t is not None:
        return t
    if "enum" in prop_schema:
        return "enum"
    return "object"


def format_default(prop_schema: dict[str, Any]) -> str:
    """Format the default value if present."""
    if "default" in prop_schema:
        val = prop_schema["default"]
        # If default is None, show as null
        if val is None:
            val = "null"
        return f"(default: {val})"
    return ""


def format_description(prop_schema: dict[str, Any]) -> str:
    """Format the description if present."""
    desc = prop_schema.get("description", "")
    return desc.strip()


def format_enum(prop_schema: dict[str, Any]) -> str:
    """Format enum values if present."""
    if "enum" in prop_schema:
        return "One of: " + ", ".join([f"`{e}`" for e in prop_schema["enum"]])
    return ""


def format_items(prop_schema: dict[str, Any]) -> str:
    """If the property is an array, attempt to describe its items."""
    if prop_schema.get("type") == "array":
        item_schema = prop_schema.get("items", {})
        if isinstance(item_schema, dict):
            item_type = item_schema.get("type", "any")
            if not item_type:
                item_type = "any"
            return f"Array of {item_type}."
    return ""


def fully_resolve_schema(schema: Any, defs: dict[str, Any], visited=None) -> Any:
    """
    Recursively resolve all $ref references in the schema.
    Returns a schema with all references inlined.
    """
    if visited is None:
        visited = set()

    if isinstance(schema, dict):
        # If there's a $ref, resolve it
        if "$ref" in schema:
            ref = schema["$ref"]
            if ref.startswith("#/$defs/"):
                ref_name = ref.split("/")[-1]
                if ref_name in defs:
                    if ref_name in visited:
                        return defs[ref_name]
                    visited.add(ref_name)
                    resolved = fully_resolve_schema(
                        defs[ref_name], defs, visited=visited
                    )
                    return resolved
                else:
                    return {}
            else:
                return schema

        # Resolve nested structures
        new_schema = {}
        for k, v in schema.items():
            if k in ("anyOf", "allOf", "oneOf"):
                # Resolve each element
                if isinstance(v, list):
                    new_schema[k] = [
                        fully_resolve_schema(x, defs, visited=visited) for x in v
                    ]
                else:
                    new_schema[k] = v
            elif k == "items":
                new_schema[k] = fully_resolve_schema(v, defs, visited=visited)
            elif k == "properties":
                props = {}
                for p_name, p_schema in v.items():
                    props[p_name] = fully_resolve_schema(
                        p_schema, defs, visited=visited
                    )
                new_schema[k] = props
            else:
                new_schema[k] = fully_resolve_schema(v, defs, visited=visited)
        return new_schema
    elif isinstance(schema, list):
        return [fully_resolve_schema(item, defs, visited=visited) for item in schema]
    else:
        return schema


def is_simple_schema(s: dict[str, Any]) -> bool:
    """
    Determine if a schema is "simple" — i.e., it just specifies a type or enum
    and does not have nested properties.
    """
    if "properties" in s:
        return False
    if any(k in s for k in ["anyOf", "oneOf", "allOf"]):
        return False
    # If it just defines a type or enum without complexity, we call it simple.
    keys = set(s.keys()) - {"type", "enum", "description", "default", "format", "const"}
    return len(keys) == 0


def render_combiners(
    prop_name: str, prop_schema: dict[str, Any], combo_key: str, level: int
) -> str:
    """
    Render anyOf/oneOf/allOf clauses. If all options are simple, show them inline.
    If not, show them as subsections.
    """
    md = []
    combo_schemas = prop_schema[combo_key]

    # Check if all options are simple
    all_simple = all(is_simple_schema(opt) for opt in combo_schemas)

    if all_simple:
        # Render a single line: "This field can be any of: ... "
        options_descriptions = []
        for opt in combo_schemas:
            # For each simple option, show its type or enum
            t = format_type(opt)
            e = format_enum(opt)
            if e:
                options_descriptions.append(e.replace("One of: ", ""))
            else:
                options_descriptions.append(f"`{t}`")
        joined = " or ".join(options_descriptions)
        md.append(f"  Can be: {joined}\n")
    else:
        # More complex: render subsections
        for i, subschema in enumerate(combo_schemas, start=1):
            md.append(
                f"\n{'#'*(level+1)} {prop_name.capitalize()} {combo_key} option {i}\n"
            )
            if subschema.get("properties"):
                s_props = subschema["properties"]
                s_required = subschema.get("required", [])
                md.append(render_properties(s_props, s_required, level + 1))
            else:
                # If no properties, just describe
                s_desc = format_description(subschema)
                t = format_type(subschema)
                e = format_enum(subschema)
                line = "- "
                if s_desc:
                    line += s_desc + " "
                if e:
                    line += f"{e} "
                if t:
                    line += f"(type: {t})"
                md.append(line + "\n")
    return "\n".join(md)


def render_properties(
    props: dict[str, Any], required: list[str], level: int = 3
) -> str:
    """Render a dictionary of properties into markdown."""
    md = []
    for prop_name, prop_schema in props.items():
        prop_type = format_type(prop_schema)
        default_val = format_default(prop_schema)
        desc = format_description(prop_schema)
        enum_info = format_enum(prop_schema)
        items_info = format_items(prop_schema)
        req = " (required)" if prop_name in required else ""

        line = f"- **{prop_name}** *{prop_type}*{req} {default_val}\n"
        if desc:
            line += f"  {desc}\n"
        if enum_info:
            line += f"  {enum_info}\n"
        if items_info:
            line += f"  {items_info}\n"
        md.append(line)

        # If this property is an object with its own properties, recurse
        if prop_schema.get("type") == "object" and "properties" in prop_schema:
            sub_props = prop_schema["properties"]
            sub_required = prop_schema.get("required", [])
            if sub_props:
                md.append("\n" + "#" * (level + 1) + f" {prop_name.capitalize()}\n")
                md.append(render_properties(sub_props, sub_required, level + 1))

        # Handle anyOf/oneOf/allOf
        for combo_key in ("anyOf", "oneOf", "allOf"):
            if combo_key in prop_schema:
                combo_str = render_combiners(prop_name, prop_schema, combo_key, level)
                md.append(combo_str)

    return "\n".join(md)


def render_schema(schema: dict[str, Any], name: str = None) -> str:
    """
    Render a JSON schema as markdown after fully resolving references.
    """
    defs = schema.get("$defs", {})
    # Resolve top-level $ref if present
    if "$ref" in schema:
        schema = fully_resolve_schema(schema, defs)
    else:
        # Even if there's no top-level $ref, still fully resolve
        schema = fully_resolve_schema(schema, defs)

    title = schema.get("title", name or "Schema")
    schema_type = schema.get("type", "object")
    required = schema.get("required", [])
    description = schema.get("description", "")

    md = []
    md.append(f"# {title}\n")
    md.append(f"**Type:** `{schema_type}`")
    # if required:
    #     md.append("\n**Required:** " + ", ".join(required))
    if description:
        md.append("\n**Description:** " + description)

    props = schema.get("properties", {})
    if props:
        md.append("\n### Properties\n")
        md.append(render_properties(props, required, 3))

    return "\n".join(md)


if __name__ == "__main__":
    # Example usage:
    # Replace 'schemas/redux-config.schema.json' with your path.
    with (
        open(os.path.abspath("schemas/redux-config.schema.json")) as f,
        open("docs/source/config.md", "w") as fout,
    ):
        your_schema_string = f.read()
        schema = json.loads(your_schema_string)
        fout.write(render_schema(schema))
