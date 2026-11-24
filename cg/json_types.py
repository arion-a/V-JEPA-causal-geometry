"""JSON-compatible value type shared by the modules that read or write JSON."""

type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
