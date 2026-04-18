"""Lightweight guardrails for keeping the new Stage2 implementation isolated."""

FORBIDDEN_IMPORT_PREFIXES = (
    "stage2_old",
    "model.contact",
    "model.crefine",
)


def is_forbidden_import(module_name):
    module_name = str(module_name).strip()
    return any(
        module_name == prefix or module_name.startswith(prefix + ".")
        for prefix in FORBIDDEN_IMPORT_PREFIXES
    )
