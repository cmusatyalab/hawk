# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from typing import Any, ClassVar

from importlib_metadata import entry_points
from logzero import logger
from pydantic import ValidationError
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from typing_extensions import Self


class EnvOverrideConfig(BaseSettings):
    # reorder initialization order so that environment vars (admin configured)
    # override init settings (user provided).
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            env_settings,
            dotenv_settings,
            file_secret_settings,
            init_settings,
        )


class HawkPluginConfig(EnvOverrideConfig):
    # have to override this for each plugin type to get the env_prefix right
    model_config = SettingsConfigDict(
        env_prefix="hawk_plugin_",
        env_file=["hawk.env", ".env"],
        env_file_encoding="utf-8",
        extra="ignore",
    )


class HawkPlugin:
    config_class: ClassVar[type[HawkPluginConfig]]
    config: HawkPluginConfig

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> HawkPluginConfig:
        return cls.config_class.model_validate(
            {k: v for k, v in config.items() if not k.startswith("_")},
        )

    @classmethod
    def scrub_config(
        cls,
        config: dict[str, Any],
        *,
        exclude: set[str] | None = None,
    ) -> dict[str, str]:
        validated = cls.validate_config(config)
        json_dict = validated.model_dump(
            mode="json",
            exclude_defaults=True,
            exclude=exclude,
        )
        return {k: str(v) for k, v in json_dict.items()}

    @classmethod
    def from_config(cls, config: dict[str, Any], **kwargs: Any) -> Self:
        return cls(cls.validate_config(config), **kwargs)

    def __init__(self, config: HawkPluginConfig) -> None:
        self.config = config


def get_plugin_entrypoint(plugin_type: str, plugin: str) -> type[HawkPlugin]:
    """Try to load a plugin into Hawk.

    raise ImportError when we failed to find or import the plugin.
    """
    try:
        # plugin type "model" -> group "cmuhawk.models", etc.
        plugin_ep = entry_points(group=f"cmuhawk.{plugin_type}s")[plugin]
    except KeyError:
        msg = f"Unknown {plugin_type}: {plugin}"
        raise ImportError(msg) from None

    try:
        plugin_cls: type[HawkPlugin] = plugin_ep.load()
        return plugin_cls
    except ModuleNotFoundError as e:
        raise ImportError from e


def load_plugin(
    plugin_type: str,
    plugin: str,
    config: dict[str, Any],
    **kwargs: Any,
) -> HawkPlugin:
    """Attempt to load and instantiate a plugin."""
    try:
        plugin_cls = get_plugin_entrypoint(plugin_type, plugin)
        return plugin_cls.from_config(config, **kwargs)
    except Exception as e:
        msg = f"Failed to load {plugin_type}.{plugin}: {e}"
        raise NotImplementedError(msg) from e


def validate_and_scrub_config(
    plugin_type: str,
    plugin: str,
    config: dict[str, Any],
    *,
    inject: dict[str, str] | None = None,
) -> dict[str, str]:
    """Attempt to validate the config and return a cleaned up config.

    inject is an optional dictionary of overrides or default values that will
    be scrubbed from the returned config.
    """
    if inject is None:
        inject = {}

    # cleanup config_dict, drop None values and convert to strings
    clean_config = {k: str(v) for k, v in config.items() if v is not None}

    try:
        # Try to validate @home, we may not be able to load the
        # retriever (missing module/dependencies) and we won't know
        # what the admin configured 'data_root' is on the scout. The
        # fallback is that we send everything to the scout and run
        # validation there.
        plugin_cls = get_plugin_entrypoint(plugin_type, plugin)

        clean_config = plugin_cls.scrub_config(
            dict(clean_config, **inject),
            exclude=set(inject),
        )

    except ImportError:
        logger.info(
            f"Import error, deferring {plugin_type}.{plugin} validation to scout.",
        )

    except Exception as e:
        errmsg = f"Failed to validate {plugin_type}.{plugin} config: {e}"
        raise NotImplementedError(errmsg) from e

    return clean_config


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", action="store_true")
    parser.add_argument("plugin_type")
    parser.add_argument("plugin")
    parser.add_argument("config", nargs="*", metavar="config=value")
    args = parser.parse_args()

    config = dict([setting.split("=", 1) for setting in args.config])

    try:
        plugin_cls = get_plugin_entrypoint(args.plugin_type, args.plugin)

        if args.schema:
            schema = plugin_cls.config_class.model_json_schema()
            print(json.dumps(schema, indent=4))
        else:
            plugin_config = plugin_cls.validate_config(config)
            print(f"{plugin_config!r}")
    except ImportError:
        print(f'Unknown {args.plugin_type}: "{args.plugin}"')
    except ValidationError as e:
        print(e)
