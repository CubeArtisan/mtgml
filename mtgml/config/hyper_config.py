from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar, overload

import yaml
from tensorflow.keras.utils import register_keras_serializable

# import mtgml.layers.configurable_layer
# from mtgml.layers.configurable_layer import ConfigurableLayer

ValueType = TypeVar('ValueType', float, int, str, bool, list, 'HyperConfig')
LayerType = TypeVar('LayerType', bound='mtgml.layers.configurable_layer.ConfigurableLayer')
LayerType2 = TypeVar('LayerType2', bound='mtgml.layers.configurable_layer.ConfigurableLayer')


@register_keras_serializable(package='mtgml.config', name='HyperConfigValue')
@dataclass
class HyperConfigValue(Generic[ValueType]):
    help: str
    min: ValueType | None = None
    max: ValueType | None = None
    step: ValueType | None = None
    logdist: bool | None = None
    choices: Sequence[ValueType] | None = None
    value: ValueType | None = None

    @classmethod
    def from_config(cls, config):
        return cls(help=config.get('help'), min=config.get('min'), max=config.get('max'),
                   step=config.get('step'), logdist=config.get('logdist'),
                   choices=config.get('choices'), value=config.get('value'))

    def get_config(self):
        config = {'help': self.help, 'value': self.value}
        if self.choices is not None:
            config['choices'] = self.choices
        return config


@register_keras_serializable(package='mtgml.config', name='HyperConfig')
class HyperConfig(Generic[LayerType]):
    def __init__(self, data: dict[str, HyperConfigValue] = {}, layer_type: type[LayerType] | None = None,
                 fixed: dict = {}, seed: int = 5723):
        self.data = dict(data or {})
        self.layer_type = layer_type
        self.fixed = fixed
        self.seed = seed

    @overload
    def get_int(self, name: str, *, default: int, help: str,
                min: int | None = None, max: int | None = None,
                step: int | None = None, logdist: bool | None = None) -> int:
        ...
    @overload
    def get_int(self, name: str, *, default: None, help: str,
                min: int | None = None, max: int | None = None,
                step: int | None = None, logdist: bool | None = None) -> int | None:
        ...

    def get_int(self, name, *, default, help,
                min=None, max=None,
                step=None, logdist=None) -> int | None:
        if name in self.fixed:
            return self.fixed[name]
        if name in self.data:
            if self.data[name].value is not None:
                return self.data[name].value
        else:
            self.data[name] = HyperConfigValue(help=help, min=min, max=max, step=step,
                                               logdist=logdist, value=default)
        return default

    @overload
    def get_float(self, name: str, *, default: float, help: str,
                  min: float | None = None, max: float | None = None,
                  step: float | None = None, logdist: bool | None = None) -> float:
        ...

    @overload
    def get_float(self, name: str, *, default: None, help: str,
                  min: float | None = None, max: float | None = None,
                  step: float | None = None, logdist: bool | None = None) -> float | None:
        ...

    def get_float(self, name: str, *, default: float | None, help: str,
                  min: float | None = None, max: float | None = None,
                  step: float | None = None, logdist: bool | None = None) -> float | None:
        if name in self.fixed:
            return self.fixed[name]
        if name in self.data:
            if self.data[name].value is not None:
                return self.data[name].value
        else:
            self.data[name] = HyperConfigValue(help=help, min=min, max=max, step=step,
                                               logdist=logdist, value=default)
        return default

    def get_bool(self, name: str, *, default: bool, help: str) -> bool:
        if name in self.fixed:
            return self.fixed[name]
        if name in self.data:
            if self.data[name].value is not None:
                return self.data[name].value or False
        else:
            self.data[name] = HyperConfigValue(help=help, value=default)
        return default

    @overload
    def get_list(self, name: str, *, default: list, help: str) -> list:
        ...

    @overload
    def get_list(self, name: str, *, default: None, help: str) -> list | None:
        ...

    def get_list(self, name: str, *, default: list | None, help: str) -> list | None:
        if name in self.fixed:
            return self.fixed[name]
        if name in self.data:
            if self.data[name].value is not None:
                return self.data[name].value
        else:
            self.data[name] = HyperConfigValue(help=help, value=default)
        return default

    @overload
    def get_choice(self, name: str, *, default: ValueType, choices: Sequence[ValueType],
                   help: str) -> ValueType:
        ...

    @overload
    def get_choice(self, name: str, *, default: ValueType | None, choices: Sequence[ValueType],
                   help: str) -> ValueType | None:
        ...

    def get_choice(self, name: str, *, default: ValueType | None, choices: Sequence[ValueType],
                   help: str) -> ValueType | None:
        if name in self.fixed:
            return self.fixed[name]
        if name in self.data:
            if self.data[name].value is not None:
                return self.data[name].value
        else:
            self.data[name] = HyperConfigValue[ValueType](help=help, choices=choices, value=default)
        return default

    def get_sublayer_config(self, name: str, *, sub_layer_type: type[LayerType2], help: str,
                            seed_mod: int = 7, fixed: dict = {}) -> 'HyperConfig':
        if name in self.fixed:
            value = self.fixed[name]
            if isinstance(value, HyperConfig):
                return value
            value = dict(value)
            value.update(fixed)
            fixed = value
        data = {}
        if name in self.data:
            value = self.data[name].value
            if value is not None:
                if isinstance(value, HyperConfig):
                    value.layer_type = sub_layer_type
                    value.fixed = fixed
                    return value
                else:
                    data = dict(value)
        config = HyperConfig(layer_type=sub_layer_type, data=data, fixed=fixed, seed=self.seed * seed_mod)
        self.data[name] = HyperConfigValue(help=help, value=config)
        return config

    def get_sublayer(self, name: str, *, sub_layer_type: type[LayerType2], help: str, seed_mod=7,
                            fixed: dict = {}) -> LayerType2:
        config = self.get_sublayer_config(name, sub_layer_type=sub_layer_type, help=help,
                                          seed_mod=seed_mod, fixed=fixed)
        return config.build(name=name)

    def build(self, *args, **kwargs) -> LayerType:
        if self.layer_type is not None:
            self.layer_type.get_properties(self, input_shapes=None)
            return self.layer_type(self, *args, **kwargs)
        else:
            raise ValueError('Tried to build a HyperConfig without a specified layer type.')

    def get_config(self) -> dict:
        data = {key: item for key, item in self.data.items() if key != 'seed' and (not isinstance(item.value, HyperConfig) or len(item.value.data) > 1)}
        return data

    @property
    def seed(self) -> int:
        return self.data['seed'].value or 7

    @seed.setter
    def seed(self, value: int):
        self.data['seed'] = HyperConfigValue(value=value, help='The seed for the rng')

    @classmethod
    def from_config(cls, config) -> 'HyperConfig':
        return cls(data=config['data'], layer_type=None, fixed=config['fixed'], seed=config['seed'])


def value_representer(dumper, data):
    return dumper.represent_mapping(u'!hcv', data.get_config())


def value_constructor(loader, node):
    kwargs = loader.construct_mapping(node)
    return HyperConfigValue(**kwargs)


def config_representer(dumper, data):
    return dumper.represent_mapping(u'!hc', data.get_config())


def config_constructor(loader, node):
    kwargs = loader.construct_mapping(node)
    return HyperConfig(data=kwargs)


yaml.add_representer(HyperConfigValue, value_representer)
yaml.add_constructor(u'!hcv', value_constructor)
yaml.add_representer(HyperConfig, config_representer)
yaml.add_constructor(u'!hc', config_constructor)
