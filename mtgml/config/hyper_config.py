from dataclasses import dataclass
from typing import Generic, TypeVar, Union

from mtgml.layers.configurable_layer import ConfigurableLayer

ValueType = TypeVar('ValueType', float, int, str, bool, list, 'HyperConfig')
LayerType = TypeVar('LayerType', bound=ConfigurableLayer)
LayerType2 = TypeVar('LayerType2', bound=ConfigurableLayer)


@dataclass
class HyperConfigValue(Generic[ValueType])
    help: str
    min: Union[ValueType, None]
    max: Union[ValueType, None]
    step: Union[ValueType, None]
    logdist: bool
    choices: Union[tuple[ValueType], None]
    value: Union[ValueType, None] = None


class HyperConfig(Generic[LayerType]):
    def __init__(self, layer_type: Type[LayerType], data: dict[str, HyperConfigValue],
                 fixed: dict = {}, seed: int = 5723):
        self.layer_type = layer_type
        self.data = data
        self.fixed = fixed

    def get_int(self, name: str, *, default: Union[int, None], help: str,
                min: Union[int, None] = None, max: Union[int, None] = None,
                step: Union[int, None] = None, logdist: bool = False) -> int:
        if name in self.fixed:
            return self.fixed[name]
        if name in data:
            if self.data[name].value is not None:
                return self.data[name].value
        else:
            self.data[name] = HyperConfigValue(help=help, min=min, max=max, step=step, logdist=logdist,
                                          choices=None)
        return default

    def get_float(self, name: str, *, default: Union[float, None], help: str,
                  min: Union[float, None] = None, max: Union[float, None] = None,
                  step: Union[float, None] = None, logdist: bool = False) -> Union[float, None]:
        if name in self.fixed:
            return self.fixed[name]
        if name in data:
            if self.data[name].value is not None:
                return self.data[name].value
        else:
            self.data[name] = HyperConfigValue(help=help, min=min, max=max, step=step, logdist=logdist,
                                          choices=None)
        return default

    def get_bool(self, name: str, *, default: bool, help: str) -> bool:
        if name in self.fixed:
            return self.fixed[name]
        if name in data:
            if self.data[name].value is not None:
                return self.data[name].value
        else:
            self.data[name] = HyperConfigValue(help=help, min=None, max=None, step=None, logdist=logdist,
                                          choices=(True, False))
        return default

    def get_list(self, name: str, *, default: Union[list, None], help: str) -> Union[list, None]:
        if name in self.fixed:
            return self.fixed[name]
        if name in data:
            if self.data[name].value is not None:
                return self.data[name].value
        else:
            self.data[name] = HyperConfigValue(help=help, min=None, max=None, step=None, logdist=False,
                                          choices=None)
        return default

    def get_choice(self, name: str, *, default: Union[ValueType, None], choices=list[ValueType],
                   help: str) -> Union[ValueType, None]:
        if name in self.fixed:
            return self.fixed[name]
        if name in data:
            if self.data[name].value is not None:
                return self.data[name].value
        else:
            self.data[name] = HyperConfigValue(help=help, min=None, max=None, step=None, logdist=False,
                                          choices=None)
        return default

    def get_sublayer_config(self, name: str, *, sub_layer_type: type[LayerType2]], help: str,
                            seed_mod: int = 7, fixed: dict = {}) -> HyperConfig:
        if name in self.fixed:
            value = self.fixed[name]
            if isinstance(value, HyperConfig):
                return value
            value = dict(value)
            value.update(fixed)
            fixed = value
        if name in data:
            if self.data[name].value is not None:
                return self.data[name].value
        config = HyperConfig(sub_layer_type, data={}, fixed=fixed, seed=self.seed * seed_mod)
        self.data[name] = HyperConfigValue(help=help, min=min, max=max, step=step, logdist=logdist,
                                      choices=None, value=config)
        return config

    def get_sublayer(self, name: str, *, sub_layer_type: type[LayerType2]], help: str, seed_mod=7,
                            fixed: dict = {}) -> LayerType2:
        config = self.get_sublayer_config(name, sub_layer_type=sub_layer_type, help=help,
                                          seed_mod=seed_mod, fixed=fixed)
        return sub_layer_type(config, name=name)

    def build(self, *args, **kwargs) -> LayerType:
        self.layer_type.get_properties(self, input_shapes=None)
        return self.layer_type(self, *args, **kwargs)

    def to_dict(self) -> dict:
        return self.data

    @classmethod
    def from_dict(cls, *args, **kwargs) -> HyperConfig:
        config = cls(*args, **kwargs)
