import tensorflow as tf


class ConfigurableLayer(tf.keras.layers.Layer):
    def __init__(self, hyper_config, **kwargs):
        super(self, ConfigurableLayer).__init__(**kwargs)
        self.hyper_config = hyper_config
        self.seed = self.hyper_config.seed

    def get_config(self):
        config = super(self, ConfigurableLayer).get_config()
        config.update({ "hyper_config": self.hyper_config.to_dict()})
        return config

    @classmethod
    def from_config(cls, config):
        config['hyper_config'] = HyperConfig(layer_type=cls, data=config['hyper_config'])
        return cls(**config)

    def build(self, input_shapes):
        properties = self.get_properties(self.hyper_config, input_shapes=input_shapes)
        for key, prop in properties.items():
            if isinstance(prop, list):
                for i in range(len(prop)):
                    if isinstance(prop[i], HyperLayer):
                        prop[i] = prop[i]()
            if isinstance(prop, HyperLayer):
                prop = prop()
            setattr(self, key, prop)
