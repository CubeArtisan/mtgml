import tensorflow as tf


class ConfigurableLayer(tf.keras.layers.Layer):
    def __init__(self, hyper_config, **kwargs):
        super(ConfigurableLayer, self).__init__(**kwargs)
        self.hyper_config = hyper_config
        self.seed = self.hyper_config.seed
        self.built = False

    def get_config(self):
        config = super(ConfigurableLayer, self).get_config()
        config.update({ "hyper_config": self.hyper_config.get_config()})
        return config

    @classmethod
    def from_config(cls, config):
        config['hyper_config'] = HyperConfig(layer_type=cls, data=config['hyper_config'])
        return cls(**config)

    def build(self, input_shapes):
        if self.built:
            return
        properties = self.get_properties(self.hyper_config, input_shapes=input_shapes)
        for key, prop in properties.items():
            setattr(self, key, prop)
        self.built = True
