def SR2Model(input_depth=3, highway_depth=4, block_depth=7, init='he_normal', init_last = RandomNormal(mean=0.0, stddev=0.001)):

    input_shape = [None, None, input_depth]
    input_lr = tf.keras.layers.Input(shape=input_shape)
    input_lr2 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(input_lr)
    
    depth_list = []
    
    x = input_lr
    for i in range(block_depth):
        x = tf.keras.layers.Conv2D(highway_depth, (3, 3), padding='same', kernel_initializer=init)(x)
        x = tf.nn.crelu(x)
        depth_list.append(x)

    x = tf.keras.layers.Concatenate(axis=-1)(depth_list)
    x = tf.keras.layers.Conv2D(4*input_depth, (1, 1), padding='same', kernel_initializer=init_last)(x)
    x = DepthToSpace2(4*input_depth)(x)
    
    x = tf.keras.layers.Add()([x, input_lr2])

    model = tf.keras.models.Model(input_lr, x)

    return model

def __init__(self):
    super(CNN, self).__init__()
    input_depth = 3
    self.conv_layer1 = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv_layer2 = nn.Conv2D(4*input_depth, (1, 1), padding=(0,0))

def forward(self, x):
    # conv layers
    block_depth = 7
    depth_list = []
    for i in range(block_depth):
        x = self.conv_layer1(x)
        x = nn.crelu(x)
        depth_list.append(x)
    
    x = torch.cat(depth_list,dim=-1)
    x = self.conc_layer2(x)
    outputs = functional.pixel_shuffle(x, 2)
    self.add
    return outputs