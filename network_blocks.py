from keras.layers import *
from keras.models import Model, Sequential
from keras.applications import *
import efficientnet.keras as eff
from attention_functions import AttnGatingBlock
from inception_resnet_v2 import InceptionResNetV2

def squeeze_excite_block(inputs, ratio=8):
    init = inputs
    channel_axis = -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = Multiply()([init, se])
    return x

def conv_block(inputs, filters, use_se=False):
    x = inputs

    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    if use_se:
        x = squeeze_excite_block(x)

    return x

def dense_block(inputs, filters, depth=2, use_se=False):
    def simple_conv_block(inputs, filters):
        x = inputs
        x = Conv2D(filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    
    x = inputs
    d = depth

    l = x
    for i in range(d):
        x = simple_conv_block(l, filters)
        l = Concatenate()([l, x])
    
    if use_se:
        l = squeeze_excite_block(x)
    
    return l

def inception_block(inputs, filters, r=4, use_se=False):
    def simple_conv_block(inputs, filters, kernel):
        x = inputs
        x = Conv2D(filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        return x
    
    x = inputs
    f = filters
    
    a = simple_conv_block(x, f, kernel=(1,1))
    b = simple_conv_block(x, f//3, kernel=(1,1))
    b = simple_conv_block(b, f, kernel=(3,3))
    c = simple_conv_block(x, f//r, kernel=(1,1))
    c = simple_conv_block(c, f, kernel=(5,5))
    d = MaxPool2D((3,3), strides=(1,1), padding="same")(x)
    d = simple_conv_block(d, f, kernel=(1,1))
    
    x = Concatenate()([a, b, c, d])

    if use_se:
        x = squeeze_excite_block(x)
    
    return x

def attention_gate(g, x, filters=64):
    """
        g: input to be upsampled
        x: feature map from skip connection
    """

    gc = Conv2D(filters, (1,1), padding="same")(g)
    gc = UpSampling2D((2, 2), interpolation='bilinear')(gc)
    xc = Conv2D(filters, (1,1), padding="same")(x)
    
    add = Add()([gc, xc])
    add = Activation("relu")(add)
    add = Conv2D(1, (1,1), padding="same")(add)
    sig = Activation("sigmoid")(add)

    mult = Multiply()([sig, xc])

    return mult

def feature_resnet_encoder(inputs):
    skip_connections = []
    
    model = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)
    model.summary()
    exit()
    names = ['stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0']
    
    for name in names:
        skip_connections.append(model.get_layer(name).output)

    output = model.get_layer("activation_49").output
    
    return output, skip_connections

def feature_densenet_encoder(inputs, tipo="densenet201"):
    skip_connections = []
    list_layers = []
    model = None

    if tipo == "densenet121":
        list_layers = [4, 51, 139, 311]
        model = DenseNet121(include_top=False, weights="imagenet", input_tensor=inputs)
    elif tipo == "densenet169":
        list_layers = [4, 51, 139, 367]
        model = DenseNet169(include_top=False, weights="imagenet", input_tensor=inputs)
    elif tipo == "densenet201":
        list_layers = [4, 51, 139, 479]
        model = DenseNet201(include_top=False, weights="imagenet", input_tensor=inputs)
    
    if model == None:
        raise("Model is None")
    
    skip_connections = [model.layers[l].output for l in range(len(model.layers)) if l in list_layers]
    skip_connections.reverse()
    
    output = model.layers[-1].output
    
    return output, skip_connections    

def feature_IRV2_encoder(inputs):
    skip_connections = []
    
    list_layers = [594, 260, 16, 9][::-1]
    model = InceptionResNetV2(include_top=False, weights="imagenet", input_tensor=inputs)   
    if model == None:
        raise("Model is None")
    
    skip_connections = [model.layers[l].output for l in range(len(model.layers)) if l in list_layers]
    skip_connections.reverse()
    
    output = model.layers[-1].output
    
    return output, skip_connections 


def feature_effb3_encoder(inputs):
    skip_connections = []
    
    model = eff.EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
   
    # model.summary()
    # exit()
    names = ['block6a_expand_activation', 'block4a_expand_activation','block3a_expand_activation', 'block2a_expand_activation']
    # names = ['block4a_expand_activation','block3a_expand_activation', 'block2a_expand_activation']
    for name in names:
        skip_connections.append(model.get_layer(name).output)

    output = model.get_layer("block7a_expand_activation").output
    
    return output, skip_connections

def feature_vgg16_encoder(inputs):
    skip_connections = []
    
    model = VGG16(include_top=False, weights="imagenet", input_tensor=inputs)
    # model.summary()
    # exit()
    names = ["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3"] #vgg16
    names = names[::-1]
    for name in names:
        skip_connections.append(model.get_layer(name).output)

    output = model.get_layer("block5_conv3").output #vgg16
    return output, skip_connections

def feature_standard_decoder_att(inputs, skip_connections, use_se=False):
    num_filters = [512, 256, 128, 64, 32]
    # num_filters = [256, 128, 64, 32]
    depth = 4
    x = inputs

    for i, f in enumerate(num_filters[:depth]):
        att = attention_gate(x, skip_connections[i], filters=f)
        x = UpSampling2D((2, 2), interpolation='bilinear')(x)
        x = Concatenate()([x, att])
        x = conv_block(x, f, use_se=use_se)

    x = UpSampling2D((2, 2), interpolation='bilinear')(x)
    x = x = conv_block(x, num_filters[-1], use_se=use_se)

    return x


def feature_standard_decoder_noatt(inputs, skip_connections, use_se=False):
    num_filters = [512, 256, 128, 64, 32]
    # num_filters = [256, 128, 64, 32]
    depth = 4
    x = inputs

    for i, f in enumerate(num_filters[:depth]):
        x = UpSampling2D((2, 2), interpolation='bilinear')(x)
        x = Concatenate()([x, skip_connections[i]])
        x = conv_block(x, f, use_se=use_se)

    x = UpSampling2D((2, 2), interpolation='bilinear')(x)
    x = x = conv_block(x, num_filters[-1], use_se=use_se)

    return x

def feature_unet_standard_decoder_noatt(inputs, skip_connections, use_se=False):
    # num_filters = [512, 256, 128, 64, 32]
    num_filters = [256, 128, 64, 32]
    depth = 4
    x = inputs

    for i, f in enumerate(num_filters[:depth]):
        x = UpSampling2D((2, 2), interpolation='bilinear')(x)
        x = Concatenate()([x, skip_connections[i]])
        x = conv_block(x, f, use_se=use_se)

    # x = UpSampling2D((2, 2), interpolation='bilinear')(x)
    # x = x = conv_block(x, num_filters[-1], use_se=use_se)

    return x


def feature_inception_decoder_noatt(inputs, skip_connections, use_se=False):
    num_filters = [512, 256, 128, 64, 32]
    # num_filters = [256, 128, 64, 32]
    depth = 4
    x = inputs

    for i, f in enumerate(num_filters[:depth]):
        x = UpSampling2D((2, 2), interpolation='bilinear')(x)
        x = Concatenate()([x, skip_connections[i]])
        x = inception_block(x, f, use_se=use_se)

    x = UpSampling2D((2, 2), interpolation='bilinear')(x)
    x = inception_block(x, num_filters[-1], use_se=use_se)

    return x

def feature_inception_decoder_att(inputs, skip_connections, use_se=False):
    num_filters = [512, 256, 128, 64, 32]
    # num_filters = [256, 128, 64, 32]
    depth = len(num_filters) - 1
    x = inputs
    att_factor = 1
    for i, f in enumerate(num_filters[:depth]):
        att = attention_gate(x, skip_connections[i], filters=int(f*att_factor))
        # att = AttnGatingBlock(skip_connections[i],x,(32*16)*att_factor)
        # att_factor = att_factor/2
        x = UpSampling2D((2, 2), interpolation='bilinear')(x)
        x = Concatenate()([x, att])
        x = inception_block(x, f, use_se=use_se)

    x = UpSampling2D((2, 2), interpolation='bilinear')(x)
    x = inception_block(x, num_filters[-1], use_se=use_se)

    return x 

def feature_densenet_decoder_att(inputs, skip_connections, use_se=False):
    num_filters = [512, 256, 128, 64, 32]
    # num_filters = [256, 128, 64, 32]
    depth = 4
    x = inputs
    att_factor = 1
    for i, f in enumerate(num_filters[:depth]):
        att = attention_gate(x, skip_connections[i], filters=int(f*att_factor))
        # att = AttnGatingBlock(skip_connections[i],x,(32*16)*att_factor)
        # att_factor = att_factor/2
        x = UpSampling2D((2, 2), interpolation='bilinear')(x)
        x = Concatenate()([x, att])
        x = dense_block(x, f, use_se=use_se)

    x = UpSampling2D((2, 2), interpolation='bilinear')(x)
    x = dense_block(x, num_filters[-1], use_se=use_se)

    return x 

def output_block(inputs):
    x = Conv2D(4, (1, 1), padding="same")(inputs)
    x = Activation('softmax')(x)
    return x

def output_block_sig(inputs):
    x = Conv2D(1, (1, 1), padding="same")(inputs)
    x = Activation('sigmoid')(x)
    return x
 
def eais_net(shape):
    #Input
    inputs = Input(shape=shape)
        
    #Encoder Part
    encoder_eff, skip = feature_effb3_encoder(inputs)
    decoder_eff = feature_inception_decoder_att(encoder_eff, skip, use_se=True)
    
    output = output_block(decoder_eff)
    model_eff = Model(inputs, output)
   
    return model_eff

def build_model_eff_simple(shape):
    #Input
    inputs = Input(shape=shape)
        
    #Encoder Part
    encoder_eff, skip = feature_effb3_encoder(inputs)
    decoder_eff = feature_standard_decoder_noatt(encoder_eff, skip, use_se=False)
    
    output = output_block_sig(decoder_eff)
    model_eff = Model(inputs, output)
   
    return model_eff

def build_model_unet_simple(shape):
    #Input
    inputs = Input(shape=shape)
    
    #Encoder Part
    encoder_eff, skip = feature_vgg16_encoder(inputs)
    decoder_eff = feature_inception_decoder_att(encoder_eff, skip, use_se=True)
    # decoder_eff = feature_unet_standard_decoder_noatt(encoder_eff, skip, use_se=False)
    
    output = output_block(decoder_eff)
    model_eff = Model(inputs, output)
   
    return model_eff


