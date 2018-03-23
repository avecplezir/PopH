import numpy as np
import tensorflow as tf
from keras.layers import Input, LSTM, Dense, Dropout, Lambda, Reshape, Permute
from keras.layers import TimeDistributed, RepeatVector, Conv1D, Activation
from keras.layers import Embedding, Flatten, dot, concatenate 
from keras.layers.merge import Concatenate, Add
from keras.models import Model
import keras.backend as K
from keras import losses

from util import one_hot
from constants import *

from keras.utils import multi_gpu_model

def primary_loss(y_true, y_pred):
    # 3 separate loss calculations based on if note is played or not
    played = y_true[:, :, :, 0]
    bce_note = losses.binary_crossentropy(y_true[:, :, :, 0], y_pred[:, :, :, 0])
    bce_replay = losses.binary_crossentropy(y_true[:, :, :, 1], tf.multiply(played, y_pred[:, :, :, 1]) + tf.multiply(1 - played, y_true[:, :, :, 1]))
    mse = losses.mean_squared_error(y_true[:, :, :, 2], tf.multiply(played, y_pred[:, :, :, 2]) + tf.multiply(1 - played, y_true[:, :, :, 2]))
    return bce_note + bce_replay + mse

def pitch_pos_in_f(time_steps):
    """
    Returns a constant containing pitch position of each note
    """
    def f(x):
        note_ranges = tf.range(NUM_NOTES, dtype='float32') / NUM_NOTES
        repeated_ranges = tf.tile(note_ranges, [tf.shape(x)[0] * time_steps])
        return tf.reshape(repeated_ranges, [tf.shape(x)[0], time_steps, NUM_NOTES, 1])
    return f

def pitch_class_in_f(time_steps):
    """
    Returns a constant containing pitch class of each note
    """
    def f(x):
        pitch_class_matrix = np.array([one_hot(n % OCTAVE, OCTAVE) for n in range(NUM_NOTES)])
        pitch_class_matrix = tf.constant(pitch_class_matrix, dtype='float32')
        pitch_class_matrix = tf.reshape(pitch_class_matrix, [1, 1, NUM_NOTES, OCTAVE])
        return tf.tile(pitch_class_matrix, [tf.shape(x)[0], time_steps, 1, 1])
    return f

def pitch_bins_f(time_steps):
    def f(x):
        bins = tf.reduce_sum([x[:, :, i::OCTAVE, 0] for i in range(OCTAVE)], axis=3)
        bins = tf.tile(bins, [NUM_OCTAVES, 1, 1])
        bins = tf.reshape(bins, [tf.shape(x)[0], time_steps, NUM_NOTES, 1])
        return bins
    return f

def time_axis(dropout):
    """
    LSTM along the time axis
    """
    def f(notes, beat):
        time_steps = int(notes.get_shape()[1])

        # TODO: Experiment with when to apply conv
        note_octave = TimeDistributed(Conv1D(OCTAVE_UNITS, 2 * OCTAVE, padding='same'))(notes)
        note_octave = Activation('tanh')(note_octave)
        note_octave = Dropout(dropout)(note_octave)

        # Create features for every single note.
        note_features = Concatenate()([
            Lambda(pitch_pos_in_f(time_steps))(notes),
            Lambda(pitch_class_in_f(time_steps))(notes),
            Lambda(pitch_bins_f(time_steps))(notes),
            note_octave,
            TimeDistributed(RepeatVector(NUM_NOTES))(beat)
        ])

        x = note_features
        # [batch, notes, time, features]
        x = Permute((2, 1, 3))(x)

        # Apply LSTMs
        for l in range(TIME_AXIS_LAYERS):

            x = TimeDistributed(LSTM(TIME_AXIS_UNITS, return_sequences=True))(x)
            x = Dropout(dropout)(x)

        # [batch, time, notes, features]
        return Permute((2, 1, 3))(x)
    return f

def note_axis(dropout):
    """
    LSTM along the note axis
    """
    lstm_layer_cache = {}
    note_dense = Dense(2, activation='sigmoid', name='note_dense')
    volume_dense = Dense(1, name='volume_dense')

    def f(x, chosen):
        time_steps = int(x.get_shape()[1])

        # Shift target one note to the left.
        shift_chosen = Lambda(lambda x: tf.pad(x[:, :, :-1, :], [[0, 0], [0, 0], [1, 0], [0, 0]]))(chosen)

        # [batch, time, notes, features + 1]
        x = Concatenate(axis=3)([x, shift_chosen])


        for l in range(NOTE_AXIS_LAYERS):
            if l not in lstm_layer_cache:
                lstm_layer_cache[l] = LSTM(NOTE_AXIS_UNITS, return_sequences=True)

            x = TimeDistributed(lstm_layer_cache[l])(x)
            x = Dropout(dropout)(x)
          
        return Concatenate()([note_dense(x), volume_dense(x)])
    return f

def build_models(time_steps=SEQ_LEN, input_dropout=0.2, dropout=0.5):
    notes_in = Input((time_steps, NUM_NOTES, NOTE_UNITS))
    beat_in = Input((time_steps, NOTES_PER_BAR))
    # Target input for conditioning
    chosen_in = Input((time_steps, NUM_NOTES, NOTE_UNITS))

    # Dropout inputs
    notes = Dropout(input_dropout)(notes_in)
    beat = Dropout(input_dropout)(beat_in)
    chosen = Dropout(input_dropout)(chosen_in)

    """ Time axis """
    time_out = time_axis(dropout)(notes, beat)

    """ Note Axis & Prediction Layer """
    naxis = note_axis(dropout)
    notes_out = naxis(time_out, chosen)

    model = Model([notes_in, chosen_in, beat_in], [notes_out])

#     if len(K.tensorflow_backend._get_available_gpus())>=2:
    model = multi_gpu_model(model)
    #print(model)
    model.compile(optimizer='nadam', loss=[primary_loss])

    """ Generation Models """
    time_model = Model([notes_in, beat_in], [time_out])

    note_features = Input((1, NUM_NOTES, TIME_AXIS_UNITS), name='note_features')
    chosen_gen_in = Input((1, NUM_NOTES, NOTE_UNITS), name='chosen_gen_in')
    style_gen_in = Input((1, NUM_STYLES), name='style_in')

    # Dropout inputs
    chosen_gen = Dropout(input_dropout)(chosen_gen_in)
    
    note_gen_out = naxis(note_features, chosen_gen)
    
    note_model = Model([note_features, chosen_gen_in], note_gen_out)

    return model, time_model, note_model

def build_models_with_attention(time_steps=SEQ_LEN, input_dropout=0.2, dropout=0.5):
    notes_in = Input((time_steps, NUM_NOTES, NOTE_UNITS))
    beat_in = Input((time_steps, NOTES_PER_BAR))
    # Target input for conditioning
    chosen_in = Input((time_steps, NUM_NOTES, NOTE_UNITS))

    # Dropout inputs
    notes = Dropout(input_dropout)(notes_in)
    beat = Dropout(input_dropout)(beat_in)
    chosen = Dropout(input_dropout)(chosen_in)

    """ Time axis """
    time_out = time_axis(dropout)(notes, beat)
    print('time_out', time_out.shape)

    """ Note Axis & Prediction Layer """
    naxis = note_axis_attention(dropout)
    notes_out = naxis(time_out)
    
    model = Model([notes_in, chosen_in, beat_in], [notes_out])

    if len(K.tensorflow_backend._get_available_gpus())>=2:
        model = multi_gpu_model(model)

    model.compile(optimizer='nadam', loss=[primary_loss])
    
    """ Generation Models """
    time_model = Model([notes_in, beat_in], [time_out])

    note_features = Input((1, NUM_NOTES, TIME_AXIS_UNITS), name='note_features')
    chosen_gen_in = Input((1, NUM_NOTES, NOTE_UNITS), name='chosen_gen_in')
   
    # Dropout inputs
    chosen_gen = Dropout(input_dropout)(chosen_gen_in)
    
    print('NUM_NOTES', NUM_NOTES)
    note_gen_out = naxis(note_features)
    
    note_model = Model([note_features, chosen_gen_in], note_gen_out)

    return model, time_model, note_model

def note_axis_attention(dropout):
    note_dense_att = Dense(2, activation='sigmoid', name='note_dense_att')
    volume_dense_att = Dense(1, name='volume_dense_att')

    def f(x):
        x = attention_layer(x, x, False)
        #print('x_att', x.shape)
        x = Dropout(dropout)(x)
        #print('x_drop', x.get_shape)
        #x = Reshape((128, 48, -1))(x)
        #print('the end')
        #print('dense_vol', v.shape)
  
        return Concatenate(axis=-1)([note_dense_att(x), volume_dense_att(x)])
    
    return f

def OneHeadAttention(a_drop, q_drop, drop_ratio=0.5):
        
    a_proj = Dense(PROJECTION_DIM, use_bias=False, kernel_initializer='glorot_normal')(a_drop)
    q_proj = Dense(PROJECTION_DIM, use_bias=False, kernel_initializer='glorot_normal')(q_drop)
    v_proj = Dense(PROJECTION_DIM, use_bias=False, kernel_initializer='glorot_normal')(a_drop)
    
    a_proj = Dropout(drop_ratio)(a_proj)
    q_proj = Dropout(drop_ratio)(q_proj)
    v_proj = Dropout(drop_ratio)(v_proj)
    #print('a_proj', a_proj.shape)
    
    n = Dense(2)(v_proj)
    #print('dense_note', n.shape)
 
    
    att_input = Lambda(lambda x: tf.matmul(x[0],x[1], transpose_b=True))([q_proj, a_proj])
    #print('att_input', att_input.shape)


    att_weights = Activation('softmax')(att_input)
    v_new = Lambda(lambda x: tf.matmul(x[0],x[1]))([att_weights, v_proj])
    #tf.matmul(att_weights, v_proj)
    #print('v_new', v_new.get_shape)
     
    v_new = Multiply()([q_proj, v_new])
    
    return v_new

def MultyHeadAttention(a_drop, q_drop):

    Attention_heads = []
    for i in range(N_HEADS):
        Attention_heads.append(OneHeadAttention(a_drop, q_drop))
        
    BigHead = concatenate(Attention_heads, axis=-1)
    #print('BigHead', BigHead.shape)
    

    attention_output = Dense(DENSE_SIZE, use_bias=False)(BigHead)
    #print('attention_output', attention_output.shape)

           
    return attention_output
    
def attention_layer(a_drop, q_drop, FF):
    
    #print('a_drop', a_drop.shape)
    res = MultyHeadAttention(a_drop, q_drop)
    #print('res', res.shape)
        
    att = Add()([res, res])
    #att = normalize()(att)    
 
    #Feed Forward
    if FF:
        att_ff = TimeDense(DENSE_SIZE*4, activation = 'relu')(att)
        att_ff = Dense(DENSE_SIZE)(att_ff)   
        att_ff = Dropout(0.1)(att_ff)
        att_add = Add()([att, att_ff])
        #att = normalize()(att_add) 
    
    return att



