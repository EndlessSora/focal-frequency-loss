import tensorflow as tf

def fft2d(img):
    img = tf.cast(img,dtype=tf.complex64)
    fft = tf.signal.fft2d(img)
    fft = tf.math.real(fft)
    return fft

#loss stable than focal frequency loss
def frequency_loss(y_true,y_pred):

    fft_true = fft2d(y_true)
    fft_pred = fft2d(y_pred)

    return tf.keras.losses.MeanSquaredError()(fft_true,fft_pred)


#easily go to inf....
def focal_frequency_loss(y_true,y_pred):
    
    fft_true = fft2d(y_true)
    fft_pred = fft2d(y_pred)
    
    alpha = tf.constant(1.0, dtype=tf.float32)
    
    sub = tf.math.subtract(fft_true,fft_pred)
    sub_abs = tf.math.abs(sub)
    norm = tf.math.divide(sub_abs, tf.math.reduce_max(sub_abs))
    
    weight = tf.math.pow(norm,alpha)
    
    se = tf.math.square(tf.math.subtract(fft_true,fft_pred))
    
    w_se = tf.math.multiply(weight,se)
    
    
    loss = tf.math.reduce_mean(w_se)
    
    
    return loss
