
def getmod():
    from tensorflow.keras.models import load_model
    
    model = load_model('../src_train_test/trained_keras_model')
    
    return model