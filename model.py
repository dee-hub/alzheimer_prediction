import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import pandas as pd

# Custom objects dictionary
custom_objects = {
    'F1Score': tfa.metrics.F1Score
}

model_paths = {
    'cnn': "CNN Model/best_model_val_loss_0.28869_val_acc_0.90186_val_auc_0.98644_val_precision_0.90343_val_recall_0.89990.h5",
    'resnet': 'ResNet/best_model_val_loss_0.19303_val_acc_0.94482_val_auc_0.99129_val_precision_0.94708_val_recall_0.94385.h5'
}

models = {}
for model_name, path in model_paths.items():
    models[model_name] = tf.keras.models.load_model(path, custom_objects=custom_objects)


X_data = np.load("CNN Model/test_data.npy")
y_data = np.load("CNN Model/test_labels.npy")

st.title("Image Classification with CNN, ResNet-50 and Ensembles")

# Model Selection
model_choice = st.selectbox('Choose a model:', ['Convolutional Neural Networks (CNN)', 
                                                'ResidualNetworks-50', 
                                                'Ensemble Network (CNN + ResNet-50)'])

#print(model.summary())


# Load the trained model
#model = tf.keras.models.load_model('best_model_val_loss_0.19901_val_acc_0.95117_val_auc_0.98822_val_precision_0.95115_val_recall_0.95068.h5')  # Update with your model's path
class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']

def get_styled_class_display(class_name):
    # Define the color and icon for each class
    class_styles = {
        'NonDemented': {'color': 'success', 'icon': '✅'},
        'VeryMildDemented': {'color': 'warning', 'icon': '⚠️'},
        'MildDemented': {'color': 'primary', 'icon': '🔵'},
        'ModerateDemented': {'color': 'danger', 'icon': '❌'}
    }
    
    style = class_styles.get(class_name, {'color': 'secondary', 'icon': '❓'})
    
    # Use Bootstrap-styled badges to display the class name
    badge_html = f"""
    <p>
        <span class='badge badge-{style['color']}'>
            {style['icon']} {class_name}
        </span>
    </p>
    """
    
    return badge_html
# Define the roundoff function
if st.button('Run on random test image'):
    # Select a random image
    idx = np.random.randint(X_data.shape[0])
    img_array = X_data[idx]
    true_label = y_data[idx]

    img_array_resnet = cv2.resize(img_array, (224, 224))
    # Convert to grayscale if it's an RGB image
    if img_array_resnet.shape[-1] == 3:
        img_array_resnet = cv2.cvtColor(img_array_resnet, cv2.COLOR_RGB2GRAY)
    
    
    
    # Depending on the user's choice, select the model and predict
    if model_choice == "Ensemble Network (CNN + ResNet-50)":
	with st.expander("Network Description and Metrics")
	    st.markdown("Residual Networks are an evolution of traditional CNNs where skip connections are added. These connections help to propagate gradients better through deeper architectures. Below are the training and testing performance on a variety of metrics for the model developed.")
	    st.image('ResNet/resnet_loss_acc.jpg', caption='ResNet-50 Training and Validation loss and accuracy.', use_column_width=True)
	    st.image('ResNet/resnet_auc_precision_recall.jpg', caption='ResNet-50 Training and Validation AUC, precision and recall.', use_column_width=True)
	st.markdown("The randomly selected image is show below and the inference thereafter.")
	st.image(img_array, caption='Selected Image.', use_column_width=True)

        pred_cnn = models['cnn'].predict(img_array[None])[0]
        pred_resnet = models['resnet'].predict(img_array_resnet[None].reshape(-1, 224, 224, 1))[0]
        preds = np.mean([pred_cnn, pred_resnet], axis=0)

    
        table_data = {
           "Model": ["CNN", "ResNet", "Ensemble (if applicable)"],
           "NonDemented": [pred_cnn[0], pred_resnet[0], np.mean([pred_cnn[0], pred_resnet[0]])],
           "VeryMildDemented": [pred_cnn[1], pred_resnet[1], np.mean([pred_cnn[1], pred_resnet[1]])],
           "MildDemented": [pred_cnn[2], pred_resnet[2], np.mean([pred_cnn[2], pred_resnet[2]])],
           "ModerateDemented": [pred_cnn[3], pred_resnet[3], np.mean([pred_cnn[3], pred_resnet[3]])]
        }

    # Display the table
        st.table(pd.DataFrame(table_data))

    elif model_choice == "ResidualNetworks-50":
	with st.expander("ResNet Network Description and Metrics")
	    st.markdown("Residual Networks are an evolution of traditional CNNs where skip connections are added. These connections help to propagate gradients better through deeper architectures. Below are the training and testing performance on a variety of metrics for the model developed.")
	    st.image('ResNet/resnet_loss_acc.jpg', caption='ResNet-50 Training and Validation loss and accuracy.', use_column_width=True)
	    st.image('ResNet/resnet_auc_precision_recall.jpg', caption='ResNet-50 Training and Validation AUC, precision and recall.', use_column_width=True)
	    st.markdown('The metrics for the model in use is shown below')
	    table_data = {
               "Validation Loss": 0.19303,
               "Validation Accuracy": 0.94482,
               "Validation AUC": 0.99129,
               "Validation Precision": 0.94708,
               "Validation Recall": 0.94385
            }
    # Display the table
            st.table(pd.DataFrame(table_data))

	st.markdown("The randomly selected image is show below and the inference thereafter.")
	st.image(img_array, caption='Selected Image.', use_column_width=True)
        preds = models['resnet'].predict(img_array_resnet[None].reshape(-1, 224, 224, 1))[0]
        table_data = {
           "Model": ["ResNet"],
           "NonDemented": [preds[0]],
           "VeryMildDemented": [preds[1]],
           "MildDemented": [preds[2]],
           "ModerateDemented": [preds[3]]
        }

    # Display the table
        st.table(pd.DataFrame(table_data))
    else:
	with st.expander("CNN Network Description and Metrics")
	    st.markdown("Convolutional neural network is a regularized type of feed-forward neural network that learns feature engineering by itself via filters optimization. Vanishing gradients and exploding gradients, seen during backpropagation in earlier neural networks, are prevented by using regularized weights over fewer connections.")
	    st.image('CNN Model/loss_acc.jpg', caption='CNN Training and Validation loss and accuracy.', use_column_width=True)
	    st.image('CNN Model/auc_precision_recall.jpg', caption='CNN Training and Validation AUC, precision and recall.', use_column_width=True)
	    st.markdown('The metrics for the model in use is shown below')
	    table_data = {
               "Validation Loss": 0.28869,
               "Validation Accuracy": 0.90186,
               "Validation AUC": 0.98644,
               "Validation Precision": 0.90343,
               "Validation Recall": 0.89990
            }
    # Display the table
            st.table(pd.DataFrame(table_data))
	st.markdown("The randomly selected image is show below and the inference thereafter.")
	st.image(img_array, caption='Selected Image.', use_column_width=True)

        preds = models['cnn'].predict(img_array[None])[0]
        table_data = {
           "Model": ["CNN"],
           "NonDemented": [preds[0]],
           "VeryMildDemented": [preds[1]],
           "MildDemented": [preds[2]],
           "ModerateDemented": [preds[3]]
        }

    # Display the table
        st.table(pd.DataFrame(table_data))
    
    predicted_class = class_names[np.argmax(preds)]
    true_class = class_names[np.argmax(true_label)]
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<strong>True Class:</strong>", unsafe_allow_html=True)
        st.markdown(get_styled_class_display(true_class), unsafe_allow_html=True)
    
    with col2:
        st.markdown("<strong>Prediction:</strong>", unsafe_allow_html=True)
        st.markdown(get_styled_class_display(predicted_class), unsafe_allow_html=True)
