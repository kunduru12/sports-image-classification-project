import streamlit as st
from PIL import Image
import numpy as np
import tensorflow_hub as hub
from tensorflow.keras.models import load_model

# Load your trained model
model_path = "mobilenetv2_6imagesv.h5"
model = load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})

# Streamlit UI
st.set_page_config(page_title="Sports Image Classifier", layout="wide", page_icon="🏀")

# Page Title
st.title("🎾 Sports Image Classifier")
st.write("Welcome! Upload an image, and the classifier will predict the type of sport and identify the player. Let's get started!")

# Sidebar for Sports Persons
st.sidebar.header("Select a Sports Person")
sports_persons = st.sidebar.selectbox("Who are you uploading an image of?", [
   'Kane Williamson','Kobe Bryant', 'lionel_messi','Maria Sharapova','ms_dhoni','neeraj_chopra'
])
#st.sidebar.write(f"**Selected Sports Person:** {sports_persons}")

# Main Section for Image Upload
st.header("Upload an Image")
uploaded_file = st.file_uploader("Choose an image file (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    def preprocess_image(image):
        image = image.resize((224, 224))  # Resize to match model input size
        image_array = np.array(image) / 255.0  # Normalize pixel values
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        return image_array

    preprocessed_image = preprocess_image(image)

    # Prediction
    prediction = model.predict(preprocessed_image)
    confidence = np.max(prediction) * 100

    # Define sport categories and class names
    categories = ["Cricket", "Basketball","Football","Tennis",'Cricket','athlete']
    class_names = [ 'Kane Williamson','Kobe Bryant', 'lionel_messi','Maria Sharapova','ms_dhoni','neeraj_chopra']

    # Get predicted category and class
    predicted_category = categories[np.argmax(prediction)]
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions, axis=1)

    # Display Results
    st.header("Prediction Results")
    st.success(f"🏅 **Sport**: {predicted_category}")
    st.info(f"**Predicted Player**: {class_names[predicted_class[0]]}")

    st.subheader("Confidence Scores")
    for i, score in enumerate(predictions[0]):
        st.write(f"{class_names[i]}: {score * 100:.2f}%")

    # Provide some feedback based on confidence
    if confidence > 80:
        st.balloons()
        st.write("🎉 Great! The model is very confident in its prediction!")
    else:
        st.warning("The model's confidence is below 80%. You might want to try another image.")
else:
    st.info("Please upload an image to see the prediction.")

# Footer Section
st.markdown("---")
st.markdown(
    "**Note**: This model is trained on a specific dataset and may not generalize to all sports images. "
    "Ensure the uploaded image is clear and focuses on the player for best results."
)
