from django.shortcuts import render
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import os
import io
from django.conf import settings

model_path = os.path.join(settings.BASE_DIR, 'saved_models', 'eye_disease_model.h5')
model = tf.keras.models.load_model(model_path)
IMG_SIZE = (224, 224)

CLASS_NAMES = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
CONFIDENCE_THRESHOLD = 0.6

# Disease information
DISEASE_INFO = {
    'cataract': {
        'hospitals': ['LV Prasad Eye Institute', 'Aravind Eye Hospital', 'Apollo Hospitals'],
        'precautions': ['Wear sunglasses', 'Avoid bright light exposure', 'Eat antioxidant-rich food'],
        'treatments': ['Cataract surgery', 'Intraocular lens implantation']
    },
    'diabetic_retinopathy': {
        'hospitals': ['Sankara Nethralaya', 'Centre for Sight', 'Max Healthcare'],
        'precautions': ['Control blood sugar', 'Regular eye exams', 'Healthy diet'],
        'treatments': ['Laser surgery', 'Anti-VEGF injections', 'Vitrectomy']
    },
    'glaucoma': {
        'hospitals': ['LV Prasad Eye Institute', 'Narayana Nethralaya', 'Dr. Agarwals Eye Hospital'],
        'precautions': ['Avoid eye strain', 'Stay hydrated', 'Don’t skip medications'],
        'treatments': ['Eye drops', 'Trabeculectomy', 'Laser therapy']
    },
    'normal': {
        'hospitals': [],
        'precautions': ['Maintain a balanced diet', 'Wear protective eyewear', 'Get regular checkups'],
        'treatments': ['No treatment needed, but continue routine checkups']
    }
}

def predict_eye_disease(request):
    prediction = None
    confidence = None
    warning = None
    predicted_class = None
    class_confidences = []
    hospitals = []
    precautions = []
    treatments = []

    if request.method == 'POST':
        uploaded_image = request.FILES.get('image')

        if uploaded_image:
            try:
                img = image.load_img(io.BytesIO(uploaded_image.read()), target_size=IMG_SIZE)
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0

                predictions = model.predict(img_array)[0]
                predicted_index = np.argmax(predictions)
                predicted_class = CLASS_NAMES[predicted_index]
                confidence = float(predictions[predicted_index])
                class_confidences = [(CLASS_NAMES[i], float(predictions[i]) * 100) for i in range(len(CLASS_NAMES))]

                # Set prediction label
                if confidence < CONFIDENCE_THRESHOLD:
                    prediction = f"Possibly {predicted_class.capitalize()} (Low confidence)"
                    warning = "⚠ Model is uncertain. Please consult a specialist or try a clearer image."
                else:
                    prediction = f"{predicted_class.capitalize()} detected"

                # Add info
                disease_info = DISEASE_INFO[predicted_class]
                hospitals = disease_info['hospitals']
                precautions = disease_info['precautions']
                treatments = disease_info['treatments']

            except Exception as e:
                prediction = f"Error during prediction: {str(e)}"

    return render(request, 'predict.html', {
        'prediction': prediction,
        'confidence': f"{confidence * 100:.2f}%" if confidence is not None else None,
        'threshold': f"{CONFIDENCE_THRESHOLD * 100:.2f}%",
        'warning': warning,
        'class_confidences': class_confidences,
        'predicted_class': predicted_class.capitalize() if predicted_class else None,
        'hospitals': hospitals,
        'precautions': precautions,
        'treatments': treatments,
    })