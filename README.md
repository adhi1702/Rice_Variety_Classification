# Rice Crop Assistant: Variety Identification and Disease Diagnosis

## 📌 Overview

**Rice Crop Assistant** is an AI-powered system designed to support rural farmers by identifying rice varieties and diagnosing crop diseases using image and text inputs. Tailored for agricultural communities in South and Southeast Asia, this tool bridges the gap between traditional farming knowledge and modern agricultural technology.

Millions of farmers depend on rice cultivation for their livelihood. However, accurately identifying rice varieties and detecting diseases like bacterial leaf blight, leaf blast, and brown spot remains a major challenge—especially in remote areas with limited access to agricultural experts. This system offers a reliable, easy-to-use solution that helps farmers make informed decisions in real-time.

---

## 🔍 Key Features

- 🌾 **Rice Variety Recognition**  
  Identifies rice types based on grain image characteristics like size, shape, and texture.

- 🌱 **Crop Disease Detection**  
  Detects common rice diseases from leaf images, including:
  - Bacterial Leaf Blight
  - Leaf Blast
  - Brown Spot

- 🧠 **Intelligent Recommendations**  
  Provides treatment suggestions and fertilizer recommendations tailored to the diagnosed disease or crop condition.

- 🗣️ **Local Language Support**  
  Supports inputs and outputs in native languages, ensuring accessibility for rural farmers.

- 🌐 **Offline Capability (optional)**  
  Offers an offline mode for areas with limited internet connectivity.

---

## 🏗️ Tech Stack

- **Frontend:** PyQt5 (GUI)
- **Backend:** Python, OpenCV, TensorFlow/Keras or PyTorch (for image classification)
- **Database:** MySQL (for storing history, logs, and user preferences)
- **Optional Integrations:**
  - Text-to-speech for audio instructions
  - SMS/WhatsApp APIs for updates

---

## 🚀 How to Use

1. **Launch the Application**  
   Run the main application file from your system.

2. **Choose Input Method**  
   - Upload an image of a rice grain or rice leaf.
   - Optionally, enter a short description (e.g., "yellow spots on leaves").

3. **Get Results**  
   - View identified rice variety or detected disease.
   - Follow step-by-step suggestions for treatment or cultivation.

4. **Save or Share**  
   - Save reports locally or send them to nearby agriculture officers.

---

## 🧪 Dataset

Training and testing are done using publicly available rice variety and rice disease image datasets. Ensure datasets are preprocessed (resized, labeled) and augmented for better model accuracy.

---

## 🔧 Installation

```bash
git clone https://github.com/yourusername/rice-crop-assistant.git
cd rice-crop-assistant
pip install -r requirements.txt
python main.py
