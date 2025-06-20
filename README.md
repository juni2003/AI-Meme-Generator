# 🧠 AI Meme Generator

An intelligent, offline, GUI-based Meme Generator that uses Natural Language Processing to recommend the most contextually relevant memes based on user input, and lets you personalize them with editing tools — all powered by TF-IDF and cosine similarity.

![Project Demo](demo_screenshot.png)

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [How to Use](#how-to-use)
- [Evaluation](#evaluation)
- [Limitations](#limitations)
- [Tech Stack](#tech-stack)
- [Contributors](#contributors)
- [License](#license)

---

## 💡 Project Overview

### 🧭 Domain

**Natural Language Processing (NLP)** — specifically text vectorization and semantic similarity.

### 📃 Description

The AI Meme Generator enables users to input a topic, phrase, or punchline and automatically recommends the most semantically similar meme from a pre-captioned dataset of 6,992 labeled memes. Users can also personalize the meme with caption removal, custom text overlays, cropping, and white blocks — making meme creation fast, fun, and intelligent.

### 🎯 Objectives

- Retrieve relevant memes using **text similarity** (TF-IDF + Cosine).
- Provide an **interactive GUI** for meme creation.
- Enable **caption editing** and **visual enhancements**.
- Deliver a **complete offline solution** for desktop users.
- Evaluate using **Top-K similarity accuracy** instead of classification metrics.

### 🎯 Target Users

- Content Creators
- Social Media Managers
- Meme Enthusiasts
- Digital Marketers

---

## 🚀 Features

- 🔍 Search memes by typing a phrase or topic
- 🧠 Smart recommendation using NLP (TF-IDF + Cosine Similarity)
- 🖼️ GUI for meme preview, edit & export (Tkinter)
- ✂️ Crop, overlay white boxes, and add custom captions
- 💾 Export and download edited memes
- 🔌 100% offline desktop application

---

## 🧱 System Architecture

### Components:

1. **Preprocessing Module**: Cleans and normalizes captions.
2. **TF-IDF Vectorizer**: Converts captions and user input into weighted word vectors.
3. **Similarity Matching Engine**: Computes cosine similarity and ranks memes.
4. **Image Editor**: Allows cropping, caption removal, overlaying, and adding new text.
5. **GUI (Tkinter)**: Provides an intuitive interface for interaction.

![Architecture](architecture_diagram.png) *(Optional)*

---

## 📁 Dataset

The system uses the **[6,992 Labeled Meme Images Dataset from Kaggle](https://www.kaggle.com/datasets/surajkum1198/memes-dataset)** which contains real-world memes with existing captions. These captions are used to find semantically relevant matches.

You must download and place the dataset manually.

### 📥 Download Instructions

1. Download the dataset from Kaggle:  
   https://www.kaggle.com/datasets/surajkum1198/memes-dataset
2. Extract it and rename the folder to `images`
3. Place the `images` folder inside the root of the cloned project repository.

---

## ⚙️ Installation

### 📦 Prerequisites

- Python 3.7+
- pip

### 🔧 Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/juni2003/AI-Meme-Generator.git
cd AI-Meme-Generator
```
2. **Create a Virtual Environment (Optional but Recommended)**

```bash

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. **Install Dependencies**

```bash

pip install -r requirements.txt
```
If requirements.txt is missing, install manually:

```bash
pip install scikit-learn pandas numpy Pillow nltk
```
4. **Download Dataset**

Follow instructions in Dataset



## 🧪 How to Use

1. **Launch the App**

   ```bash
   python main.py
   ```

2. **Input a Phrase or Punchline**

   Enter a phrase like _"when you forget your homework"_ and click **"Generate"**.

3. **Preview Meme**

   The most relevant meme will be shown.

4. **Edit Options**

   - Remove old caption using **white box**
   - Add new caption using text tool
   - Crop unwanted parts of the meme

5. **Save the Meme**

   Click **Export** to save the final meme locally.

---

## 📊 Evaluation

### 📈 Accuracy Metric

- **Top-5 Similarity Accuracy**:  
  Measures whether the correct meme is within the top 5 recommended results.

### ✅ Result

Achieved **99.21% Top-5 accuracy**, indicating high semantic match quality.

### 🧪 Testing Method

- 80% of the data used to fit the TF-IDF vectorizer  
- 20% used as test captions  
- Cosine similarity calculated for each test caption with all others  
- If correct meme is in top-5, it's counted as correct

---

## ⚠️ Limitations

- Only **text-based similarity** is used — no image content matching.
- The dataset contains **pre-captioned** memes, not clean templates.
- Very short user inputs may lead to low-quality vector representations.
- Deep learning models like BERT were avoided to maintain simplicity and offline capability.

---

## 🛠️ Tech Stack

| Component     | Library / Tool        |
|---------------|------------------------|
| Programming   | Python 3               |
| GUI           | Tkinter                |
| Image Editing | Pillow (PIL)           |
| NLP           | TF-IDF (scikit-learn)  |
| Data Handling | Pandas, NumPy          |

---

## 👨‍💻 Contributors

- **Junaid Satti** - Developer & Researcher  
  GitHub: [@juni2003](https://github.com/juni2003)

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙌 Acknowledgements

- Kaggle Dataset  
- Scikit-learn team for robust NLP tools  
- Open-source communities supporting Pillow and Tkinter

---

> 🚀 _“Creating memes shouldn't require design skills — just great ideas. Let AI handle the rest.”_
