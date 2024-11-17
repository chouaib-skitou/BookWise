
# BookWise

**BookWise** is a machine learning project aimed at classifying books into various categories based on their descriptions. It uses advanced deep learning models to predict book genres, providing insights into textual data.

---

## 🚀 Features
- Classifies books into predefined categories.
- Processes large datasets efficiently.
- Generates detailed reports and analysis.
- Provides visualizations of results, including confusion matrices.

---

## 📁 Project Structure
```
BookWise/
├── data.py                        # Data processing scripts
├── main.py                        # Main script for training and evaluation
├── BooksDataSet.csv               # Sample dataset
├── history/                       # Archived files and intermediate results
├── models/                        # Trained models and configurations
├── report.txt                     # Final report summarizing the project
├── results/                       # Visualizations and evaluation outputs
├── README.md                      # Project documentation
```

---

## 🛠️ Technologies Used
- Python
- TensorFlow/Keras
- Pandas
- NumPy
- Matplotlib

---

## 📊 Dataset
- **Source**: `BooksDataSet.csv`
- **Description**: Contains book metadata such as titles, descriptions, and other attributes for classification tasks.

---

## 🏃‍♂️ Usage
### 1. Clone the Repository
```bash
git clone https://github.com/chouaib-skitou/BookWise.git
cd BookWise
```

### 2. Install Dependencies
Ensure you have Python installed. Install required packages:
```bash
pip install -r requirements.txt
```

### 3. Train the Model
Run the training script:
```bash
python main.py
```

### 4. Evaluate the Model
Generate metrics and visualize results:
```bash
python data.py
```

---

## 🌟 Features to Add
- Implement user-friendly API for predictions.
- Enhance dataset with additional metadata.
- Support for multilingual text classification.

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!  
Feel free to fork the repository and submit pull requests.

---

## 🙌 Acknowledgments
- Thanks to all open-source contributors.
- Special thanks to the [Polytech Paris-Saclay](https://www.polytech.universite-paris-saclay.fr/) team for their support.
