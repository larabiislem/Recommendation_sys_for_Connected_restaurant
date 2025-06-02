# Recommendation System for Connected Restaurant

This repository contains a content-based recommendation system designed for a connected restaurant environment. The system suggests similar dishes to customers based on dish ingredients and category using machine learning techniques.

## Features

- **Content-Based Filtering:** Recommends dishes that are similar in ingredients and category.
- **Data Preprocessing:** Handles data cleaning, feature engineering, and transformation.
- **TF-IDF Vectorization:** Converts textual features into numeric vectors for similarity calculations.
- **Cosine Similarity:** Measures similarity between dishes using their feature vectors.
- **Easy Customization:** The system can be adapted to other datasets or extended with collaborative filtering.

## How It Works

1. **Data Preparation:**  
   The dataset (`data.csv`) includes columns for dish ID, name, category (`C_Type`), vegetarian/non-vegetarian flag (`Veg_Non`), and a description of ingredients.
   - The description column is renamed to "Ingredient".
   - All punctuation is removed from the ingredient descriptions.
   - A new "features" column is created by combining the category and ingredients.

2. **Vectorization:**  
   - The `TfidfVectorizer` from scikit-learn converts dish features into a TF-IDF matrix, capturing the importance of each word.
   - Stop words (common words like "and", "or") are removed automatically.

3. **Similarity Calculation:**  
   - Cosine similarity is computed between all dish vectors using `linear_kernel`.
   - This results in a matrix showing how similar each dish is to every other dish.

4. **Recommendation Function:**  
   - The function `get_recommendations()` takes a dish name as input and returns the top similar dishes based on cosine similarity.

## Example

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load and clean the data as shown in the notebook
# ...

# Create TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(food['features'])

# Compute cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get recommendations
def get_recommendations(name, cosine_sim=cosine_sim):
    # Implementation as in the notebook
    pass
```

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- scipy

Install dependencies with:
```bash
pip install pandas numpy scikit-learn scipy
```

## Usage

1. Place your dataset as `data.csv` in the working directory.
2. Run the notebook `sys_rec.ipynb` to preprocess data, train the model, and test recommendations.

## Dataset Format

| Food_ID | Name                  | C_Type        | Veg_Non | Ingredient                                  |
|---------|-----------------------|---------------|---------|----------------------------------------------|
| 1       | summer squash salad   | Healthy Food  | veg     | white balsamic vinegar lemon juice ...       |
| ...     | ...                   | ...           | ...     | ...                                          |

## Project Structure

- `sys_rec.ipynb` - Main Jupyter notebook containing all data processing and model code.
- `data.csv` - Dataset of dishes (not included in repo, add your own).

## License

This project is for educational and demonstration purposes. Feel free to use and adapt it for your own needs.

---

*Developed by [larabiislem](https://github.com/larabiislem)*
