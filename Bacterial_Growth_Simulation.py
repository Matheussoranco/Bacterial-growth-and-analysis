import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

class BacterialCultureClassifier:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.feature_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None

    def preprocess_data(self, X, y=None, training=True):
        """
        Preprocess the input data with advanced techniques
        """
        if training:
            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X)

            # Encode labels
            if y is not None:
                y_encoded = self.label_encoder.fit_transform(y)

                # Handle class imbalance using SMOTE
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X_scaled, y_encoded)

                # Feature selection using Random Forest importance
                self.feature_selector = SelectFromModel(
                    RandomForestClassifier(n_estimators=100, random_state=42)
                )
                self.feature_selector.fit(X_resampled, y_resampled)
                X_selected = self.feature_selector.transform(X_resampled)

                # Dimensional reduction using PCA
                self.pca = PCA(n_components=0.95)  # Preserve 95% of variance
                X_reduced = self.pca.fit_transform(X_selected)

                return X_reduced, y_resampled

        else:
            X_scaled = self.feature_scaler.transform(X)
            X_selected = self.feature_selector.transform(X_scaled)
            X_reduced = self.pca.transform(X_selected)
            return X_reduced

    def train_models(self, X, y):
        """
        Train multiple advanced models with hyperparameter tuning
        """
        X_processed, y_processed = self.preprocess_data(X, y, training=True)

        # Define models with hyperparameter grids
        models = {
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'xgboost': {
                'model': XGBClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'lightgbm': {
                'model': lgb.LGBMClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'num_leaves': [31, 63, 127]
                }
            },
            'svm': {
                'model': SVC(random_state=42),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'poly'],
                    'gamma': ['scale', 'auto', 0.1, 0.01]
                }
            }
        }

        # Train and tune each model
        for name, model_info in models.items():
            print(f"Training {name}...")
            grid_search = GridSearchCV(
                model_info['model'],
                model_info['params'],
                cv=5,
                scoring='f1_weighted',
                n_jobs=-1
            )
            grid_search.fit(X_processed, y_processed)
            self.models[name] = grid_search

            print(f"Best parameters for {name}: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        # Select the best model
        best_score = 0
        for name, model in self.models.items():
            if model.best_score_ > best_score:
                best_score = model.best_score_
                self.best_model = name

        print(f"\nBest performing model: {self.best_model}")

    def predict(self, X):
        """
        Make predictions using the best model
        """
        X_processed = self.preprocess_data(X, training=False)
        predictions = self.models[self.best_model].predict(X_processed)
        return self.label_encoder.inverse_transform(predictions)

    def predict_proba(self, X):
        """
        Get probability predictions for each class
        """
        X_processed = self.preprocess_data(X, training=False)
        return self.models[self.best_model].predict_proba(X_processed)

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data
        """
        X_processed = self.preprocess_data(X_test, training=False)
        y_pred = self.models[self.best_model].predict(X_processed)

        print("Classification Report:")
        print(classification_report(y_test, self.label_encoder.inverse_transform(y_pred)))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, self.label_encoder.inverse_transform(y_pred)))

    def save_model(self, filepath):
        """
        Save the trained model and preprocessors
        """
        model_data = {
            'models': self.models,
            'best_model': self.best_model,
            'feature_scaler': self.feature_scaler,
            'label_encoder': self.label_encoder,
            'feature_selector': self.feature_selector,
            'pca': self.pca
        }
        joblib.dump(model_data, filepath)

    @classmethod
    def load_model(cls, filepath):
        """
        Load a trained model
        """
        model_data = joblib.load(filepath)
        classifier = cls()
        classifier.models = model_data['models']
        classifier.best_model = model_data['best_model']
        classifier.feature_scaler = model_data['feature_scaler']
        classifier.label_encoder = model_data['label_encoder']
        classifier.feature_selector = model_data['feature_selector']
        classifier.pca = model_data['pca']
        return classifier

# Example usage
if __name__ == "__main__":
    # Sample data preparation (replace with your actual data)
    # Features should include relevant bacterial characteristics like:
    # - Growth rate
    # - Temperature tolerance
    # - pH tolerance
    # - Oxygen requirements
    # - Nutrient requirements
    # - Colony morphology
    # - Gram staining results
    # - Antibiotic resistance patterns
    # - Metabolic indicators
    # - Genetic markers

    # Example data structure
    data = {
        'growth_rate': np.random.normal(0.5, 0.1, 1000),
        'temp_tolerance': np.random.uniform(20, 45, 1000),
        'ph_tolerance': np.random.uniform(4, 9, 1000),
        'oxygen_req': np.random.choice(['aerobic', 'anaerobic', 'facultative'], 1000),
        'colony_size': np.random.uniform(0.1, 5.0, 1000),
        'gram_stain': np.random.choice(['positive', 'negative'], 1000),
        'antibiotic_resistance': np.random.randint(0, 5, 1000),
        'metabolic_activity': np.random.uniform(0.1, 1.0, 1000)
    }

    # Convert categorical variables to numeric
    df = pd.DataFrame(data)
    df = pd.get_dummies(df, columns=['oxygen_req', 'gram_stain'])

    # Generate sample labels (bacterial species)
    species = ['E.coli', 'S.aureus', 'P.aeruginosa', 'B.subtilis']
    y = np.random.choice(species, 1000)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42
    )

    # Create and train the classifier
    classifier = BacterialCultureClassifier()
    classifier.train_models(X_train, y_train)

    # Evaluate the model
    print("\nModel Evaluation:")
    classifier.evaluate(X_test, y_test)

    # Save the model
    classifier.save_model('bacterial_classifier.joblib')

    # Make predictions on new data
    new_sample = X_test.iloc[:5]
    predictions = classifier.predict(new_sample)
    probabilities = classifier.predict_proba(new_sample)

    print("\nSample Predictions:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        print(f"Sample {i+1}:")
        print(f"Predicted species: {pred}")
        print(f"Probability distribution: {dict(zip(classifier.label_encoder.classes_, prob))}\n")