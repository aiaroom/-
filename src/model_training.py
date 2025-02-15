from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import warnings
import pandas

warnings.filterwarnings('ignore')

def train_model(X, y):
    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Балансировка классов с использованием SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # Подбор гиперпараметров для логистической регрессии
    param_grid_lr = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],
        'penalty': ['l1', 'l2']
    }
    
    grid_search_lr = GridSearchCV(LogisticRegression(max_iter=1000), param_grid_lr, cv=5, scoring='accuracy')
    grid_search_lr.fit(X_resampled, y_resampled)
    best_lr_model = grid_search_lr.best_estimator_
    
    # Обучение случайного леса
    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='accuracy')
    grid_search_rf.fit(X_resampled, y_resampled)
    best_rf_model = grid_search_rf.best_estimator_
    

    
    # Предсказание на тестовой выборке с использованием лучших моделей
    y_pred_lr = best_lr_model.predict(X_test)
    y_pred_rf = best_rf_model.predict(X_test)

    
    # Оценка качества моделей
    print("Точность (Логистическая регрессия):", accuracy_score(y_test, y_pred_lr))
    print("Отчет классификации (Логистическая регрессия):\n", classification_report(y_test, y_pred_lr))
    
    print("Точность (Случайный лес):", accuracy_score(y_test, y_pred_rf))
    print("Отчет классификации (Случайный лес):\n", classification_report(y_test, y_pred_rf))
    
    
    return best_lr_model, best_rf_model

if __name__ == "__main__":
    df = pd.read_csv('data/titanic.csv')
    from data_preprocessing import preprocess_data
    X, y = preprocess_data(df)
    print("Обучение модели...")
    lr_model, rf_model, gb_model = train_model(X, y)