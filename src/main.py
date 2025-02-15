import pandas as pd
from data_preprocessing import preprocess_data
from model_training import train_model
from exploratory_data_analysis import perform_eda

def main():
    df = pd.read_csv('data/titanic.csv')
    
    print("Выполнение анализа данных...")
    perform_eda(df)
    
    print("Предобработка данных...")
    X, y = preprocess_data(df)
    
    print("Обучение модели...")
    lr_model, rf_model = train_model(X, y)
    
    print("Обучение модели завершено.")

if __name__ == "__main__":
    main()