import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df):
    # Удаление ненужных столбцов
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    
    # Обработка пропущенных значений
    df['Age'] = df['Age'].fillna(df['Age'].median())  # Заполнение пропущенных значений возраста медианным значением
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # Заполнение пропущенных значений порта посадки модальным значением
    
    # Кодирование категориальных признаков
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'])
    
    # Описательная статистика
    print(df.describe())
    
    # Корреляционная матрица
    corr_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.show()
    
    # Гистограммы признаков
    df.hist(bins=50, figsize=(20,15))
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv('data/titanic.csv')
    print("Выполнение анализа данных...")
    perform_eda(df)