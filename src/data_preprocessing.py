import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(file_path):
    # Загрузка данных из CSV файла
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Удаление ненужных столбцов
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    
    # Обработка пропущенных значений
    df['Age'] = df['Age'].fillna(df['Age'].median())  # Заполнение пропущенных значений возраста медианным значением
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # Заполнение пропущенных значений порта посадки модальным значением
    
    # Создание новых признаков
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 65, 100], labels=['Child', 'Teenager', 'Adult', 'Senior'])  # Создание признака "AgeGroup"
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1  # Создание признака "FamilySize"
    df['IsAlone'] = df['FamilySize'].apply(lambda x: 1 if x == 1 else 0)  # Создание признака "IsAlone"
    
    # Кодирование категориальных признаков с помощью get_dummies
    df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'AgeGroup'])
    
    # Выборка признаков
    features = [
        'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone',
        'Sex_male', 'Sex_female', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
        'AgeGroup_Child', 'AgeGroup_Teenager', 'AgeGroup_Adult', 'AgeGroup_Senior'
    ]
    X = df[features]
    y = df['Survived']
    
    # Масштабирование признаков
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

if __name__ == "__main__":
    df = load_data('data/titanic.csv')
    X, y = preprocess_data(df)
    print("Предобработка данных завершена.")