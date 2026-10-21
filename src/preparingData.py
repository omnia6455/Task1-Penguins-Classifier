import pandas as pd

# Read data and map categorical mapping
data = pd.read_csv('penguins.csv')
data['OriginLocation'] = data['OriginLocation'].map({'Torgersen': 0, 'Biscoe': 1, 'Dream': 2})

def get_train_test_split(data, class1, class2, feature1, feature2, train_size=30, seed=42):
    subset = data[data['Species'].isin([class1, class2])].copy()
    subset['Target'] = subset['Species'].map({class1: 1, class2: -1})

    training, testing = [], []
    for selectedClasses in (class1, class2):
        selectedClasssFt = subset[subset['Species'] == selectedClasses].copy()

        #Spliit the data into training and testing sets based on the specified train_size and available samples
        available_samples = len(selectedClasssFt)
        dynamic_train_size = min(train_size, int(available_samples * 0.6))

        train_part = selectedClasssFt.sample(n=dynamic_train_size, random_state=seed).copy()
        test_part = selectedClasssFt.drop(train_part.index).copy()

        #Handle missing values in the numerical columns by filling them with the mean of the training set
        numerical_cols = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass']
        for col in numerical_cols:
            train_mean = train_part[col].mean()
            train_part[col] = train_part[col].fillna(train_mean)
            test_part[col] = test_part[col].fillna(train_mean)

        #Handle missing values in the categorical column 'OriginLocation' by filling them with the mode of the training set
        if 'OriginLocation' in train_part.columns:
            train_mode = train_part['OriginLocation'].mode()[0]
            train_part['OriginLocation'] = train_part['OriginLocation'].fillna(train_mode)
            test_part['OriginLocation'] = test_part['OriginLocation'].fillna(train_mode)

        training.append(train_part)
        testing.append(test_part)

    trainData = pd.concat(training).sample(frac=1, random_state=seed).reset_index(drop=True)
    testData  = pd.concat(testing).sample(frac=1, random_state=seed).reset_index(drop=True)

    X_train = trainData[[feature1, feature2]].to_numpy(dtype=float)
    X_test  = testData[[feature1, feature2]].to_numpy(dtype=float)
    y_train = trainData['Target'].to_numpy(dtype=int)
    y_test  = testData['Target'].to_numpy(dtype=int)

    return X_train, y_train, X_test, y_test

# Testing the function
if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te = get_train_test_split(data, 'Adelie', 'Gentoo', 'CulmenDepth', 'FlipperLength')
    print("Train shape:", X_tr.shape, "Test shape:", X_te.shape)

