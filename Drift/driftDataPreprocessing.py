from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def sequenceTrim(sequence):
    if len(sequence) > 450:
        return sequence[:450, :]
    else:
        # We might need to add more exceptions here
        assert len(sequence) == 450
        return sequence

def sequenceNormalize(sequences):
    scaler = MinMaxScaler()
    return scaler.fit(sequences)

def dataSplitShuffle(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
