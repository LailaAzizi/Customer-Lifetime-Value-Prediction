import pandas as pd
from sklearn.linear_model import LinearRegression
from data_preprocessing import load_data, preprocess_data

def predict_clv(df):
    # Simple linear regression example
    X = df[['total_purchases', 'average_purchase_value']]
    y = df['lifetime_value']
    model = LinearRegression()
    model.fit(X, y)
    df['predicted_clv'] = model.predict(X)
    return df

if __name__ == "__main__":
    df = load_data('customer_data.db')
    df = preprocess_data(df)
    df_with_clv = predict_clv(df)
    print(df_with_clv[['customer_id', 'predicted_clv']])
