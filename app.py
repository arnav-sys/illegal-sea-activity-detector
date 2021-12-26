import joblib
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd

model_file = './finalized_model.sav'
model = joblib.load(model_file)


# Lambda handler code

def lambda_handler(event, context):
    df = pd.read_json(event['body']) 
    std = MinMaxScaler()
    df[["lat","lon","vessel.mmsi","median_speed_knots","elevation_m","distance_from_shore_m","distance_from_port_m"]] = std.fit_transform(df[["lat","lon","vessel.mmsi","median_speed_knots","elevation_m","distance_from_shore_m","distance_from_port_m"]])

    encoder = LabelEncoder()
    labels = ["start","id","end","vessel.id","vessel.type","vessel.flag","vessel.name","regions.rfmo","start_day","end_day"]

    def encode_data():
        for label in labels:
            if label:
                df[label] = encoder.fit_transform(df[label])

    encode_data()


    prediction = model.predict(X)

    return {
        'statusCode': 200,
        'body': json.dumps(
            {
                "predicted_label": prediction,
            }
        )
    }
