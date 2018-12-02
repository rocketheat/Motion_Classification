import turicreate as tc

# Load sessions from preprocessed data
data = tc.SFrame('hapt_data.sframe')

# Load the model
model = tc.load_model('./mymodel.model')

walking_3_sec = data[(data['activity'] == 'laying') & (data['exp_id'] == 1)][1000:1150]
print(model.predict(walking_3_sec, output_frequency='per_window'))
