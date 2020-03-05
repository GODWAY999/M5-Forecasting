from function.DataProcess import gen_submission, load_data
from model.CrossValidation import CV



data = load_data('../data/test.csv')
my_model = CV.load_model()
prediction = my_model.predict(data)
gen_submission(prediction, file_name="prediction.csv", loc='../')
