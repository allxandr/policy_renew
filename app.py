from flask import Flask, request, render_template
from mlmodel import Model
app = Flask(__name__)

model = Model('model.ml')

months = ['Январь', 
            'Февраль', 
            'Март', 
            'Апрель', 
            'Май', 
            'Июнь', 
            'Июль', 
            'Август', 
            'Сентябрь', 
            'Октябрь', 
            'Ноябрь', 
            'Декабрь']


@app.route('/', )
def index():
    return render_template('index.html')


@app.route('/predict', methods = ['POST'])
def predict():
    #print(request.form)
    params = form_to_json(request.form)
    array = model.transform_parameters(params)
    if array is None:
        return {'error': 'parameters are not correct'}
    percentage = model.predict([array])[0][0]*100
    return render_template('index.html', percentage = f'{percentage:.2f}%')


@app.route('/api', methods = ['GET'])
def render_api():
    return render_template('index.html')


@app.route('/api/info', methods = ['POST'])
def api_info():
    #papametres = request.get_json()
    return {'api': {'version': '0.0', 'methods': ['info', 'predict'], 'model': 'GBClassifier'}}


@app.route('/api/predict', methods = ['POST'])
def api_predict():
    parametres = request.get_json()
    #print(parametres)
    array = model.transform_parameters(parametres)
    if array is None:
        return {'error': 'parameters are not correct'}
    percentage = model.predict([array])[0][0]*100
    return {'renew probability': percentage}

# if __name__ == '__main__':
    
#     app.jinja_env.auto_reload = True
#     app.config['TEMPLATES_AUTO_RELOAD'] = True
#     app.run()


def form_to_json(form):
    ''' transforms html form to json format for preditction '''
    params = {}
    for i in form:
        if form[i] == 'Да':
            params[i] = 1
        elif form[i] == 'Нет':
            params[i] = 0
        elif form[i] == 'Мужской':
            params[i] = 'M'
        elif form[i] == 'Женский':
            params[i] = 'F'
        elif form[i] == '1S - небольшая сумма убытка':
            params[i] = '1S'
        elif form[i] == '1L - большая сумма убытка':
            params[i] = '1L'
        elif form[i] == 'Не известно' and i == 'POLICY_CLM_GLT_N' or i == 'POLICY_CLM_N':
            params[i] = 'n/d'
        elif form[i] == 'Не известно' and i == 'POLICY_PRV_CLM_GLT_N' or i == 'POLICY_PRV_CLM_N':
            params[i] = 'N'
        else:
            params[i] = form[i]
    
    params['POLICY_SALES_CHANNEL'] = int(params['POLICY_SALES_CHANNEL'])
    params['POLICY_MIN_AGE'] = int(params['POLICY_MIN_AGE'])
    params['POLICY_MIN_DRIVING_EXPERIENCE'] = int(params['POLICY_MIN_DRIVING_EXPERIENCE'])
    params['VEHICLE_ENGINE_POWER'] = int(params['VEHICLE_ENGINE_POWER'])
    params['VEHICLE_SUM_INSURED'] = int(params['VEHICLE_SUM_INSURED'])
    params['POLICY_YEARS_RENEWED_N'] = int(params['POLICY_YEARS_RENEWED_N'])
    params['CLAIM_AVG_ACC_ST_PRD'] = int(params['CLAIM_AVG_ACC_ST_PRD'])
    params['POLICY_DEDUCT_VALUE'] = int(params['POLICY_DEDUCT_VALUE'])
    params['POLICY_PRICE_CHANGE'] = int(params['POLICY_PRICE_CHANGE'])
    params['POLICY_END_MONTH'] = months.index(params['POLICY_END_MONTH'])
    params['POLICY_BEGIN_MONTH'] = months.index(params['POLICY_BEGIN_MONTH'])
    return params


    
