import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle

class Model:
    def __init__(self, file_name = None):
       
        self.x_train = None
        self.x_val = None
        self.y_train = None
        self.y_val = None
        self.parametres = []
        self.model = None
        self.data = pd.DataFrame()
        if file_name:
            self.model = pickle.load(open(file_name, 'rb'))
        

    def load_data(self, file_name):
        ''' loads data from csv file '''
        self.data = pd.read_csv(file_name, sep = ';')
        self.parametres = self.data.columns

    def delete_anomaly(self):
        ''' deletes anomaly like 2015 of experience '''
        assert not self.data.empty, 'No data loaded'
        self.data.drop(self.data.index[self.data['POLICY_MIN_DRIVING_EXPERIENCE'] > 150], inplace = True)

    def delete_usesless_columns(self):
        ''' deletes useless columns like DATA_TYPE and POLICY_ID '''
        assert not self.data.empty, 'No data loaded'
        drops = [
                'DATA_TYPE',
                'POLICY_ID', 
                'POLICY_SALES_CHANNEL_GROUP', 
                'POLICY_INTERMEDIARY', 
                'VEHICLE_MAKE',
                 ]
        self.data.drop(labels = drops, axis = 1, inplace = True)

    def delete_test_rows(self):
        ''' deletes rows with test data '''
        assert  not self.data.empty, 'No data loaded'
        self.data.drop(self.data.index[self.data['DATA_TYPE'] == 'TEST '], inplace = True)

    def make_dummies(self):
        ''' makes dummies '''
        assert not self.data.empty, 'No data loaded'
        # transform months
        self.data['POLICY_BEGIN_MONTH'] = self.data['POLICY_END_MONTH']-self.data['POLICY_BEGIN_MONTH']
        self.data['POLICY_BEGIN_MONTH'] = self.data['POLICY_BEGIN_MONTH'].map(lambda x: 'N' if (x!=0 and x!=-1) else x)
        self.data.drop('POLICY_END_MONTH', axis = 1, inplace = True)

        #transform vehicle model
        g = self.data.groupby('VEHICLE_MODEL')
        self.data.loc[g['VEHICLE_MODEL'].transform(lambda x: len(x) <= 601).astype(bool), 'VEHICLE_MODEL'] = 'RARE'

        #transform client region
        g = self.data.groupby('CLIENT_REGISTRATION_REGION')
        self.data.loc[g['CLIENT_REGISTRATION_REGION'].transform(lambda x: len(x) <= 1000).astype(bool), 'CLIENT_REGISTRATION_REGION'] = 'RARE'

        # edit N years to sth more than 10
        self.data['POLICY_YEARS_RENEWED_N'] = self.data['POLICY_YEARS_RENEWED_N'].map(lambda x: 15 if x=='N' else x)
        #get dummies
        dummies_columns = [
                            'POLICY_BEGIN_MONTH', 
                            'POLICY_BRANCH', 
                            'INSURER_GENDER', 
                            'POLICY_CLM_GLT_N', 
                            'POLICY_PRV_CLM_N', 
                            'POLICY_PRV_CLM_GLT_N', 
                            'CLIENT_REGISTRATION_REGION', 
                            'POLICY_CLM_N', 
                            'VEHICLE_MODEL', 
                            ]
        self.data = pd.get_dummies(self.data, columns = dummies_columns)


    def transform_data(self):
        assert not self.data.empty, 'No data loaded'
        ''' makes all the needed transformations with data'''
        self.delete_anomaly()
        self.delete_test_rows()
        self.delete_usesless_columns()        
        self.make_dummies()

    def get_data_parts(self):
        ''' makes suitable for training data parts '''
        assert  not self.data.empty, 'No data loaded'
        y_train = self.data['POLICY_IS_RENEWED']
        x_train = self.data.drop('POLICY_IS_RENEWED', axis = 1)
        # scaler = MinMaxScaler()
        # x_train = scaler.fit_transform(x_train)          
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x_train, y_train,  test_size=0.2)

    def train(self, n_estimators, learning_rate, max_features, max_depth):
        ''' creates the model and trains it'''
        assert not self.data.empty, 'No data loaded'
        if not self.x_train or not self.x_val:
            self.get_data_parts()
        self.model = GradientBoostingClassifier(n_estimators=n_estimators, 
            learning_rate=learning_rate, 
            max_features=max_features, 
            max_depth=max_depth)
        self.model.fit(self.x_train, self.y_train)

        print('training score: ', self.model.score(self.x_train, self.y_train))
        print('validating score: ', self.model.score(self.x_val, self.y_val))

    def predict(self, x):
        assert self.model, 'No ML model created'
        return self.model.predict_proba(x)

    def save(self, file_name):
        assert self.model, 'No ML model created'
        pickle.dump(self.model, open(file_name, 'wb'))

    def transform_months(self, begin, end, mask=None):
        if begin == None or end == None:
            return None

        if begin-end == mask:
            return 1
        if not mask:
            return 1
        return 0

    def transform_attribute(self, attribute, mask=None):
        if attribute == None:
            return None
        if attribute == mask:
            return 1
        if not mask:
            return 1
        return 0


    def transform_parameters(self, params):
        parameters = [
                        params.get('POLICY_SALES_CHANNEL', None),
                        params.get('POLICY_MIN_AGE', None),
                        params.get('POLICY_MIN_DRIVING_EXPERIENCE', None),
                        params.get('VEHICLE_ENGINE_POWER', None),
                        params.get('VEHICLE_IN_CREDIT', None),
                        params.get('VEHICLE_SUM_INSURED', None),
                        params.get('CLIENT_HAS_DAGO', None),
                        params.get('CLIENT_HAS_OSAGO', None),
                        params.get('POLICY_COURT_SIGN', None),
                        params.get('CLAIM_AVG_ACC_ST_PRD', None),
                        params.get('POLICY_HAS_COMPLAINTS', None),
                        params.get('POLICY_YEARS_RENEWED_N', None),
                        params.get('POLICY_DEDUCT_VALUE', None),
                        params.get('POLICY_PRICE_CHANGE', None),
                        self.transform_months(params.get('POLICY_BEGIN_MONTH', None), 
                                params.get('POLICY_END_MONTH', None), -1), # 'POLICY_BEGIN_MONTH_-1'
                        self.transform_months(params.get('POLICY_BEGIN_MONTH', None), 
                                params.get('POLICY_END_MONTH', None), 0), # 'POLICY_BEGIN_MONTH_0'
                        self.transform_months(params.get('POLICY_BEGIN_MONTH', None), 
                                params.get('POLICY_END_MONTH', None)), # 'POLICY_BEGIN_MONTH_N'
                        self.transform_attribute(params.get('POLICY_BRANCH', None), 
                                'Москва'), #'POLICY_BRANCH_Москва'
                        self.transform_attribute(params.get('POLICY_BRANCH', None), 
                                'Санкт-Петербург'), #'POLICY_BRANCH_Санкт-Петербург'
                        self.transform_attribute(params.get('INSURER_GENDER', None), 
                                'F'), #INSURER_GENDER_F
                        self.transform_attribute(params.get('INSURER_GENDER', None), 
                                'M'), # INSURER_GENDER_M
                        self.transform_attribute(params.get('POLICY_CLM_GLT_N', None), 
                                '0'), # POLICY_CLM_GLT_N_0
                        self.transform_attribute(params.get('POLICY_CLM_GLT_N', None), 
                                '1L'), # POLICY_CLM_GLT_N_1L
                        self.transform_attribute(params.get('POLICY_CLM_GLT_N', None), 
                                '1S'), # POLICY_CLM_GLT_N_1S
                        self.transform_attribute(params.get('POLICY_CLM_GLT_N', None), 
                                '2'), # POLICY_CLM_GLT_N_2
                        self.transform_attribute(params.get('POLICY_CLM_GLT_N', None), 
                                '3'), # POLICY_CLM_GLT_N_3
                        self.transform_attribute(params.get('POLICY_CLM_GLT_N', None), 
                                '4+'), # POLICY_CLM_GLT_N_4+
                        self.transform_attribute(params.get('POLICY_CLM_GLT_N', None), 
                                'n/d'), # POLICY_CLM_GLT_N_n/d
                        self.transform_attribute(params.get('POLICY_PRV_CLM_N', None), 
                                '0'), # POLICY_PRV_CLM_N_0
                        self.transform_attribute(params.get('POLICY_PRV_CLM_N', None), 
                                '1L'), # POLICY_PRV_CLM_N_1L
                        self.transform_attribute(params.get('POLICY_PRV_CLM_N', None), 
                                '1S'), # POLICY_PRV_CLM_N_1S
                        self.transform_attribute(params.get('POLICY_PRV_CLM_N', None), 
                                '2'), # POLICY_PRV_CLM_N_2
                        self.transform_attribute(params.get('POLICY_PRV_CLM_N', None), 
                                '3'), # POLICY_PRV_CLM_N_3
                        self.transform_attribute(params.get('POLICY_PRV_CLM_N', None), 
                                '4+'), # POLICY_PRV_CLM_N_4+
                        self.transform_attribute(params.get('POLICY_PRV_CLM_N', None), 
                                'N'), # POLICY_PRV_CLM_N_N
                        self.transform_attribute(params.get('POLICY_PRV_CLM_GLT_N', None), 
                                '0'), # POLICY_PRV_CLM_GLT_N_0
                        self.transform_attribute(params.get('POLICY_PRV_CLM_GLT_N', None), 
                                '1L'), # POLICY_PRV_CLM_GLT_N_1L
                        self.transform_attribute(params.get('POLICY_PRV_CLM_GLT_N', None), 
                                '1S'), # POLICY_PRV_CLM_GLT_N_1S
                        self.transform_attribute(params.get('POLICY_PRV_CLM_GLT_N', None), 
                                '2'), # POLICY_PRV_CLM_GLT_N_2
                        self.transform_attribute(params.get('POLICY_PRV_CLM_GLT_N', None), 
                                '3'), # POLICY_PRV_CLM_GLT_N_3
                        self.transform_attribute(params.get('POLICY_PRV_CLM_GLT_N', None), 
                                '4+'), # POLICY_PRV_CLM_GLT_N_4+
                        self.transform_attribute(params.get('POLICY_PRV_CLM_GLT_N', None), 
                                'N'), # POLICY_PRV_CLM_GLT_N_N
                        self.transform_attribute(params.get('CLIENT_REGISTRATION_REGION', None)), # CLIENT_REGISTRATION_REGION_N
                        self.transform_attribute(params.get('CLIENT_REGISTRATION_REGION', None), 
                                'Ленинградская'), # CLIENT_REGISTRATION_REGION_Ленинградская
                        self.transform_attribute(params.get('CLIENT_REGISTRATION_REGION', None), 
                                'Москва'), # CLIENT_REGISTRATION_REGION_Москва'
                        self.transform_attribute(params.get('CLIENT_REGISTRATION_REGION', None), 
                                'Московская'), # CLIENT_REGISTRATION_REGION_Московская
                        self.transform_attribute(params.get('CLIENT_REGISTRATION_REGION', None), 
                                'Санкт-Петербург'), # CLIENT_REGISTRATION_REGION_Санкт-Петербург
                        self.transform_attribute(params.get('POLICY_CLM_N', None), 
                                '0'), # POLICY_CLM_N_0
                        self.transform_attribute(params.get('POLICY_CLM_N', None), 
                                '1L'), # POLICY_CLM_N_1L
                        self.transform_attribute(params.get('POLICY_CLM_N', None), 
                                '1S'), # POLICY_CLM_N_1S
                        self.transform_attribute(params.get('POLICY_CLM_N', None), 
                                '2'), # POLICY_CLM_N_2
                        self.transform_attribute(params.get('POLICY_CLM_N', None), 
                                '3'), # POLICY_CLM_N_3
                        self.transform_attribute(params.get('POLICY_CLM_N', None), 
                                '4+'), # POLICY_CLM_N_4+
                        self.transform_attribute(params.get('POLICY_CLM_N', None), 
                                'n/d'), # POLICY_CLM_N_n/d
                        self.transform_attribute(params.get('VEHICLE_MODEL', None), 
                                'ASX'), # POLICY_CLM_N_ASX
                        self.transform_attribute(params.get('VEHICLE_MODEL', None), 
                                'CR_V'), # POLICY_CLM_N_CR_V
                        self.transform_attribute(params.get('VEHICLE_MODEL', None), 
                                'Ceed'), # POLICY_CLM_N_Ceed
                        self.transform_attribute(params.get('VEHICLE_MODEL', None), 
                                'Duster'), # POLICY_CLM_N_Duster
                        self.transform_attribute(params.get('VEHICLE_MODEL', None), 
                                'Fabia'), # POLICY_CLM_N_Fabia
                        self.transform_attribute(params.get('VEHICLE_MODEL', None), 
                                'Fluence'), # POLICY_CLM_N_Fluence
                        self.transform_attribute(params.get('VEHICLE_MODEL', None), 
                                'Focus'), # POLICY_CLM_N_Focus
                        self.transform_attribute(params.get('VEHICLE_MODEL', None), 
                                'Grand Vitara'), # POLICY_CLM_N_Grand Vitara
                        self.transform_attribute(params.get('VEHICLE_MODEL', None), 
                                'Juke'), # POLICY_CLM_N_Juke
                        self.transform_attribute(params.get('VEHICLE_MODEL', None), 
                                'Kuga'), # POLICY_CLM_N_Kuga
                        self.transform_attribute(params.get('VEHICLE_MODEL', None), 
                                'Land Cruiser Prado'), # POLICY_CLM_N_Land Cruiser Prado
                        self.transform_attribute(params.get('VEHICLE_MODEL', None), 
                                'Logan'), # POLICY_CLM_N_Logan
                        self.transform_attribute(params.get('VEHICLE_MODEL', None), 
                                'Megane'), # POLICY_CLM_N_Megane
                        self.transform_attribute(params.get('VEHICLE_MODEL', None), 
                                'Octavia'), # POLICY_CLM_N_Octavia
                        self.transform_attribute(params.get('VEHICLE_MODEL', None), 
                                'Outlander'), # POLICY_CLM_N_Outlander
                        self.transform_attribute(params.get('VEHICLE_MODEL', None), 
                                'Qashqai'), # POLICY_CLM_N_Qashqai
                        self.transform_attribute(params.get('VEHICLE_MODEL', None), ), # POLICY_CLM_N_Rare
                        self.transform_attribute(params.get('VEHICLE_MODEL', None), 
                                'RAV4'), # POLICY_CLM_N_RAV4
                        self.transform_attribute(params.get('VEHICLE_MODEL', None), 
                                'Rapid'), # POLICY_CLM_N_Rapid
                        self.transform_attribute(params.get('VEHICLE_MODEL', None), 
                                'Rio'), # POLICY_CLM_N_Rio
                        self.transform_attribute(params.get('VEHICLE_MODEL', None), 
                                'Sandero'), # POLICY_CLM_N_Sandero
                        self.transform_attribute(params.get('VEHICLE_MODEL', None), 
                                'Santa Fe'), # POLICY_CLM_N_Santa Fe
                        self.transform_attribute(params.get('VEHICLE_MODEL', None), 
                                'Solaris'), # POLICY_CLM_N_Solaris
                        self.transform_attribute(params.get('VEHICLE_MODEL', None), 
                                'Sorento'), # POLICY_CLM_N_Sorento
                        self.transform_attribute(params.get('VEHICLE_MODEL', None), 
                                'Sportage'), # POLICY_CLM_N_Sportage
                        self.transform_attribute(params.get('VEHICLE_MODEL', None), 
                                'Tiguan'), # POLICY_CLM_N_Tiguan
                        self.transform_attribute(params.get('VEHICLE_MODEL', None), 
                                'X-Trail'), # POLICY_CLM_N_X-Trail
                        self.transform_attribute(params.get('VEHICLE_MODEL', None), 
                                'X1'), # POLICY_CLM_N_X1
                        self.transform_attribute(params.get('VEHICLE_MODEL', None), 
                                'X3'), # POLICY_CLM_N_X3
                        self.transform_attribute(params.get('VEHICLE_MODEL', None), 
                                'Yeti'), # POLICY_CLM_N_Yeti
                        self.transform_attribute(params.get('VEHICLE_MODEL', None), 
                                'ix35'), # POLICY_CLM_N_ix35
                    ]
        if  any(i == None for i in parameters):
            return None
        return np.array(parameters).astype(float)

