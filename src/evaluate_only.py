# Evaluation-only runner used when MLflow is not installed.
# The MLflow version of the full workflow is in src/mlflow_pipeline.py.
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/'data/raw/Bike_Rental_Demand_Data.csv').drop_duplicates()
df['Date']=pd.to_datetime(df['Date'],dayfirst=True,errors='coerce'); df['Time']=pd.to_datetime(df['Time'],format='%H:%M',errors='coerce')
df['Year']=df['Date'].dt.year; df['Month']=df['Date'].dt.month; df['Day']=df['Date'].dt.day; df['DayOfWeek_Num']=df['Date'].dt.dayofweek
df['Hour']=df['Time'].dt.hour.fillna(0).astype(int); df['Is_Weekend']=(df['DayOfWeek_Num']>=5).astype(int)
df['Month_Sin']=np.sin(2*np.pi*df['Month']/12); df['Month_Cos']=np.cos(2*np.pi*df['Month']/12); df['Hour_Sin']=np.sin(2*np.pi*df['Hour']/24); df['Hour_Cos']=np.cos(2*np.pi*df['Hour']/24)
features=['DayOfWeek','Season','Weather','Temperature','Humidity','WindSpeed','RentalDuration','UserType','PaymentMethod','Cancelled','Year','Month','Day','DayOfWeek_Num','Hour','Is_Weekend','Month_Sin','Month_Cos','Hour_Sin','Hour_Cos']
X=df[features]; y=df['RentalCount']; Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
cat=X.select_dtypes(include='object').columns.tolist(); num=[c for c in features if c not in cat]
pre=ColumnTransformer([('num',SimpleImputer(strategy='median'),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat)])
models={'Linear Regression':LinearRegression(),'Ridge Regression':Ridge(alpha=1.0),'Decision Tree':DecisionTreeRegressor(random_state=42),'Random Forest':RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=-1),'Gradient Boosting':GradientBoostingRegressor(n_estimators=100,random_state=42),'SVR':SVR()}
rows=[]
for name,m in models.items():
    steps=[('preprocessor',pre)];
    if name=='SVR': steps.append(('scaler',StandardScaler(with_mean=False)))
    steps.append(('model',m)); pipe=Pipeline(steps); pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); mse=mean_squared_error(yte,pred)
    rows.append([name,mean_absolute_error(yte,pred),mse,np.sqrt(mse),r2_score(yte,pred)])
res=pd.DataFrame(rows,columns=['Model','MAE','MSE','RMSE','R2']).sort_values('R2',ascending=False); res.to_csv(ROOT/'results.csv',index=False); print(res.to_string(index=False))
