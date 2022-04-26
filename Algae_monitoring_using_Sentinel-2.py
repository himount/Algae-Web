import geopandas
import numpy as np
import folium, os, joblib
import pyalgae_ai.analysis as AiAnal
import pyalgae_ai.plotting as AiPlot
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import GridSearchCV
from hydroeval import evaluator, nse, kge

print(os.getcwd())
yjd_gdf = geopandas.read_file('./data/algae_obs.shp')
yjd_gdf['lon'] = yjd_gdf.geometry.apply(lambda p: p.x)
yjd_gdf['lat'] = yjd_gdf.geometry.apply(lambda p: p.y)

lat = yjd_gdf['lat'].mean()
lon = yjd_gdf['lon'].mean()

m = folium.Map([lat,lon],zoom_start=13)

for i in yjd_gdf.index:
    sub_lat = yjd_gdf.loc[i,'lat']
    sub_lon = yjd_gdf.loc[i,'lon']

    if (np.isnan(sub_lat) == False) & (np.isnan(sub_lat) == False) :
        folium.CircleMarker([sub_lat,sub_lon], icon_size=(1,1), radius=2, color="crimson", fill=True).add_to(m)
#print(m)
raw_obs = "./data/20220224_sentinels_and_algae_obs.csv"
algae =AiAnal.Machine_Learning(raw_obs, [1,6,20])
input_sentinel, label_algae = algae.preprocessing()
#print(label_algae)

pair_plot1 = pd.concat([label_algae, input_sentinel[list(input_sentinel)[0:3]]], axis=1)
pair_plot2 = pd.concat([label_algae, input_sentinel[list(input_sentinel)[3:6]]], axis=1)
pair_plot3 = pd.concat([label_algae, input_sentinel[list(input_sentinel)[6:9]]], axis=1)
pair_plot4 = pd.concat([label_algae, input_sentinel[list(input_sentinel)[9:12]]], axis=1)


plot1 = sns.pairplot(pair_plot1, corner=True)
plot1.fig.suptitle(" The correlation between Algae Observation (1.total_chla, 2.Green_Algae, 3.Bluegreen, 4.Diatoms, 5.Cryptophyta) VS Sentinel-2 (B1, B2, B3) ", size = 18)
#plt.figure(figsize=(5,5))
plt.show()

# plot2 = sns.pairplot(pair_plot2, corner=True)
# plot2.fig.suptitle(" The correlation between Algae Observation (1.total_chla, 2.Green_Algae, 3.Bluegreen, 4.Diatoms, 5.Cryptophyta) VS Sentinel-2 (B4, B5, B6) ", size = 18)
# plt.show()

# plot3 = sns.pairplot(pair_plot3, corner=True)
# plot3.fig.suptitle(" The correlation between Algae Observation (1.total_chla, 2.Green_Algae, 3.Bluegreen, 4.Diatoms, 5.Cryptophyta) VS Sentinel-2 (B7, B8, B8A) ", size = 18)
# plt.show()

# plot4 =sns.pairplot(pair_plot4, corner=True)
# plot4.fig.suptitle(" The correlation between Algae Observation (1.total_chla, 2.Green_Algae, 3.Bluegreen, 4.Diatoms, 5.Cryptophyta) VS Sentinel-2 (B9, B11, B12) ", size = 18)
# plt.show()

select_columns = [['B1 B2 B3'], ['B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B11 B12 AT CLOUD'], ['B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B11 B12']]
# %time
results = AiAnal.algae_monitor(input_sentinel, label_algae, select_columns, model_list=["RF"], trainSize_rate=0.8, n_estimators=200, random_state=42)

# print(len(results), len(results[0]))

# 전체 결과 확인
for i in range(len(results)):
    score_train, score_test = AiAnal.performance_test("NSE", results[i])
    f_result = '"{}" and "{}"의 결과: score_train={}, score_test={}'.format(' '.join(list(results[i][1])), results[i][2].name, score_train, score_test)
    #print(f_result)

# 가장 좋은 결과
score_train, score_test = AiAnal.performance_test("NSE", results[6])
#print(score_train, score_test)

param_grid = [
    {'n_estimators':[50, 100, 200], 'max_features':[2, 4, 6, 8]},
    {'bootstrap': [False]}
]

grid_search = GridSearchCV(results[1][0], param_grid, cv=5, 
                           scoring='r2',
                          return_train_score=True)

# %time
grid_search.fit(results[6][1], results[6][2])
print(grid_search.best_params_)

Y_test = results[6][4]
Y_test_predict = results[6][6]

AiPlot.linear_regression(Y_test, Y_test_predict, "NSE", score_test[0])

joblib.dump(results[6][0], "rf_model.pkl")

rf_model_loaded = joblib.load("rf_model.pkl")

Y_test_predict_loaded = rf_model_loaded.predict(results[6][3])
print(Y_test_predict_loaded)

nse_score_test_loaded = evaluator(nse, results[1][4], Y_test_predict_loaded, axis=1)
print(nse_score_test_loaded[0])