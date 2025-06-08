
from prefunc import *
from funs_训练测试集95CI import *
plt.rcParams['font.family'] = 'Times New Roman'
from sklearn.utils import resample
from scipy.stats import norm
import statsmodels.api as sm


#----------------------------------------------------------------------------------
# 加载数据
df = pd.read_csv('C:/Users/75454/Desktop/毕业课题/机器学习风险预测/手稿/2.CRRT-AKI/18.12-23.10AADAKI1_3插补.csv', encoding="utf_8_sig")


#----------------------------------------------------------------------------------
# 特征和标签
X = df.drop(columns=['ID', 'CKMB','ALT','AST','UA', 'intraoperative_urine', 'aki_serious','aki1_3','natriuretic_peptide','CRRT','CRRT_time','ICU_time','death','MIN_urine', 'MAX_urine' ])
y = df['aki1_3'].astype(int)  

# 确保分类特征一致性 (假设存在一些分类特征列)
categorical_columns = X.select_dtypes(include=['object', 'category']).columns
# 将分类特征转换为类别编码，保证训练和测试集类别一致
for col in categorical_columns:
    X[col] = pd.Categorical(X[col])


# 切分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=45, test_size=0.2) #修改为最有random_state=i
# LightGBM 模型
clf = lgb.LGBMClassifier(max_depth=5, num_leaves=14, learning_rate=0.08, n_estimators=288, verbose=-1, force_col_wise=True) #修改为gbm.best_params
# 交叉验证
cv2 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  #10折交叉验证
shap_elimination = ShapRFECV(clf=clf, step=0.3, cv=cv2, scoring='roc_auc', n_jobs=5)
# 计算 SHAP 特征重要性
report = shap_elimination.fit_compute(X_train, y_train, check_additivity=False)
# 绘制性能图
performance_plot = shap_elimination.plot()
report[['num_features', 'train_metric_mean', 'val_metric_mean']]

R.ggsave(filename="rfe.pdf", plot=ref_curve(report), width=120, height=80, unit='mm')



#——————————————————————————————————————————————————————————————-SHAP图
clf = lgb.LGBMClassifier(max_depth=5, num_leaves=14, learning_rate=0.08, n_estimators=288, verbose=-1, force_col_wise=True) #修改
clf.fit(X_train,y_train)
explainer = shap.TreeExplainer(clf)
val_X=X
shap_values =  explainer.shap_values(val_X) 

shap.summary_plot(shap_values[1], X, max_display=20, show=True, plot_size=[4,6])
plt.savefig('importance_1_20.pdf', format="pdf", dpi=1000, bbox_inches="tight")
plt.figure(figsize=(10, 10))
shap.summary_plot(shap_values[1], X, plot_type='bar', max_display=20,show=True, plot_size=[3,6])
plt.savefig('importance_2_20.pdf', format="pdf", bbox_inches="tight")


clf = lgb.LGBMClassifier(max_depth=5, num_leaves=14, learning_rate=0.08, n_estimators=288, verbose=-1, force_col_wise=True) #修改
clf.fit(X_train,y_train)
explainer = shap.TreeExplainer(clf)
val_X=X
shap_values =  explainer.shap_values(val_X) 

shap.summary_plot(shap_values[1], X, max_display=10, show=True, plot_size=[4,6])
plt.savefig('importance_1_10.pdf', format="pdf", dpi=1000, bbox_inches="tight")
plt.figure(figsize=(10, 10))
shap.summary_plot(shap_values[1], X, plot_type='bar', max_display=10,show=True, plot_size=[3,6])
plt.savefig('importance_2_10.pdf', format="pdf", bbox_inches="tight")

#------------------------------------------------------------DCA+ROC_test与10FCV
R['source']('C:/Users/75454/Desktop/毕业课题/机器学习风险预测/数据&代码/Python代码等/ATAAD-AKI代码/dca_roc.r')
params={}
params["max_depth"] = 5
params["num_leaves"] = 14
params["learning_rate"] = 0.08
params["n_estimators"] = 288
params["verbose"] = -1
params["force_col_wise"] = True

df_val=pd.DataFrame({"y":y_train,"p":run_gbm( X_train, y_train,params,sp="p")[:,1],"g":"10FCV"})

model = lgb.LGBMClassifier(max_depth=5, num_leaves=14, learning_rate=0.08, n_estimators=288, verbose=-1, force_col_wise=True) #修改为gbm.best_params
model.fit(X_train,y_train)
explainer = shap.TreeExplainer(model)

df_test=pd.DataFrame({"y":y_test,"p": model.predict_proba(X_test)[:, 1],"g":"test"})
dt_p=pd.concat([df_val,df_test])
R.ggsave(filename="auc.pdf", plot=R.roc_r(dt_p), height = 100, width = 100, unit='mm')
res=R.dca_r(dt_p['g'].unique(),dt_p)
R.ggsave(filename="dca.pdf", plot=R.plot_ntbft(res), height = 100, width = 100, unit='mm')

#---------------------------------------------------------------------------训练和测试组多模型比较95%CI
models = get_models(params)
P = train_predict(models, X_train, X_test, y_train, y_test)
P.columns = score_models(P, y_test)

P_m = P.melt()
P_m["y"] = y_test.to_list() * len(models)
P_m.columns = ["g", "p", "y"]
R.ggsave(filename="all_model.pdf", plot=R.roc_r(P_m), height=100, width=100, unit='mm')

# 获取模型并计算性能指标和置信区间
models = get_models(params)
model_metrics = model_metrics_test(models, X_train, X_test, y_train, y_test)

# 打印每个模型的性能指标和95%置信区间
for model_name, metrics in model_metrics.items():
    print(f"Model: {model_name}")
    print("Train Metrics:")
    for metric_name, metric_value in metrics['Train Metrics'].items():
        ci_lower, ci_upper = metrics['Train 95% CI'][metric_name]
        print(f"  {metric_name}: {metric_value:.3f} (95% CI: {ci_lower:.3f} - {ci_upper:.3f})")
    print("Test Metrics:")
    for metric_name, metric_value in metrics['Test Metrics'].items():
        ci_lower, ci_upper = metrics['Test 95% CI'][metric_name]
        print(f"  {metric_name}: {metric_value:.3f} (95% CI: {ci_lower:.3f} - {ci_upper:.3f})")
    print()

# 重新定义 P 以便计算相关矩阵
P = pd.DataFrame({model_name: model.predict_proba(X_test)[:, 1] for model_name, model in models.items()})
corrmat_p = corrmat(P.corr(), inflate=False)



#-------------------------------------------------------------------------force_plot
X = df.drop(columns=['ID', 'CKMB','ALT','AST','UA', 'intraoperative_urine', 'aki_serious','aki1_3','natriuretic_peptide','CRRT','CRRT_time','ICU_time','death','MIN_urine', 'MAX_urine' ])
y = df['aki1_3'].astype(int)  

X_train,X_test,y_train,y_test =train_test_split(X,y,random_state=45,test_size=0.2) #修改为最有random_state=i

clf = lgb.LGBMClassifier(max_depth=5, num_leaves=14, learning_rate=0.08, n_estimators=288, verbose=-1, force_col_wise=True) #修改为gbm.best_params
clf.fit(X_train, y_train)

explainer = shap.TreeExplainer(clf)
val_X=X
shap_values =  explainer.shap_values(val_X)

predicted_probabilities = clf.predict_proba(X_train)[:, 1] # 计算所有患者的预测概率
# 将NumPy数组转换为Pandas DataFrame
predicted_probabilities_df = pd.DataFrame({'predicted_probabilities': predicted_probabilities})

# 保存为CSV文件
# predicted_probabilities_df.to_csv('predicted_probabilities.csv', encoding='utf_8_sig', index=False)

similar_patients = (predicted_probabilities >= 0.00000000000000000001) & (predicted_probabilities <= 0.1) # 筛选具有相似预测概率的患者

# 创建一个PDF文件
with PdfPages('force_plots.pdf') as pdf:
    for patient_index in similar_patients.nonzero()[0]:
        fig = plt.figure() # 创建一个Matplotlib图形对象

        # 生成force plot图
        force_plot = shap.force_plot(explainer.expected_value[1], shap_values[1][patient_index], X_train.iloc[patient_index])
        
        # 将force plot图添加到图形对象中
        shap.force_plot(explainer.expected_value[1], shap_values[1][patient_index], X_train.iloc[patient_index], matplotlib=True)
        pdf.savefig(fig) # 将图形对象保存到PDF文件中
        plt.close(fig)  # 关闭当前图形对象

similar_patients = (predicted_probabilities >= 0.9) & (predicted_probabilities <= 0.99999999) # 筛选具有相似预测概率的患者

# 创建一个PDF文件
with PdfPages('force_plots.pdf') as pdf:
    for patient_index in similar_patients.nonzero()[0]:
        fig = plt.figure() # 创建一个Matplotlib图形对象

        # 生成force plot图
        force_plot = shap.force_plot(explainer.expected_value[1], shap_values[1][patient_index], X_train.iloc[patient_index])
        
        # 将force plot图添加到图形对象中
        shap.force_plot(explainer.expected_value[1], shap_values[1][patient_index], X_train.iloc[patient_index], matplotlib=True)
        pdf.savefig(fig) # 将图形对象保存到PDF文件中
        plt.close(fig)  # 关闭当前图形对象

#--------------------------------------------------------------------------决策图
similar_patients_indices = np.where((predicted_probabilities >= 0.00000000000001) & (predicted_probabilities <= 0.1))[0]

zip_filename = "decision_plots_top10.zip"
with zipfile.ZipFile(zip_filename, "w") as zip_file:
    for patient_index in similar_patients_indices:
        # Get the top 10 features for the current patient
        top10_features_idx = np.argsort(np.abs(shap_values[1][patient_index]))[::-1][:10]
        
        # Plot decision plot for the top 10 features
        shap.decision_plot(explainer.expected_value[1], shap_values[1][patient_index][top10_features_idx], 
                           X_train.iloc[patient_index, top10_features_idx], show=False)
        
        plt.savefig("decision_plot.png", format="png")
        zip_file.write("decision_plot.png", f"decision_plot_{patient_index}.png")
        plt.close()
        os.remove("decision_plot.png")

similar_patients_indices = np.where((predicted_probabilities >= 0.9) & (predicted_probabilities <= 0.999999999))[0]

zip_filename = "decision_plots_top10_dayu0.9.zip"
with zipfile.ZipFile(zip_filename, "w") as zip_file:
    for patient_index in similar_patients_indices:
        # Get the top 10 features for the current patient
        top10_features_idx = np.argsort(np.abs(shap_values[1][patient_index]))[::-1][:10]
        
        # Plot decision plot for the top 10 features
        shap.decision_plot(explainer.expected_value[1], shap_values[1][patient_index][top10_features_idx], 
                           X_train.iloc[patient_index, top10_features_idx], show=False)
        
        plt.savefig("decision_plot.png", format="png")
        zip_file.write("decision_plot.png", f"decision_plot_{patient_index}.png")
        plt.close()
        os.remove("decision_plot.png")

#---------------------------------------------------------------------------交互图
X = df.drop(columns=['ID', 'CKMB','ALT','AST','UA', 'intraoperative_urine', 'aki_serious','aki1_3','natriuretic_peptide','CRRT','CRRT_time','ICU_time','death','MIN_urine', 'MAX_urine' ])
y = df['aki1_3'].astype(int)  

X_train,X_test,y_train,y_test =train_test_split(X,y,random_state=45,test_size=0.2) #修改为最有random_state=i

clf = lgb.LGBMClassifier(max_depth=5, num_leaves=14, learning_rate=0.08, n_estimators=288, verbose=-1, force_col_wise=True) #修改为gbm.best_params
clf.fit(X_train, y_train)
explainer = shap.TreeExplainer(clf)
val_X = X
shap_values = explainer.shap_values(val_X)
interaction_values = explainer.shap_interaction_values(val_X)
shap.summary_plot(interaction_values, X, max_display=10, show=False, plot_size=[10, 10])
plt.savefig('interaction_plot.pdf', format="pdf", bbox_inches="tight")

#---------------------------------------------------------------------------依赖图
def generate_dependency_plot(shap_values, X, feature1, feature2, cmap=red_blue):
    # 获取特征索引
    feature1_index = X.columns.get_loc(feature1)
    feature2_index = X.columns.get_loc(feature2)

    # 获取特征名称
    feature1_name = X.columns[feature1_index]
    feature2_name = X.columns[feature2_index]

    # 获取特征值
    feature1_values = X[feature1].values
    shap_feature1_values = shap_values_class1[:, feature1_index]

    # 根据传递的颜色风格绘制散点图
    plt.scatter(feature1_values, shap_feature1_values, c=X[feature2], cmap=red_blue, alpha=0.7, s=3)

    # 添加颜色条
    colorbar = plt.colorbar(label=feature2_name)

    plt.xlabel(feature1_name)
    plt.ylabel(f'{feature1_name} SHAP Value')
    plt.title(f'Dependency Plot: {feature1_name} vs {feature2_name} SHAP Value')

    plt.savefig(f'dependence_plot_{feature1}_{feature2}.pdf', format="pdf", dpi=1000, bbox_inches="tight")
    plt.close()


clf = lgb.LGBMClassifier(max_depth=5, num_leaves=14, learning_rate=0.08, n_estimators=288, verbose=-1, force_col_wise=True)
clf.fit(X_train, y_train)
explainer = shap.TreeExplainer(clf)
val_X = X
shap_values = explainer.shap_values(val_X)  # 获取 SHAP 值

# 选择类别 1 的 SHAP 值
shap_values_class1 = shap_values[1]

shap_values_pos_class = shap_values[1]  # 假定是二分类问题，并且我们关注的是正类

# 计算每个特征对正类预测的平均绝对SHAP值
mean_abs_shap_values = np.abs(shap_values_pos_class).mean(axis=0)

# 确定平均绝对SHAP值最高的前10个特征
top_10_indices = np.argsort(-mean_abs_shap_values)[:10]
top_10_features = X.columns[top_10_indices].tolist()

# 创建一个名为 "dependency_plots_Top10.zip" 的压缩包
with zipfile.ZipFile('dependency_plots_Top10.zip', 'w') as zipf:
    for feature1, feature2 in itertools.permutations(top_10_features, 2):
        generate_dependency_plot(shap_values_class1, X, feature1, feature2)
        zipf.write(f'dependence_plot_{feature1}_{feature2}.pdf', arcname=f'{feature1}_{feature2}.pdf')

# 删除生成的图像文件
for feature1, feature2 in itertools.permutations(top_10_features, 2):
    os.remove(f'dependence_plot_{feature1}_{feature2}.pdf')


#---------------------------------------------------------------------logistic回归样本量计算
import numpy as np
from scipy.stats import norm
import statsmodels.api as sm


X = df.drop(columns=['ID', 'CKMB','ALT','AST','UA', 'intraoperative_urine', 'aki_serious','aki1_3','natriuretic_peptide','CRRT','CRRT_time','ICU_time','death','MIN_urine', 'MAX_urine' ])
y = df['aki1_3'].astype(int)  

X_train,X_test,y_train,y_test =train_test_split(X,y,random_state=45,test_size=0.2) #修改为最有random_state=i

clf = lgb.LGBMClassifier(max_depth=5, num_leaves=14, learning_rate=0.08, n_estimators=288, verbose=-1, force_col_wise=True) #修改为gbm.best_params

cv2 = StratifiedKFold(n_splits=10, shuffle=True,random_state=45)  #10折交叉验证
clf.fit(X_train, y_train)

# 使用已训练好的模型 clf 对所有样本进行预测
all_samples_prob = clf.predict_proba(X)[:, 1]

# 将预测概率添加到 DataFrame 中
df['prob_aki1_3'] = all_samples_prob

df = df[['prob_aki1_3', 'aki1_3']]

# 假设的参数
alpha = 0.05
power = 0.8
effect_size = 0.10

# 准备数据
X = df['prob_aki1_3']
y = df['aki1_3']
X = sm.add_constant(X)  # 添加截距项

# 计算 Logistic 回归模型
logit_model = sm.Logit(y, X)
result = logit_model.fit()

# 查看回归模型的摘要
print(result.summary())

# 计算所需样本量
z_alpha = np.abs(norm.ppf(alpha / 2))
z_power = np.abs(norm.ppf(power))
std_err = result.bse['prob_aki1_3']  # prob_CKD 列的标准误差
n = ((z_alpha + z_power) ** 2) * (std_err ** 2) / (effect_size ** 2)

# 输出所需样本量
print(f"Estimated Required Sample Size: {int(np.ceil(n))}")
