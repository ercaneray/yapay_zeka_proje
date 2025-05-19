import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

file_path = 'C:\Users\ufuk0\Desktop\yapay_zeka\SEER Breast Cancer Dataset .csv'  # Kullanıcının sağladığı dosya yolu
data = pd.read_csv(file_path)

print(data.head())
print("Sütunlar:", data.columns.tolist())
print("Data Types:\n", data.dtypes)
print("Eksik Veriler:\n", data.isnull().sum())
# 'Unnamed: 3' sütununu kaldırma
data_cleaned = data.drop(columns=["Unnamed: 3"], errors='ignore')

def plot_distributions(dataframe, columns, num_bins=20):

    for column in columns:
        if dataframe[column].dtype in ['int64', 'float64']:  # Sadece sayısal sütunları seç
            plt.figure(figsize=(8, 4))  # Şekil boyutu
            plt.hist(dataframe[column], bins=num_bins, alpha=0.7, color='blue', edgecolor='black')
            plt.title(f'Distribution of {column}')  # Grafik başlığı
            plt.xlabel(column)  # X ekseni etiketi
            plt.ylabel('Frequency')  # Y ekseni etiketi
            plt.grid(axis='y')  # Y ekseninde ızgara çizgileri ekle
            plt.show()

# Sayısal sütunların listesini çıkarma
numerical_columns = data_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Sayısal sütunların dağılımlarını görselleştirme
plot_distributions(data_cleaned, numerical_columns)

numerical_data = data_cleaned.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numerical_data.corr()

plt.figure(figsize=(10, 8)) 
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title("Korelasyon Matrisi")
plt.show()

model = LogisticRegression(max_iter=1000, random_state=42)

# Bağımsız ve bağımlı değişkenleri ayırma
ozellikler = data_cleaned.drop(columns=["Status"])  # Bağımsız değişkenler
hedef = data_cleaned["Status"]  # Bağımlı değişken (hedef)

# Sayısal özellikleri seçme (RFE sadece sayısal verilerle çalışır)
ozellikler_sayisal = ozellikler.select_dtypes(include=['int64', 'float64'])

# Recursive Feature Elimination (RFE) işlemi
rfe = RFE(estimator=model, n_features_to_select=5)  # 5 en önemli özelliği seçmek için
rfe.fit(ozellikler_sayisal, hedef)

# En önemli özellikleri seçme
secilen_ozellikler = ozellikler_sayisal.columns[rfe.support_]  # Desteklenen (önemli) sütunlar
onem_siralamasi = dict(zip(ozellikler_sayisal.columns, rfe.ranking_))  # Özelliklerin sıralaması

# Sonuçları yazdırma
print("Seçilen Özellikler:")
print(secilen_ozellikler.tolist())  # Önemli özelliklerin listesi

print("\nÖzellik Önem Sıralaması:")
print(onem_siralamasi)  # Tüm özelliklerin önem sıralaması

print(data_cleaned.columns.tolist())
data_cleaned.columns = [col.strip() for col in data_cleaned.columns]

print(data_cleaned.columns)

ozellikler = data_cleaned[["Age", "Tumor Size", "Regional Node Examined", "Reginol Node Positive", "Survival Months"]]
hedef = data_cleaned["Status"].apply(lambda x: 1 if x == "Alive" else 0)

# 2. Eğitim ve test veri setlerini oluşturma
X_train, X_test, y_train, y_test = train_test_split(ozellikler, hedef, test_size=0.2, random_state=42)

# 3. Lojistik Regresyon Modeli
lojistik_model = LogisticRegression(max_iter=1000, random_state=42)
lojistik_model.fit(X_train, y_train)  # Modeli eğitme

# 4. Test veri setiyle tahmin yapma
y_tahmin = lojistik_model.predict(X_test)
y_olasilik = lojistik_model.predict_proba(X_test)[:, 1]  # Pozitif sınıf olasılıkları

# 5. Performans Metriklerini Hesaplama
dogruluk = accuracy_score(y_test, y_tahmin)
f1_skoru = f1_score(y_test, y_tahmin)
kesinlik = precision_score(y_test, y_tahmin)
hassasiyet = recall_score(y_test, y_tahmin)
roc_auc = roc_auc_score(y_test, y_olasilik)

# 6. ROC Eğrisi Çizimi
fpr, tpr, _ = roc_curve(y_test, y_olasilik)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Eğrisi (AUC = {roc_auc:.2f})", color="blue")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Rastgele Tahmin")
plt.xlabel("Yanlış Pozitif Oranı (FPR)")
plt.ylabel("Doğru Pozitif Oranı (TPR)")
plt.title("ROC Eğrisi - Lojistik Regresyon")
plt.legend()
plt.grid()
plt.show()

# 7. Sonuçları Görüntüleme
print({
    "Doğruluk (Accuracy)": dogruluk,
    "F1 Skoru": f1_skoru,
    "Kesinlik (Precision)": kesinlik,
    "Hassasiyet (Recall)": hassasiyet,
    "ROC AUC": roc_auc
})

karar_agaci_model = DecisionTreeClassifier(random_state=42)
karar_agaci_model.fit(X_train, y_train)  # Modeli eğitme

# 2. Test veri setiyle tahmin yapma
y_tahmin_karar_agaci = karar_agaci_model.predict(X_test)
y_olasilik_karar_agaci = karar_agaci_model.predict_proba(X_test)[:, 1]

# 3. Performans Metriklerini Hesaplama
dogruluk_karar_agaci = accuracy_score(y_test, y_tahmin_karar_agaci)
f1_skoru_karar_agaci = f1_score(y_test, y_tahmin_karar_agaci)
kesinlik_karar_agaci = precision_score(y_test, y_tahmin_karar_agaci)
hassasiyet_karar_agaci = recall_score(y_test, y_tahmin_karar_agaci)
roc_auc_karar_agaci = roc_auc_score(y_test, y_olasilik_karar_agaci)

# 4. Performans Sonuçlarını Görüntüleme
print({
    "Doğruluk (Accuracy)": dogruluk_karar_agaci,
    "F1 Skoru": f1_skoru_karar_agaci,
    "Kesinlik (Precision)": kesinlik_karar_agaci,
    "Hassasiyet (Recall)": hassasiyet_karar_agaci,
    "ROC AUC": roc_auc_karar_agaci
})

# 5. ROC Doğrusu Çizimi
fpr_karar_agaci, tpr_karar_agaci, _ = roc_curve(y_test, y_olasilik_karar_agaci)
plt.figure(figsize=(8, 6))
plt.plot(fpr_karar_agaci, tpr_karar_agaci, label=f"ROC Eğrisi (AUC = {roc_auc_karar_agaci:.2f})", color="green")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Rastgele Tahmin")
plt.xlabel("Yanlış Pozitif Oranı (FPR)")
plt.ylabel("Doğru Pozitif Oranı (TPR)")
plt.title("ROC Eğrisi - Karar Ağaçları")
plt.legend()
plt.grid()
plt.show()

random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)  # 100 ağaç ile model
random_forest_model.fit(X_train, y_train)  # Eğitim veri seti ile modeli eğitme

# 2. Test Veri Seti ile Tahmin
y_tahmin_rf = random_forest_model.predict(X_test)  # Tahmin sınıfları
y_olasilik_rf = random_forest_model.predict_proba(X_test)[:, 1]  # Pozitif sınıf olasılıkları

# 3. Performans Metriklerini Hesaplama
dogruluk_rf = accuracy_score(y_test, y_tahmin_rf)  # Doğruluk
f1_skoru_rf = f1_score(y_test, y_tahmin_rf)  # F1 Skoru
kesinlik_rf = precision_score(y_test, y_tahmin_rf)  # Kesinlik
hassasiyet_rf = recall_score(y_test, y_tahmin_rf)  # Hassasiyet
roc_auc_rf = roc_auc_score(y_test, y_olasilik_rf)  # ROC AUC

# 4. Performans Sonuçlarını Görüntüleme
print({
    "Doğruluk (Accuracy)": dogruluk_rf,
    "F1 Skoru": f1_skoru_rf,
    "Kesinlik (Precision)": kesinlik_rf,
    "Hassasiyet (Recall)": hassasiyet_rf,
    "ROC AUC": roc_auc_rf
})

# 5. ROC Eğrisi Çizimi
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_olasilik_rf)  # Yanlış ve doğru pozitif oranları
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f"ROC Eğrisi (AUC = {roc_auc_rf:.2f})", color="purple")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Rastgele Tahmin")
plt.xlabel("Yanlış Pozitif Oranı (FPR)")
plt.ylabel("Doğru Pozitif Oranı (TPR)")
plt.title("ROC Eğrisi - Random Forest")
plt.legend()
plt.grid()
plt.show()

svm_model = SVC(probability=True, random_state=42)  # Sınıf olasılıkları için probability=True
svm_model.fit(X_train, y_train)  # Modeli eğitme

# 2. Test Veri Seti ile Tahmin
y_tahmin_svm = svm_model.predict(X_test)  # Sınıf tahminleri
y_olasilik_svm = svm_model.predict_proba(X_test)[:, 1]  # Pozitif sınıf olasılıkları

# 3. Performans Metriklerini Hesaplama
dogruluk_svm = accuracy_score(y_test, y_tahmin_svm)  # Doğruluk
f1_skoru_svm = f1_score(y_test, y_tahmin_svm)  # F1 Skoru
kesinlik_svm = precision_score(y_test, y_tahmin_svm)  # Kesinlik
hassasiyet_svm = recall_score(y_test, y_tahmin_svm)  # Hassasiyet
roc_auc_svm = roc_auc_score(y_test, y_olasilik_svm)  # ROC AUC

# 4. Performans Sonuçlarını Görüntüleme
print({
    "Doğruluk (Accuracy)": dogruluk_svm,
    "F1 Skoru": f1_skoru_svm,
    "Kesinlik (Precision)": kesinlik_svm,
    "Hassasiyet (Recall)": hassasiyet_svm,
    "ROC AUC": roc_auc_svm
})

# 5. ROC Eğrisi Çizimi
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_olasilik_svm)  # Yanlış ve doğru pozitif oranları
plt.figure(figsize=(8, 6))
plt.plot(fpr_svm, tpr_svm, label=f"ROC Eğrisi (AUC = {roc_auc_svm:.2f})", color="orange")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Rastgele Tahmin")
plt.xlabel("Yanlış Pozitif Oranı (FPR)")
plt.ylabel("Doğru Pozitif Oranı (TPR)")
plt.title("ROC Eğrisi - SVM")
plt.legend()
plt.grid()
plt.show()

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)  # Modeli oluşturma
xgb_model.fit(X_train, y_train)  # Eğitim veri seti ile modeli eğitme

# 2. Test Veri Seti ile Tahmin
y_tahmin_xgb = xgb_model.predict(X_test)  # Sınıf tahminleri
y_olasilik_xgb = xgb_model.predict_proba(X_test)[:, 1]  # Pozitif sınıf olasılıkları

# 3. Performans Metriklerini Hesaplama
dogruluk_xgb = accuracy_score(y_test, y_tahmin_xgb)  # Doğruluk
f1_skoru_xgb = f1_score(y_test, y_tahmin_xgb)  # F1 Skoru
kesinlik_xgb = precision_score(y_test, y_tahmin_xgb)  # Kesinlik
hassasiyet_xgb = recall_score(y_test, y_tahmin_xgb)  # Hassasiyet
roc_auc_xgb = roc_auc_score(y_test, y_olasilik_xgb)  # ROC AUC

# 4. Performans Sonuçlarını Görüntüleme
print({
    "Doğruluk (Accuracy)": dogruluk_xgb,
    "F1 Skoru": f1_skoru_xgb,
    "Kesinlik (Precision)": kesinlik_xgb,
    "Hassasiyet (Recall)": hassasiyet_xgb,
    "ROC AUC": roc_auc_xgb
})

# 5. ROC Eğrisi Çizimi
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_olasilik_xgb)  # Yanlış ve doğru pozitif oranları
plt.figure(figsize=(8, 6))
plt.plot(fpr_xgb, tpr_xgb, label=f"ROC Eğrisi (AUC = {roc_auc_xgb:.2f})", color="red")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Rastgele Tahmin")
plt.xlabel("Yanlış Pozitif Oranı (FPR)")
plt.ylabel("Doğru Pozitif Oranı (TPR)")
plt.title("ROC Eğrisi - XGBoost")
plt.legend()
plt.grid()
plt.show()

model_sonuclari = pd.DataFrame({
    "Model": ["Lojistik Regresyon", "Karar Ağaçları", "Random Forest", "SVM", "XGBoost"],
    "Doğruluk (Accuracy)": [0.884, 0.720, 0.899, 0.887, 0.841],
    "F1 Skoru": [0.933, 0.720, 0.941, 0.935, 0.938],
    "Kesinlik (Precision)": [0.900, 0.750, 0.922, 0.896, 0.915],
    "Hassasiyet (Recall)": [0.968, 0.690, 0.961, 0.978, 0.962],
    "ROC AUC": [0.878, 0.720, 0.872, 0.831, 0.840]
})

# Performans sonuçlarını tablo olarak görüntüleme
print(model_sonuclari)

# CSV dosyası olarak kaydetmek isterseniz:
model_sonuclari.to_csv("model_performans_karsilastirma.csv", index=False)

# X ve y'nin oluşturulması (Doğru sütun isimleri ile)
X = data_cleaned[["Age", "Tumor Size", "Regional Node Examined", "Reginol Node Positive", "Survival Months"]]
y = data_cleaned["Status"].apply(lambda x: 1 if x == "Alive" else 0)

# Eğitim ve test veri setlerini ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Grid Search için hiperparametre ayarları
parametreler = {
    "n_estimators": [50, 100, 200],  # Ağaç sayısı
    "max_depth": [None, 10, 20, 30],  # Maksimum derinlik
    "min_samples_split": [2, 5, 10],  # Minimum bölünme sayısı
    "min_samples_leaf": [1, 2, 4],  # Minimum yaprak örneği
}

# GridSearchCV oluşturma
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=parametreler,
    cv=5,  # 5 katlı çapraz doğrulama
    scoring="accuracy",
    verbose=2,
    n_jobs=-1  # Paralel işlem
)

# Grid Search işlemini başlatma
grid_search.fit(X_train, y_train)

# En iyi hiperparametreler ve doğruluk skoru
en_iyi_parametreler = grid_search.best_params_
en_iyi_skor = grid_search.best_score_

print("En İyi Parametreler:", en_iyi_parametreler)
print("En İyi Doğruluk Skoru:", en_iyi_skor)

# Optimizasyon sonrası modeli yeniden oluşturma
optimize_edilmis_model = RandomForestClassifier(**en_iyi_parametreler, random_state=42)
optimize_edilmis_model.fit(X_train, y_train)

# Test veri seti ile tahmin yapma
y_tahmin = optimize_edilmis_model.predict(X_test)
y_olasilik = optimize_edilmis_model.predict_proba(X_test)[:, 1]

# Performans Metriklerini Hesaplama
dogruluk = accuracy_score(y_test, y_tahmin)
f1 = f1_score(y_test, y_tahmin)
kesinlik = precision_score(y_test, y_tahmin)
hassasiyet = recall_score(y_test, y_tahmin)
roc_auc = roc_auc_score(y_test, y_olasilik)

# Performans Sonuçlarını Görüntüleme
print({
    "Doğruluk (Accuracy)": dogruluk,
    "F1 Skoru": f1,
    "Kesinlik (Precision)": kesinlik,
    "Hassasiyet (Recall)": hassasiyet,
    "ROC AUC": roc_auc
})

# ROC Eğrisi Çizimi
fpr, tpr, _ = roc_curve(y_test, y_olasilik)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Eğrisi (AUC = {roc_auc:.2f})", color="blue")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Rastgele Tahmin")
plt.xlabel("Yanlış Pozitif Oranı (FPR)")
plt.ylabel("Doğru Pozitif Oranı (TPR)")
plt.title("ROC Eğrisi - Optimize Edilmiş Random Forest")
plt.legend()
plt.grid()
plt.show()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),  # Giriş katmanı
    Dropout(0.3),  # Aşırı öğrenmeyi engellemek için Dropout
    Dense(32, activation='relu'),  # Gizli katman
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Çıkış katmanı (Binary Classification)
])

# 6. Modeli Derleme
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 7. Modeli Eğitme
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1
)

# 8. Eğitim ve Doğrulama Sonuçlarını Görselleştirme
# Doğruluk grafiği
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Model Doğruluk Grafiği')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.grid()
plt.show()

# Kayıp grafiği
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Model Kayıp Grafiği')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.grid()
plt.show()

# 9. Test Veri Seti ile Değerlendirme
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Doğruluğu: {test_accuracy:.2f}")
print(f"Test Kaybı: {test_loss:.2f}")

