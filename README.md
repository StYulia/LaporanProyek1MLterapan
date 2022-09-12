# Laporan Proyek Machine Learning – Siti Yulianingsih
## Domain Proyek
Pertumbuhan penduduk yang terus meningkat mendorong tingginya kebutuhan masyarakat akan tempat tinggal. Tempat tinggal atau rumah merupakan salah satu dari banyaknya kebutuhan primer bagi manusia.  Maka dari itu sangat penting untuk membuat perencanaan agar tiap keluarga dapat memiliki tempat tinggal pribadi.  Dalam perencanaan tersebut dibutuhkan prediksi atau perkiraan harga di masa mendatang.  Tiap manusia membutuhkan rumah untuk tempat berlindung dan sebagai tempat berkumpul dan berlangsungnya aktivitas keluarga, sekaligus sebagai sarana investasi. Fungsi rumah juga telah berubah, dari yang semula hanya sekedar sebagai tempat berlindung. Kini sebuah rumah tak cukup hanya untuk berteduh namun juga dituntut untuk mengakomodir kebutuhan dan keinginan pemiliknya. Seperti luas lahan, luas bangunan  berdiri,  banyaknya  ruangan,  hingga  ketersediaan  tempat  parkir  mobil.

Harga adalah salah satu hal yang dipertimbangkan oleh pembeli rumah oleh masyarakat. menurut penelitian yang dibuat oleh Agustinus Primnanda Alasan masyarakat mempertimbangkan faktor harga karena hal tersebut berkaitan dengan pendapatan mereka. Bagi mereka yang memiliki pendapatan besar mungkin harga tidak akan menjadi masalah, tapi mereka lebih mempertimbangkan luas dan kualitas produk dalam hal ini faktor bangunan.
Dengan melihat kondisi semacam ini mendorong produsen untuk melebarkan sayapnya di bidang perumahan. Maka tidak mengherankan jika akhirakhir ini bisnis di bidang perumahan semakin marak, banyak perusahaan muncul dengan memberikan berbagai macam fasilitas dalam menawarkan produknya. Perkembangan bisnis perumahan semakin marak dewasa ini, tidak hanya terpusat di kota-kota besar akan tetapi sudah meluas di kota-kota kecil.
penelitian agustinus:[FAKTOR-FAKTOR YANG MEMPENGARUHI KONSUMEN DALAM MEMBELI RUMAH)(http://eprints.undip.ac.id/23081/)

## Business Understanding
### Problem Statements
Suatu perusahaan harus selalu survive agar dapat terus bersaing dengan perusahaan-perusahaan sejenis yang sama-sama bergerak dalam bisnis perumahan. Karena banyaknya perusahaan yang bergerak di bidang perumahan, maka perusahaan harus mengenali apa yang mempengaruhi harga jual pada perumahan lain. Jangan sampai kita yang seharusnya mendapat keuntungan karna terdapat kualitas yang bagus pada perumahan kita, kita malah menjualnya dengan harga rendah.
Berdasarkan kondisi yang telah diuraikan sebelumnya, perusahaan akan mengembangkan sebuah sistem prediksi harga rumah untuk menjawab permasalahan berikut:
- apa fitur yang paling mempengaruhi harga rumah?
- Berapa harga rumah pada perusahaan lain dengan karakteristik atau fitur tertentu
### Goals
- Mengetahui fitur yang paling berkorelasi dengan harga rumah.
- Membuat model machine learning yang dapat memprediksi harga rumah berdasarkan fitur-fitur yang ada

**Metodologi**
Prediksi harga adalah tujuan yang ingin dicapai. Seperti yang kita tahu, harga merupakan variabel kontinu. Dalam predictive analytics, saat membuat prediksi variabel kontinu artinya Anda sedang menyelesaikan permasalahan regresi. Oleh karena itu, metodologi pada proyek ini adalah: membangun model regresi dengan harga diamonds sebagai target.
**Metrik**
Pengembangan model akan menggunakan beberapa algoritma machine learning yaitu K-Nearest Neighbor, Random Forest, dan Boosting Algorithm. Dari ketiga model ini, akan dipilih satu model yang memiliki nilai kesalahan prediksi terkecil.

## Data Understanding
Data yang akan digunakan pada proyek kali ini adalah housePrice dataset. Dataset ini memiliki 3.474 sampel data dengan berbagai kualitas atau karakteristik dan harga. Karakteristik yang dimaksud di sini adalah fitur non-numerik seperti Parking, Warehouse, Elevator, serta fitur numerik seperti Room dan  Area. Kesembilan fitur ini adalah fitur yang akan Anda gunakan dalam menemukan pola pada data, sedangkan harga merupakan fitur target.
Adapun uraikanlah seluruh variabel atau fitur pada data, sebagai berikut:
Parking: adalah keterangan apakah rumah tersebut tersedia tempat parker atau tidak
Warehouse: berisi informasi apakah rumah terdapat Gudang atau tidak
Elevator: informasi apakah rumah terdapat tangga  atau tidak
Room: informasi yang berisi ada berapa ruangan pada rumah tersebut
Area: informasi mengenai luas bangunan
Price: informasi harga

Dataset dapat di unduh pada link berikut: [housePrice dataset](https://www.kaggle.com/datasets/mokar2001/house-price-tehran-iran)

- hal pertama yang dilakukan adalah import library yang dibutuhkan
- lalu melakukan Exploratory Data Analysis, yang bertujuan  untuk mengetahui apakah tipe data pada setiap kolom sudah sesuai atau belum
- setelah itu menangani missing value jika ada. Pada dataset kali ini terdapat 10 missing value pada kolom room. Karena 10 missing value merupakan jumlah yang kecil jika dibandingkan dengan jumlah total sampel yaitu 3.474. jadi kita hapus saja 10 sampel ini, karena kita akan kehilangan beberapa informasi.
![This is an image](https://drive.google.com/file/d/1ePbWaDOmj1K6TX0CAGBQjBZk8r-_9Ehp/view?usp=sharing)
-	Selanjutnya membagi fitur menjadi dua bagian dengan proses Univariate Analysis. Dengan code :
numerical_features = [ 'Area', 'Room', 'Price']
categorical_features = ['Parking', 'Warehouse', 'Elevator']
-	Melakukan Exploratory Data Analysis, untuk menunjukkan hubungan antara dua atau lebih variabel pada data. Pada kasus kali ini kita melakukan nya pada fitur katagori yaitu parking, warehouse, elevator, dan pada fitur numerik yaitu harga.

## Data Preparation
-	Kita memiiki tiga variable kategori yaitu parking, warehouse, dan elevator. Untuk melakukan proses encoding fitur kategori agar menjadi variable numerik, salah satu teknik yang umum dilakukan adalah teknik one-hot-encoding
-	Selanjutnya membagi dataset menjadi data train dan data test agar dapat mempertahankan data yang ada menguji seberapa baik generalisasi model terhadap data baru
-	 Kemudia kita perlu melakukan standarisasi pada data train untuk menhindari kebpgpran informasi pada data test

## Modeling
Pada tahap modelling menggunaka  tiga model yaitu: K-Nearest Neighbor (KNN), Random Forest (RF), dan  Boosting Algorithm.
-	KKN

        models = pd.DataFrame(index=['train_mse', 'test_mse'], 
                      
                      columns=['KNN', 'RandomForest', 'Boosting'])
                      
        from sklearn.neighbors import KNeighborsRegressor
        
        from sklearn.metrics import mean_squared_error
        
        knn = KNeighborsRegressor(n_neighbors=10)
        
        knn.fit(X_train, y_train)
        
        models.loc['train_mse','knn'] = mean_squared_error(y_pred = knn.predict(X_train), y_true=y_train)

pemodelan KKN adalah algoritma yang relatif sederhana dibandingkan dengan algoritma lain sehingga sangat mudah dipahami, namun ia memiliki kekurangan jika dihadapkan pada jumlah fitur atau dimensi yang besar. permasalahan ini muncul ketika jumlah sampel meningkat secara eksponensial seiring dengan jumlah dimensi (fitur) pada data
-	Random Forest
from sklearn.ensemble import RandomForestRegressor
 
RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
RF.fit(X_train, y_train)
 
models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)

random forest adalah salah satu algoritma supervised learning yang dapat digunakan untuk menyelesaikan masalah klasifikasi dan regresi. Random forest juga cukup sederhana tetapi memiliki stabilitas yang mumpuni. Namun random forest  tidak akan memberikan hasil maksimal ketika data yang kita pakai sangat jarang.
-	Boosting Algorithm
from sklearn.ensemble import AdaBoostRegressor
boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)
boosting.fit(X_train, y_train)
models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train), y_true=y_train)

boosting algorithm dapat meningkatkan performa atau akurasi prediksi. Namun hal ini tetap bergantung pada kasus per kasus, ruang lingkup masalah, dan dataset yang digunakan

## Evaluation
Pada evaluasi model kali ini menggunakan metrik MSE. Untuk menghitung model MSE pada model kita perlu melakukan proses scalling fitur nuerik pada data test gar skala antara data latih dan data uji sama dan kita bisa melakukan evaluasi.
Untuk proses scaling, perlu menjalankan code berikut:
X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])

Jika sudah lakukan evaluasi model dengan metrik MSE, yang di dapatkan hasi evaluasi pada data train dan data test berikut:


