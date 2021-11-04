from keras.models import load_model
from keras.models import Sequential

test_path='?'  #可在此輸入test_image路徑
out_path='?'   #可在此輸入輸出路徑


model = Sequential()
model = load_model('filepath+model-resnet50-final.h5')


train = pd.read_csv('training_labels.txt', sep=" ", header=None)
train.columns = ["id", "label"]

y = train['label'].values


integer_mapping = {x: i for i,x in enumerate(y)}
vec = [integer_mapping[word] for word in y]
y = np.array(y)
label_encoder = LabelEncoder()
vec = label_encoder.fit_transform(y)

y = to_categorical(vec)
'''
抱歉以上這邊做得不夠好，當初在colab時並沒有做出另外一個檔案
來讀取model，所以這邊的y是原本讀取標籤內容所做的編號，
現在也需要用y的編號來重新輸出回新的txt內
'''
test = pd.read_csv('testing_img_order.txt', header=None)
test.columns = ["id"]

test_image = []
for i in tqdm(range(test.shape[0])):
    img = image.load_img(str(test_path) + str(test['id'][i]), target_size=(200,200,3)) #224
    img = image.img_to_array(img)
    img = img/255
    test_image.append(img)
    
test = np.array(test_image)


predict = model.predict(test) 
classes_x = np.argmax(predict,axis=1)


vec = label_encoder.inverse_transform(classes_x)


test = pd.read_csv('testing_img_order.txt', header=None)
test.columns = ["id"]
test['label'] = ''
for i in tqdm(range(test.shape[0])):
  #test['label'][i] = int(classes_x[i])
  test['label'][i] = vec[i]


test.to_csv(out_path+'answer-max.txt', sep=" " ,header=None, index=None)
