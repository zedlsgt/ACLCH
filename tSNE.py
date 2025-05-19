from hdf5storage import loadmat
from sklearn import manifold,datasets
# inp_glove6B = loadmat('embedding\mirflickr-inp-glove6B.mat')["inp"]
imgs = loadmat('result\\mirflickr_imgs_m.mat')["imgs"]
txts = loadmat('result\\mirflickr_txts_m.mat')["txts"]
# mlabels = loadmat('result\\mirflickr_mlabels.mat')["mlabels"]
# print(inp_glove6B.shape)
tsne1 = manifold.TSNE(perplexity=30,n_components=2, init='pca', random_state=501)
X1 = tsne1.fit_transform(imgs)
print("imgs降维完成！")
X2 = tsne1.fit_transform(txts)
print("txts降维完成！")
# X3 = tsne1.fit_transform(mlabels)
print("mlabels降维完成！")
# print(X1.shape)
# file1= open(r'D:\code\dp\ACLCH-clip\result\inp降维.txt', 'w',encoding='UTF-8')
file1 = open(r'D:\code\dp\ACLCH-clip\result\\NUS-WIDE_imgs降维_m30.txt', 'w',encoding='UTF-8')
file2 = open(r'D:\code\dp\ACLCH-clip\result\\NUS-WIDE_txts降维_m30.txt', 'w',encoding='UTF-8')
# file3 = open(r'D:\code\dp\ACLCH-clip\result\\NUS-WIDE_mlabels降维_m30.txt', 'w',encoding='UTF-8')
for i in range (len (X1)):
    file1.write(str(X1 [i])+'\n')
file1.close()

for i in range (len (X2)):
    file2.write(str(X2 [i])+'\n')
file2.close()

# for i in range (len (X3)):
#     file3.write(str(X3 [i])+'\n')
# file3.close()
