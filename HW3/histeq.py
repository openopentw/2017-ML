from PIL import Image
import numpy as np

file = open("data/train.csv","r")
train = []
for lines in file.readlines():
    train += lines.replace(","," ").split()
train = np.array(train[2::],dtype=float)
train = train.reshape((28709,2305))
x_train = train[:,1::]
# img = np.loadtxt("1.txt",dtype=int)

print("start")
for a in range(28709):
    print(a)
    img = x_train[a]
    tmp = np.reshape(img,(48,48))
    im = Image.new("L",(48,48),0)
    for i in range(48):
        for j in range(48):
            im.putpixel((i,j),int(tmp[i][j]))
    filename = "image/origin/"+str(a)+".png"
    im.save(filename)


    imhist, bins = np.histogram(img,256,normed=True)
    cdf = imhist.cumsum()
    cdf = 255*cdf/cdf[-1]
    im2 = np.interp(img,bins[:-1],cdf)
    # print(im2.shape)
    im2 = im2.reshape(48,48)

    im = Image.new("L",(48,48),0)
    for i in range(48):
        for j in range(48):
            im.putpixel((i,j),int(im2[i][j]))
    filename = "image/histeq/"+str(a)+".png"
    im.save(filename)
