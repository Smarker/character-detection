
from os import listdir,read
from os.path import join,isfile
from PIL import Image

def crop_function(image,left,top,right,bottom,textname):
    image_url=image.strip()
    print(image_url)
    img=Image.open(image_url)
    # size=img.size
    box=(left,top,right,bottom)
    crop_area=img.crop(box)
    indice=0

    x=image_url.split(".")
    print(x)
    s="."
    y=x[1].split("/")
    l=len(y)
    i=0
    for i in range(l):
        if (i==l-1):
            s=s+"results/"+y[i]
        else:
            s=s+y[i]+"/"
    test=True
    while test:
        crop_url=s+"_"+textname+"_"+str(indice)+"."+x[2]
        try:
            with open(crop_url): pass
        except IOError:
            test=False
        indice+=1
    crop_area.save(crop_url)


mypath = './labels_crops'
for filename in listdir(mypath):
    if(isfile(mypath+'/'+filename)):
        imagefilename = './images/'+filename.split('.')[0]+'.jpg'
        filec = open(mypath+'/'+filename,'r')
        for line in filec.readlines():
            params = line.split(',')  
            crop_function(imagefilename,int(params[0]),int(params[1]),int(params[2]),int(params[3]),params[4].strip())
    else:
        print 'no files'    

